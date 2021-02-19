# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     split_at_heading: true
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# hide
# skip
from nbdev.export import notebook2script
from torch.autograd import Function
from torch.utils import cpp_extension
from nbdev.showdoc import *
from fastai.text.models.awdlstm import dropout_mask
from fastai.text.core import *
from fastai.data.all import *
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# export

# hide

# +
# all_cpp

# +
# default_exp text.models.qrnn
# default_cls_lvl 3
# -

# # QRNN
#
# > Quasi-recurrent neural networks introduced in [Bradbury et al.](https://arxiv.org/abs/1611.01576)

# ## ForgetMult

# export

__file__ = Path.cwd().parent / 'fastai' / 'text' / 'models' / 'qrnn.py'


# export
def load_cpp(name, files, path):
    os.makedirs(Config().model / 'qrnn', exist_ok=True)
    return cpp_extension.load(name=name, sources=[path / f for f in files], build_directory=Config().model / 'qrnn')


# export
class _LazyBuiltModule():
    "A module with a CPP extension that builds itself at first use"

    def __init__(self, name, files): self.name, self.files, self.mod = name, files, None

    def _build(self):
        self.mod = load_cpp(name=self.name, files=self.files, path=Path(__file__).parent)

    def forward(self, *args, **kwargs):
        if self.mod is None:
            self._build()
        return self.mod.forward(*args, **kwargs)

    def backward(self, *args, **kwargs):
        if self.mod is None:
            self._build()
        return self.mod.backward(*args, **kwargs)


# export
forget_mult_cuda = _LazyBuiltModule('forget_mult_cuda', ['forget_mult_cuda.cpp', 'forget_mult_cuda_kernel.cu'])
bwd_forget_mult_cuda = _LazyBuiltModule('bwd_forget_mult_cuda', ['bwd_forget_mult_cuda.cpp', 'bwd_forget_mult_cuda_kernel.cu'])


# export
def dispatch_cuda(cuda_class, cpu_func, x):
    "Depending on `x.device` uses `cpu_func` or `cuda_class.apply`"
    return cuda_class.apply if x.device.type == 'cuda' else cpu_func


# The ForgetMult gate is the quasi-recurrent part of the network, computing the following from `x` and `f`.
# ``` python
# h[i+1] = x[i] * f[i] + h[i] + (1-f[i])
# ```
# The initial value for `h[0]` is either a tensor of zeros or the previous hidden state.

# export
def forget_mult_CPU(x, f, first_h=None, batch_first=True, backward=False):
    "ForgetMult gate applied to `x` and `f` on the CPU."
    result = []
    dim = (1 if batch_first else 0)
    forgets = f.split(1, dim=dim)
    inputs = x.split(1, dim=dim)
    prev_h = None if first_h is None else first_h.unsqueeze(dim)
    idx_range = range(len(inputs) - 1, -1, -1) if backward else range(len(inputs))
    for i in idx_range:
        prev_h = inputs[i] * forgets[i] if prev_h is None else inputs[i] * forgets[i] + (1 - forgets[i]) * prev_h
        if backward:
            result.insert(0, prev_h)
        else:
            result.append(prev_h)
    return torch.cat(result, dim=dim)


# `first_h` is the tensor used for the value of `h[0]` (defaults to a tensor of zeros). If `batch_first=True`, `x` and `f` are expected to be of shape `batch_size x seq_length x n_hid`, otherwise they are expected to be of shape `seq_length x batch_size x n_hid`. If `backwards=True`, the elements in `x` and `f` on the sequence dimension are read in reverse.

# +
def manual_forget_mult(x, f, h=None, batch_first=True, backward=False):
    if batch_first:
        x, f = x.transpose(0, 1), f.transpose(0, 1)
    out = torch.zeros_like(x)
    prev = h if h is not None else torch.zeros_like(out[0])
    idx_range = range(x.shape[0] - 1, -1, -1) if backward else range(x.shape[0])
    for i in idx_range:
        out[i] = f[i] * x[i] + (1 - f[i]) * prev
        prev = out[i]
    if batch_first:
        out = out.transpose(0, 1)
    return out


x, f = torch.randn(5, 3, 20).chunk(2, dim=2)
for (bf, bw) in [(True, True), (False, True), (True, False), (False, False)]:
    th_out = manual_forget_mult(x, f, batch_first=bf, backward=bw)
    out = forget_mult_CPU(x, f, batch_first=bf, backward=bw)
    test_close(th_out, out)
    h = torch.randn((5 if bf else 3), 10)
    th_out = manual_forget_mult(x, f, h=h, batch_first=bf, backward=bw)
    out = forget_mult_CPU(x, f, first_h=h, batch_first=bf, backward=bw)
    test_close(th_out, out)
# -

x = torch.randn(3, 4, 5)
x.size() + torch.Size([0, 1, 0])


# export
class ForgetMultGPU(Function):
    "Wrapper around the CUDA kernels for the ForgetMult gate."
    @staticmethod
    def forward(ctx, x, f, first_h=None, batch_first=True, backward=False):
        ind = -1 if backward else 0
        (i, j) = (0, 1) if batch_first else (1, 0)
        output = x.new_zeros(x.shape[0] + i, x.shape[1] + j, x.shape[2])
        if first_h is not None:
            if batch_first:
                output[:, ind] = first_h
            else:
                output[ind] = first_h
        else:
            output.zero_()
        ctx.forget_mult = bwd_forget_mult_cuda if backward else forget_mult_cuda
        output = ctx.forget_mult.forward(x, f, output, batch_first)
        ctx.save_for_backward(x, f, first_h, output)
        ctx.batch_first = batch_first
        if backward:
            return output[:, :-1] if batch_first else output[:-1]
        else:
            return output[:, 1:] if batch_first else output[1:]

    @staticmethod
    def backward(ctx, grad_output):
        x, f, first_h, output = ctx.saved_tensors
        grad_x, grad_f, grad_h = ctx.forget_mult.backward(x, f, output, grad_output, ctx.batch_first)
        return (grad_x, grad_f, (None if first_h is None else grad_h), None, None)


# +
# hide
# cuda
# cpp
def detach_and_clone(t):
    return t.detach().clone().requires_grad_(True)


x, f = torch.randn(5, 3, 20).cuda().chunk(2, dim=2)
x, f = x.contiguous().requires_grad_(True), f.contiguous().requires_grad_(True)
th_x, th_f = detach_and_clone(x), detach_and_clone(f)
for (bf, bw) in [(True, True), (False, True), (True, False), (False, False)]:
    th_out = forget_mult_CPU(th_x, th_f, first_h=None, batch_first=bf, backward=bw)
    th_loss = th_out.pow(2).mean()
    th_loss.backward()
    out = ForgetMultGPU.apply(x, f, None, bf, bw)
    loss = out.pow(2).mean()
    loss.backward()
    test_close(th_out, out, eps=1e-4)
    test_close(th_x.grad, x.grad, eps=1e-4)
    test_close(th_f.grad, f.grad, eps=1e-4)
    for p in [x, f, th_x, th_f]:
        p = p.detach()
        p.grad = None
    h = torch.randn((5 if bf else 3), 10).cuda().requires_grad_(True)
    th_h = detach_and_clone(h)
    th_out = forget_mult_CPU(th_x, th_f, first_h=th_h, batch_first=bf, backward=bw)
    th_loss = th_out.pow(2).mean()
    th_loss.backward()
    out = ForgetMultGPU.apply(x.contiguous(), f.contiguous(), h, bf, bw)
    loss = out.pow(2).mean()
    loss.backward()
    test_close(th_out, out, eps=1e-4)
    test_close(th_x.grad, x.grad, eps=1e-4)
    test_close(th_f.grad, f.grad, eps=1e-4)
    test_close(th_h.grad, h.grad, eps=1e-4)
    for p in [x, f, th_x, th_f]:
        p = p.detach()
        p.grad = None


# -

# ## QRNN

# export
class QRNNLayer(Module):
    "Apply a single layer Quasi-Recurrent Neural Network (QRNN) to an input sequence."

    def __init__(self, input_size, hidden_size=None, save_prev_x=False, zoneout=0, window=1,
                 output_gate=True, batch_first=True, backward=False):
        assert window in [1, 2], "This QRNN implementation currently only handles convolutional window of size 1 or size 2"
        self.save_prev_x, self.zoneout, self.window = save_prev_x, zoneout, window
        self.output_gate, self.batch_first, self.backward = output_gate, batch_first, backward
        hidden_size = ifnone(hidden_size, input_size)
        # One large matmul with concat is faster than N small matmuls and no concat
        mult = (3 if output_gate else 2)
        self.linear = nn.Linear(window * input_size, mult * hidden_size)
        self.prevX = None

    def reset(self): self.prevX = None

    def forward(self, inp, hid=None):
        y = self.linear(self._get_source(inp))
        if self.output_gate:
            z_gate, f_gate, o_gate = y.chunk(3, dim=2)
        else:
            z_gate, f_gate = y.chunk(2, dim=2)
        z_gate, f_gate = z_gate.tanh(), f_gate.sigmoid()
        if self.zoneout and self.training:
            f_gate *= dropout_mask(f_gate, f_gate.size(), self.zoneout).requires_grad_(False)
        forget_mult = dispatch_cuda(ForgetMultGPU, partial(forget_mult_CPU), inp)
        c_gate = forget_mult(z_gate, f_gate, hid, self.batch_first, self.backward)
        output = torch.sigmoid(o_gate) * c_gate if self.output_gate else c_gate
        if self.window > 1 and self.save_prev_x:
            if self.backward:
                self.prevX = (inp[:, :1] if self.batch_first else inp[:1]) .detach()
            else:
                self.prevX = (inp[:, -1:] if self.batch_first else inp[-1:]).detach()
        idx = 0 if self.backward else -1
        return output, (c_gate[:, idx] if self.batch_first else c_gate[idx])

    def _get_source(self, inp):
        if self.window == 1:
            return inp
        dim = (1 if self.batch_first else 0)
        if self.batch_first:
            prev = torch.zeros_like(inp[:, :1]) if self.prevX is None else self.prevX
            if prev.shape[0] < inp.shape[0]:
                prev = torch.cat([prev, torch.zeros_like(inp[prev.shape[0]:, :1])], dim=0)
            if prev.shape[0] > inp.shape[0]:
                prev = prev[:inp.shape[0]]
        else:
            prev = torch.zeros_like(inp[:1]) if self.prevX is None else self.prevX
            if prev.shape[1] < inp.shape[1]:
                prev = torch.cat([prev, torch.zeros_like(inp[:1, prev.shape[0]:])], dim=1)
            if prev.shape[1] > inp.shape[1]:
                prev = prev[:, :inp.shape[1]]
        inp_shift = [prev]
        if self.backward:
            inp_shift.insert(0, inp[:, 1:] if self.batch_first else inp[1:])
        else:
            inp_shift.append(inp[:, :-1] if self.batch_first else inp[:-1])
        inp_shift = torch.cat(inp_shift, dim)
        return torch.cat([inp, inp_shift], 2)


qrnn_fwd = QRNNLayer(10, 20, save_prev_x=True, zoneout=0, window=2, output_gate=True)
qrnn_bwd = QRNNLayer(10, 20, save_prev_x=True, zoneout=0, window=2, output_gate=True, backward=True)
qrnn_bwd.load_state_dict(qrnn_fwd.state_dict())
x_fwd = torch.randn(7, 5, 10)
x_bwd = x_fwd.clone().flip(1)
y_fwd, h_fwd = qrnn_fwd(x_fwd)
y_bwd, h_bwd = qrnn_bwd(x_bwd)
test_close(y_fwd, y_bwd.flip(1), eps=1e-4)
test_close(h_fwd, h_bwd, eps=1e-4)
y_fwd, h_fwd = qrnn_fwd(x_fwd, h_fwd)
y_bwd, h_bwd = qrnn_bwd(x_bwd, h_bwd)
test_close(y_fwd, y_bwd.flip(1), eps=1e-4)
test_close(h_fwd, h_bwd, eps=1e-4)


# export
class QRNN(Module):
    "Apply a multiple layer Quasi-Recurrent Neural Network (QRNN) to an input sequence."

    def __init__(self, input_size, hidden_size, n_layers=1, batch_first=True, dropout=0,
                 bidirectional=False, save_prev_x=False, zoneout=0, window=None, output_gate=True):
        assert not (save_prev_x and bidirectional), "Can't save the previous X with bidirectional."
        kwargs = dict(batch_first=batch_first, zoneout=zoneout, output_gate=output_gate)
        self.layers = nn.ModuleList([QRNNLayer(input_size if l == 0 else hidden_size, hidden_size, save_prev_x=save_prev_x,
                                               window=((2 if l == 0 else 1) if window is None else window), **kwargs)
                                     for l in range(n_layers)])
        if bidirectional:
            self.layers_bwd = nn.ModuleList([QRNNLayer(input_size if l == 0 else hidden_size, hidden_size,
                                                       backward=True, window=((2 if l == 0 else 1) if window is None else window),
                                                       **kwargs) for l in range(n_layers)])
        self.n_layers, self.batch_first, self.dropout, self.bidirectional = n_layers, batch_first, dropout, bidirectional

    def reset(self):
        "Reset the hidden state."
        for layer in self.layers:
            layer.reset()
        if self.bidirectional:
            for layer in self.layers_bwd:
                layer.reset()

    def forward(self, inp, hid=None):
        new_hid = []
        if self.bidirectional:
            inp_bwd = inp.clone()
        for i, layer in enumerate(self.layers):
            inp, h = layer(inp, None if hid is None else hid[2 * i if self.bidirectional else i])
            new_hid.append(h)
            if self.bidirectional:
                inp_bwd, h_bwd = self.layers_bwd[i](inp_bwd, None if hid is None else hid[2 * i + 1])
                new_hid.append(h_bwd)
            if self.dropout != 0 and i < len(self.layers) - 1:
                for o in ([inp, inp_bwd] if self.bidirectional else [inp]):
                    o = F.dropout(o, p=self.dropout, training=self.training, inplace=False)
        if self.bidirectional:
            inp = torch.cat([inp, inp_bwd], dim=2)
        return inp, torch.stack(new_hid, 0)


qrnn = QRNN(10, 20, 2, bidirectional=True, batch_first=True, window=2, output_gate=False)
x = torch.randn(7, 5, 10)
y, h = qrnn(x)
test_eq(y.size(), [7, 5, 40])
test_eq(h.size(), [4, 7, 20])
# Without an out gate, the last timestamp in the forward output is the second to last hidden
# and the first timestamp of the backward output is the last hidden
test_close(y[:, -1, :20], h[2])
test_close(y[:, 0, 20:], h[3])

# ## Export -

# hide
notebook2script()
