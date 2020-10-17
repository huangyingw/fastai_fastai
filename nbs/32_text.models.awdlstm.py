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
from nbdev.showdoc import *
from fastai.text.core import *
from fastai.data.all import *
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# export

# hide


# +
# default_exp text.models.awdlstm
# default_cls_lvl 3
# -

# # AWD-LSTM
#
# > AWD LSTM from [Smerity et al.](https://arxiv.org/pdf/1708.02182.pdf)

# ## Basic NLP modules

# On top of the pytorch or the fastai [`layers`](/layers.html#layers), the language models use some custom layers specific to NLP.

# export
def dropout_mask(x, sz, p):
    "Return a dropout mask of the same type as `x`, size `sz`, with probability `p` to cancel an element."
    return x.new(*sz).bernoulli_(1 - p).div_(1 - p)


t = dropout_mask(torch.randn(3, 4), [4, 3], 0.25)
test_eq(t.shape, [4, 3])
assert ((t == 4 / 3) + (t == 0)).all()


# export
class RNNDropout(Module):
    "Dropout with probability `p` that is consistent on the seq_len dimension."
    def __init__(self, p=0.5): self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.:
            return x
        return x * dropout_mask(x.data, (x.size(0), 1, *x.shape[2:]), self.p)


dp = RNNDropout(0.3)
tst_inp = torch.randn(4, 3, 7)
tst_out = dp(tst_inp)
for i in range(4):
    for j in range(7):
        if tst_out[i, 0, j] == 0:
            assert (tst_out[i, :, j] == 0).all()
        else:
            test_close(tst_out[i, :, j], tst_inp[i, :, j] / (1 - 0.3))

# It also supports doing dropout over a sequence of images where time dimesion is the 1st axis, 10 images of 3 channels and 32 by 32.

_ = dp(torch.rand(4, 10, 3, 32, 32))


# export
class WeightDropout(Module):
    "A module that wraps another layer in which some weights will be replaced by 0 during training."

    def __init__(self, module, weight_p, layer_names='weight_hh_l0'):
        self.module, self.weight_p, self.layer_names = module, weight_p, L(layer_names)
        for layer in self.layer_names:
            # Makes a copy of the weights of the selected layers.
            w = getattr(self.module, layer)
            delattr(self.module, layer)
            self.register_parameter(f'{layer}_raw', nn.Parameter(w.data))
            setattr(self.module, layer, w.clone())
            if isinstance(self.module, (nn.RNNBase, nn.modules.rnn.RNNBase)):
                self.module.flatten_parameters = self._do_nothing

    def _setweights(self):
        "Apply dropout to the raw weights."
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            if self.training:
                w = F.dropout(raw_w, p=self.weight_p)
            else:
                w = raw_w.clone()
            setattr(self.module, layer, w)

    def forward(self, *args):
        self._setweights()
        with warnings.catch_warnings():
            # To avoid the warning that comes because the weights aren't flattened.
            warnings.simplefilter("ignore", category=UserWarning)
            return self.module.forward(*args)

    def reset(self):
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            setattr(self.module, layer, raw_w.clone())
        if hasattr(self.module, 'reset'):
            self.module.reset()

    def _do_nothing(self): pass


module = nn.LSTM(5, 7)
dp_module = WeightDropout(module, 0.4)
wgts = dp_module.module.weight_hh_l0
tst_inp = torch.randn(10, 20, 5)
h = torch.zeros(1, 20, 7), torch.zeros(1, 20, 7)
dp_module.reset()
x, h = dp_module(tst_inp, h)
loss = x.sum()
loss.backward()
new_wgts = getattr(dp_module.module, 'weight_hh_l0')
test_eq(wgts, getattr(dp_module, 'weight_hh_l0_raw'))
assert 0.2 <= (new_wgts == 0).sum().float() / new_wgts.numel() <= 0.6
assert dp_module.weight_hh_l0_raw.requires_grad
assert dp_module.weight_hh_l0_raw.grad is not None
assert ((dp_module.weight_hh_l0_raw.grad == 0.) & (new_wgts == 0.)).any()


# export
class EmbeddingDropout(Module):
    "Apply dropout with probability `embed_p` to an embedding layer `emb`."

    def __init__(self, emb, embed_p):
        self.emb, self.embed_p = emb, embed_p

    def forward(self, words, scale=None):
        if self.training and self.embed_p != 0:
            size = (self.emb.weight.size(0), 1)
            mask = dropout_mask(self.emb.weight.data, size, self.embed_p)
            masked_embed = self.emb.weight * mask
        else:
            masked_embed = self.emb.weight
        if scale:
            masked_embed.mul_(scale)
        return F.embedding(words, masked_embed, ifnone(self.emb.padding_idx, -1), self.emb.max_norm,
                           self.emb.norm_type, self.emb.scale_grad_by_freq, self.emb.sparse)


enc = nn.Embedding(10, 7, padding_idx=1)
enc_dp = EmbeddingDropout(enc, 0.5)
tst_inp = torch.randint(0, 10, (8,))
tst_out = enc_dp(tst_inp)
for i in range(8):
    assert (tst_out[i] == 0).all() or torch.allclose(tst_out[i], 2 * enc.weight[tst_inp[i]])


# export
class AWD_LSTM(Module):
    "AWD-LSTM inspired by https://arxiv.org/abs/1708.02182"
    initrange = 0.1

    def __init__(self, vocab_sz, emb_sz, n_hid, n_layers, pad_token=1, hidden_p=0.2, input_p=0.6, embed_p=0.1,
                 weight_p=0.5, bidir=False):
        store_attr('emb_sz,n_hid,n_layers,pad_token')
        self.bs = 1
        self.n_dir = 2 if bidir else 1
        self.encoder = nn.Embedding(vocab_sz, emb_sz, padding_idx=pad_token)
        self.encoder_dp = EmbeddingDropout(self.encoder, embed_p)
        self.rnns = nn.ModuleList([self._one_rnn(emb_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else emb_sz) // self.n_dir,
                                                 bidir, weight_p, l) for l in range(n_layers)])
        self.encoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.input_dp = RNNDropout(input_p)
        self.hidden_dps = nn.ModuleList([RNNDropout(hidden_p) for l in range(n_layers)])
        self.reset()

    def forward(self, inp, from_embeds=False):
        bs, sl = inp.shape[:2] if from_embeds else inp.shape
        if bs != self.bs:
            self._change_hidden(bs)

        output = self.input_dp(inp if from_embeds else self.encoder_dp(inp))
        new_hidden = []
        for l, (rnn, hid_dp) in enumerate(zip(self.rnns, self.hidden_dps)):
            output, new_h = rnn(output, self.hidden[l])
            new_hidden.append(new_h)
            if l != self.n_layers - 1:
                output = hid_dp(output)
        self.hidden = to_detach(new_hidden, cpu=False, gather=False)
        return output

    def _change_hidden(self, bs):
        self.hidden = [self._change_one_hidden(l, bs) for l in range(self.n_layers)]
        self.bs = bs

    def _one_rnn(self, n_in, n_out, bidir, weight_p, l):
        "Return one of the inner rnn"
        rnn = nn.LSTM(n_in, n_out, 1, batch_first=True, bidirectional=bidir)
        return WeightDropout(rnn, weight_p)

    def _one_hidden(self, l):
        "Return one hidden state"
        nh = (self.n_hid if l != self.n_layers - 1 else self.emb_sz) // self.n_dir
        return (one_param(self).new_zeros(self.n_dir, self.bs, nh), one_param(self).new_zeros(self.n_dir, self.bs, nh))

    def _change_one_hidden(self, l, bs):
        if self.bs < bs:
            nh = (self.n_hid if l != self.n_layers - 1 else self.emb_sz) // self.n_dir
            return tuple(torch.cat([h, h.new_zeros(self.n_dir, bs - self.bs, nh)], dim=1) for h in self.hidden[l])
        if self.bs > bs:
            return (self.hidden[l][0][:, :bs].contiguous(), self.hidden[l][1][:, :bs].contiguous())
        return self.hidden[l]

    def reset(self):
        "Reset the hidden states"
        [r.reset() for r in self.rnns if hasattr(r, 'reset')]
        self.hidden = [self._one_hidden(l) for l in range(self.n_layers)]


# This is the core of an AWD-LSTM model, with embeddings from `vocab_sz` and `emb_sz`, `n_layers` LSTMs potentially `bidir` stacked, the first one going from `emb_sz` to `n_hid`, the last one from `n_hid` to `emb_sz` and all the inner ones from `n_hid` to `n_hid`. `pad_token` is passed to the PyTorch embedding layer. The dropouts are applied as such:
# - the embeddings are wrapped in `EmbeddingDropout` of probability `embed_p`;
# - the result of this embedding layer goes through an `RNNDropout` of probability `input_p`;
# - each LSTM has `WeightDropout` applied with probability `weight_p`;
# - between two of the inner LSTM, an `RNNDropout` is applied with probability `hidden_p`.
#
# THe module returns two lists: the raw outputs (without being applied the dropout of `hidden_p`) of each inner LSTM and the list of outputs with dropout. Since there is no dropout applied on the last output, those two lists have the same last element, which is the output that should be fed to a decoder (in the case of a language model).

# +
tst = AWD_LSTM(100, 20, 10, 2, hidden_p=0.2, embed_p=0.02, input_p=0.1, weight_p=0.2)
x = torch.randint(0, 100, (10, 5))
r = tst(x)
test_eq(tst.bs, 10)
test_eq(len(tst.hidden), 2)
test_eq([h_.shape for h_ in tst.hidden[0]], [[1, 10, 10], [1, 10, 10]])
test_eq([h_.shape for h_ in tst.hidden[1]], [[1, 10, 20], [1, 10, 20]])

test_eq(r.shape, [10, 5, 20])
test_eq(r[:, -1], tst.hidden[-1][0][0])  # hidden state is the last timestep in raw outputs

tst.eval()
tst.reset()
tst(x)
tst(x)
# -

# hide
# test bs change
x = torch.randint(0, 100, (6, 5))
r = tst(x)
test_eq(tst.bs, 6)

# +
# hide
# cuda
tst = AWD_LSTM(100, 20, 10, 2, bidir=True).to('cuda')
tst.reset()
x = torch.randint(0, 100, (10, 5)).to('cuda')
r = tst(x)

x = torch.randint(0, 100, (6, 5), device='cuda')
r = tst(x)


# -

# export
def awd_lstm_lm_split(model):
    "Split a RNN `model` in groups for differential learning rates."
    groups = [nn.Sequential(rnn, dp) for rnn, dp in zip(model[0].rnns, model[0].hidden_dps)]
    groups = L(groups + [nn.Sequential(model[0].encoder, model[0].encoder_dp, model[1])])
    return groups.map(params)


# export
awd_lstm_lm_config = dict(emb_sz=400, n_hid=1152, n_layers=3, pad_token=1, bidir=False, output_p=0.1,
                          hidden_p=0.15, input_p=0.25, embed_p=0.02, weight_p=0.2, tie_weights=True, out_bias=True)


# export
def awd_lstm_clas_split(model):
    "Split a RNN `model` in groups for differential learning rates."
    groups = [nn.Sequential(model[0].module.encoder, model[0].module.encoder_dp)]
    groups += [nn.Sequential(rnn, dp) for rnn, dp in zip(model[0].module.rnns, model[0].module.hidden_dps)]
    groups = L(groups + [model[1]])
    return groups.map(params)


# export
awd_lstm_clas_config = dict(emb_sz=400, n_hid=1152, n_layers=3, pad_token=1, bidir=False, output_p=0.4,
                            hidden_p=0.3, input_p=0.4, embed_p=0.05, weight_p=0.5)


# ## QRNN

# export
class AWD_QRNN(AWD_LSTM):
    "Same as an AWD-LSTM, but using QRNNs instead of LSTMs"
    def _one_rnn(self, n_in, n_out, bidir, weight_p, l):
        from fastai.text.models.qrnn import QRNN
        rnn = QRNN(n_in, n_out, 1, save_prev_x=(not bidir), zoneout=0, window=2 if l == 0 else 1, output_gate=True, bidirectional=bidir)
        rnn.layers[0].linear = WeightDropout(rnn.layers[0].linear, weight_p, layer_names='weight')
        return rnn

    def _one_hidden(self, l):
        "Return one hidden state"
        nh = (self.n_hid if l != self.n_layers - 1 else self.emb_sz) // self.n_dir
        return one_param(self).new_zeros(self.n_dir, self.bs, nh)

    def _change_one_hidden(self, l, bs):
        if self.bs < bs:
            nh = (self.n_hid if l != self.n_layers - 1 else self.emb_sz) // self.n_dir
            return torch.cat([self.hidden[l], self.hidden[l].new_zeros(self.n_dir, bs - self.bs, nh)], dim=1)
        if self.bs > bs:
            return self.hidden[l][:, :bs]
        return self.hidden[l]


# cuda
# cpp
model = AWD_QRNN(vocab_sz=10, emb_sz=20, n_hid=16, n_layers=2, bidir=False)
x = torch.randint(0, 10, (7, 5))
y = model(x)
test_eq(y.shape, (7, 5, 20))

# hide
# cuda
# cpp
# test bidir=True
model = AWD_QRNN(vocab_sz=10, emb_sz=20, n_hid=16, n_layers=2, bidir=True)
x = torch.randint(0, 10, (7, 5))
y = model(x)
test_eq(y.shape, (7, 5, 20))

# export
awd_qrnn_lm_config = dict(emb_sz=400, n_hid=1552, n_layers=4, pad_token=1, bidir=False, output_p=0.1,
                          hidden_p=0.15, input_p=0.25, embed_p=0.02, weight_p=0.2, tie_weights=True, out_bias=True)

# export
awd_qrnn_clas_config = dict(emb_sz=400, n_hid=1552, n_layers=4, pad_token=1, bidir=False, output_p=0.4,
                            hidden_p=0.3, input_p=0.4, embed_p=0.05, weight_p=0.5)

# ## Export -

# hide
notebook2script()
