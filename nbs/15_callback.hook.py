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
import math
from fastai.test_utils import *
from nbdev.showdoc import *
from fastai.basics import *
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# export

# hide

# +
# default_exp callback.hook
# -

# # Model hooks
#
# > Callback and helper function to add hooks in models


# ## What are hooks?

# Hooks are functions you can attach to a particular layer in your model and that will be executed in the forward pass (for forward hooks) or backward pass (for backward hooks). Here we begin with an introduction around hooks, but you should jump to `HookCallback` if you quickly want to implement one (and read the following example `ActivationStats`).
#
# Forward hooks are functions that take three arguments: the layer it's applied to, the input of that layer and the output of that layer.

# +
tst_model = nn.Linear(5, 3)


def example_forward_hook(m, i, o): print(m, i, o)


x = torch.randn(4, 5)
hook = tst_model.register_forward_hook(example_forward_hook)
y = tst_model(x)
hook.remove()


# -

# Backward hooks are functions that take three arguments: the layer it's applied to, the gradients of the loss with respect to the input, and the gradients with respect to the output.

# +
def example_backward_hook(m, gi, go): print(m, gi, go)


hook = tst_model.register_backward_hook(example_backward_hook)

x = torch.randn(4, 5)
y = tst_model(x)
loss = y.pow(2).mean()
loss.backward()
hook.remove()


# -

# Hooks can change the input/output of a layer, or the gradients, print values or shapes. If you want to store something related to theses inputs/outputs, it's best to have your hook associated to a class so that it can put it in the state of an instance of that class.

# ## Hook -

# export
@docs
class Hook():
    "Create a hook on `m` with `hook_func`."

    def __init__(self, m, hook_func, is_forward=True, detach=True, cpu=False, gather=False):
        store_attr('hook_func,detach,cpu,gather')
        f = m.register_forward_hook if is_forward else m.register_backward_hook
        self.hook = f(self.hook_fn)
        self.stored, self.removed = None, False

    def hook_fn(self, module, input, output):
        "Applies `hook_func` to `module`, `input`, `output`."
        if self.detach:
            input, output = to_detach(input, cpu=self.cpu, gather=self.gather), to_detach(output, cpu=self.cpu, gather=self.gather)
        self.stored = self.hook_func(module, input, output)

    def remove(self):
        "Remove the hook from the model."
        if not self.removed:
            self.hook.remove()
            self.removed = True

    def __enter__(self, *args): return self
    def __exit__(self, *args): self.remove()

    _docs = dict(__enter__="Register the hook",
                 __exit__="Remove the hook")


# This will be called during the forward pass if `is_forward=True`, the backward pass otherwise, and will optionally `detach`, `gather` and put on the `cpu` the (gradient of the) input/output of the model before passing them to `hook_func`. The result of `hook_func` will be stored in the `stored` attribute of the `Hook`.

tst_model = nn.Linear(5, 3)
hook = Hook(tst_model, lambda m, i, o: o)
y = tst_model(x)
test_eq(hook.stored, y)

show_doc(Hook.hook_fn)

show_doc(Hook.remove)

# > Note: It's important to properly remove your hooks for your model when you're done to avoid them being called again next time your model is applied to some inputs, and to free the memory that go with their state.

tst_model = nn.Linear(5, 10)
x = torch.randn(4, 5)
y = tst_model(x)
hook = Hook(tst_model, example_forward_hook)
test_stdout(lambda: tst_model(x), f"{tst_model} ({x},) {y.detach()}")
hook.remove()
test_stdout(lambda: tst_model(x), "")

# ### Context Manager

# Since it's very important to remove your `Hook` even if your code is interrupted by some bug, `Hook` can be used as context managers.

show_doc(Hook.__enter__)

show_doc(Hook.__exit__)

tst_model = nn.Linear(5, 10)
x = torch.randn(4, 5)
y = tst_model(x)
with Hook(tst_model, example_forward_hook) as h:
    test_stdout(lambda: tst_model(x), f"{tst_model} ({x},) {y.detach()}")
test_stdout(lambda: tst_model(x), "")


# +
# export
def _hook_inner(m, i, o): return o if isinstance(o, Tensor) or is_listy(o) else list(o)


def hook_output(module, detach=True, cpu=False, grad=False):
    "Return a `Hook` that stores activations of `module` in `self.stored`"
    return Hook(module, _hook_inner, detach=detach, cpu=cpu, is_forward=not grad)


# -

# The activations stored are the gradients if `grad=True`, otherwise the output of `module`. If `detach=True` they are detached from their history, and if `cpu=True`, they're put on the CPU.

# +
tst_model = nn.Linear(5, 10)
x = torch.randn(4, 5)
with hook_output(tst_model) as h:
    y = tst_model(x)
    test_eq(y, h.stored)
    assert not h.stored.requires_grad

with hook_output(tst_model, grad=True) as h:
    y = tst_model(x)
    loss = y.pow(2).mean()
    loss.backward()
    test_close(2 * y / y.numel(), h.stored[0])
# -

# cuda
with hook_output(tst_model, cpu=True) as h:
    y = tst_model.cuda()(x.cuda())
    test_eq(h.stored.device, torch.device('cpu'))


# ## Hooks -

# export
@docs
class Hooks():
    "Create several hooks on the modules in `ms` with `hook_func`."

    def __init__(self, ms, hook_func, is_forward=True, detach=True, cpu=False):
        self.hooks = [Hook(m, hook_func, is_forward, detach, cpu) for m in ms]

    def __getitem__(self, i): return self.hooks[i]
    def __len__(self): return len(self.hooks)
    def __iter__(self): return iter(self.hooks)
    @property
    def stored(self): return L(o.stored for o in self)

    def remove(self):
        "Remove the hooks from the model."
        for h in self.hooks:
            h.remove()

    def __enter__(self, *args): return self
    def __exit__(self, *args): self.remove()

    _docs = dict(stored="The states saved in each hook.",
                 __enter__="Register the hooks",
                 __exit__="Remove the hooks")


layers = [nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 3)]
tst_model = nn.Sequential(*layers)
hooks = Hooks(tst_model, lambda m, i, o: o)
y = tst_model(x)
test_eq(hooks.stored[0], layers[0](x))
test_eq(hooks.stored[1], F.relu(layers[0](x)))
test_eq(hooks.stored[2], y)
hooks.remove()

show_doc(Hooks.stored, name='Hooks.stored')

show_doc(Hooks.remove)

# ### Context Manager

# Like `Hook` , you can use `Hooks` as context managers.

show_doc(Hooks.__enter__)

show_doc(Hooks.__exit__)

layers = [nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 3)]
tst_model = nn.Sequential(*layers)
with Hooks(layers, lambda m, i, o: o) as h:
    y = tst_model(x)
    test_eq(h.stored[0], layers[0](x))
    test_eq(h.stored[1], F.relu(layers[0](x)))
    test_eq(h.stored[2], y)


# export
def hook_outputs(modules, detach=True, cpu=False, grad=False):
    "Return `Hooks` that store activations of all `modules` in `self.stored`"
    return Hooks(modules, _hook_inner, detach=detach, cpu=cpu, is_forward=not grad)


# The activations stored are the gradients if `grad=True`, otherwise the output of `modules`. If `detach=True` they are detached from their history, and if `cpu=True`, they're put on the CPU.

# +
layers = [nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 3)]
tst_model = nn.Sequential(*layers)
x = torch.randn(4, 5)
with hook_outputs(layers) as h:
    y = tst_model(x)
    test_eq(h.stored[0], layers[0](x))
    test_eq(h.stored[1], F.relu(layers[0](x)))
    test_eq(h.stored[2], y)
    for s in h.stored:
        assert not s.requires_grad

with hook_outputs(layers, grad=True) as h:
    y = tst_model(x)
    loss = y.pow(2).mean()
    loss.backward()
    g = 2 * y / y.numel()
    test_close(g, h.stored[2][0])
    g = g @ layers[2].weight.data
    test_close(g, h.stored[1][0])
    g = g * (layers[0](x) > 0).float()
    test_close(g, h.stored[0][0])
# -

# cuda
with hook_outputs(tst_model, cpu=True) as h:
    y = tst_model.cuda()(x.cuda())
    for s in h.stored:
        test_eq(s.device, torch.device('cpu'))


# export
def dummy_eval(m, size=(64, 64)):
    "Evaluate `m` on a dummy input of a certain `size`"
    ch_in = in_channels(m)
    x = one_param(m).new(1, ch_in, *size).requires_grad_(False).uniform_(-1., 1.)
    with torch.no_grad():
        return m.eval()(x)


# export
def model_sizes(m, size=(64, 64)):
    "Pass a dummy input through the model `m` to get the various sizes of activations."
    with hook_outputs(m) as hooks:
        _ = dummy_eval(m, size=size)
        return [o.stored.shape for o in hooks]


m = nn.Sequential(ConvLayer(3, 16), ConvLayer(16, 32, stride=2), ConvLayer(32, 32))
test_eq(model_sizes(m), [[1, 16, 64, 64], [1, 32, 32, 32], [1, 32, 32, 32]])


# export
def num_features_model(m):
    "Return the number of output features for `m`."
    sz, ch_in = 32, in_channels(m)
    while True:
        # Trying for a few sizes in case the model requires a big input size.
        try:
            return model_sizes(m, (sz, sz))[-1][1]
        except Exception as e:
            sz *= 2
            if sz > 2048:
                raise e


m = nn.Sequential(nn.Conv2d(5, 4, 3), nn.Conv2d(4, 3, 3))
test_eq(num_features_model(m), 3)
m = nn.Sequential(ConvLayer(3, 16), ConvLayer(16, 32, stride=2), ConvLayer(32, 32))
test_eq(num_features_model(m), 32)


# ## HookCallback -

# To make hooks easy to use, we wrapped a version in a Callback where you just have to implement a `hook` function (plus any element you might need).

# export
def has_params(m):
    "Check if `m` has at least one parameter"
    return len(list(m.parameters())) > 0


assert has_params(nn.Linear(3, 4))
assert has_params(nn.LSTM(4, 5, 2))
assert not has_params(nn.ReLU())


# export
@funcs_kwargs
class HookCallback(Callback):
    "`Callback` that can be used to register hooks on `modules`"
    _methods = ["hook"]
    hook = noops

    def __init__(self, modules=None, every=None, remove_end=True, is_forward=True, detach=True, cpu=True, **kwargs):
        store_attr('modules,every,remove_end,is_forward,detach,cpu')
        assert not kwargs

    def before_fit(self):
        "Register the `Hooks` on `self.modules`."
        if self.modules is None:
            self.modules = [m for m in flatten_model(self.model) if has_params(m)]
        if self.every is None:
            self._register()

    def before_batch(self):
        if self.every is None:
            return
        if self.training and self.train_iter % self.every == 0:
            self._register()

    def after_batch(self):
        if self.every is None:
            return
        if self.training and self.train_iter % self.every == 0:
            self._remove()

    def after_fit(self):
        "Remove the `Hooks`."
        if self.remove_end:
            self._remove()

    def _register(self): self.hooks = Hooks(self.modules, self.hook, self.is_forward, self.detach, self.cpu)

    def _remove(self):
        if getattr(self, 'hooks', None):
            self.hooks.remove()

    def __del__(self): self._remove()


# You can either subclass and implement a `hook` function (along with any event you want) or pass that a `hook` function when initializing. Such a function needs to take three argument: a layer, input and output (for a backward hook, input means gradient with respect to the inputs, output, gradient with respect to the output) and can either modify them or update the state according to them.
#
# If not provided, `modules` will default to the layers of `self.model` that have a `weight` attribute. Depending on `do_remove`, the hooks will be properly removed at the end of training (or in case of error). `is_forward` , `detach` and `cpu` are passed to `Hooks`.
#
# The function called at each forward (or backward) pass is `self.hook` and must be implemented when subclassing this callback.

# +
class TstCallback(HookCallback):
    def hook(self, m, i, o): return o
    def after_batch(self): test_eq(self.hooks.stored[0], self.pred)


learn = synth_learner(n_trn=5, cbs=TstCallback())
learn.fit(1)


# +
class TstCallback(HookCallback):
    def __init__(self, modules=None, remove_end=True, detach=True, cpu=False):
        super().__init__(modules, None, remove_end, False, detach, cpu)

    def hook(self, m, i, o): return o

    def after_batch(self):
        if self.training:
            test_eq(self.hooks.stored[0][0], 2 * (self.pred - self.y) / self.pred.shape[0])


learn = synth_learner(n_trn=5, cbs=TstCallback())
learn.fit(1)
# -

show_doc(HookCallback.before_fit)

show_doc(HookCallback.after_fit)


# ## Model summary

# export
def total_params(m):
    "Give the number of parameters of a module and if it's trainable or not"
    params = sum([p.numel() for p in m.parameters()])
    trains = [p.requires_grad for p in m.parameters()]
    return params, (False if len(trains) == 0 else trains[0])


test_eq(total_params(nn.Linear(10, 32)), (32 * 10 + 32, True))
test_eq(total_params(nn.Linear(10, 32, bias=False)), (32 * 10, True))
test_eq(total_params(nn.BatchNorm2d(20)), (20 * 2, True))
test_eq(total_params(nn.BatchNorm2d(20, affine=False)), (0, False))
test_eq(total_params(nn.Conv2d(16, 32, 3)), (16 * 32 * 3 * 3 + 32, True))
test_eq(total_params(nn.Conv2d(16, 32, 3, bias=False)), (16 * 32 * 3 * 3, True))
# First ih layer 20--10, all else 10--10. *4 for the four gates
test_eq(total_params(nn.LSTM(20, 10, 2)), (4 * (20 * 10 + 10) + 3 * 4 * (10 * 10 + 10), True))


# export
def layer_info(learn, *xb):
    "Return layer infos of `model` on `xb` (only support batch first inputs)"
    def _track(m, i, o): return (m.__class__.__name__,) + total_params(m) + (apply(lambda x: x.shape, o),)
    with Hooks(flatten_model(learn.model), _track) as h:
        batch = apply(lambda o: o[:1], xb)
        with learn:
            r = learn.get_preds(dl=[batch], inner=True, reorder=False)
        return h.stored


def _m(): return nn.Sequential(nn.Linear(1, 50), nn.ReLU(), nn.BatchNorm1d(50), nn.Linear(50, 1))


sample_input = torch.randn((16, 1))
test_eq(layer_info(synth_learner(model=_m()), sample_input), [
    ('Linear', 100, True, [1, 50]),
    ('ReLU', 0, False, [1, 50]),
    ('BatchNorm1d', 100, True, [1, 50]),
    ('Linear', 51, True, [1, 1])
])


# +
# hide
# Test for multiple inputs model
class _2InpModel(Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(2, 50), nn.ReLU(), nn.BatchNorm1d(50), nn.Linear(50, 1))

    def forward(self, *inps):
        outputs = torch.cat(inps, dim=-1)
        return self.seq(outputs)


sample_inputs = (torch.randn(16, 1), torch.randn(16, 1))
learn = synth_learner(model=_2InpModel())
learn.dls.n_inp = 2
test_eq(layer_info(learn, *sample_inputs), [
    ('Linear', 150, True, [1, 50]),
    ('ReLU', 0, False, [1, 50]),
    ('BatchNorm1d', 100, True, [1, 50]),
    ('Linear', 51, True, [1, 1])
])


# -

# export
def _print_shapes(o, bs):
    if isinstance(o, torch.Size):
        return ' x '.join([str(bs)] + [str(t) for t in o[1:]])
    else:
        return str([_print_shapes(x, bs) for x in o])


# export
def module_summary(learn, *xb):
    "Print a summary of `model` using `xb`"
    # Individual parameters wrapped in ParameterModule aren't called through the hooks in `layer_info`,
    #  thus are not counted inside the summary
    # TODO: find a way to have them counted in param number somehow
    infos = layer_info(learn, *xb)
    n, bs = 64, find_bs(xb)
    inp_sz = _print_shapes(apply(lambda x: x.shape, xb), bs)
    res = f"{learn.model.__class__.__name__} (Input shape: {inp_sz})\n"
    res += "=" * n + "\n"
    res += f"{'Layer (type)':<20} {'Output Shape':<20} {'Param #':<10} {'Trainable':<10}\n"
    res += "=" * n + "\n"
    ps, trn_ps = 0, 0
    infos = [o for o in infos if o is not None]  # see comment in previous cell
    for typ, np, trn, sz in infos:
        if sz is None:
            continue
        ps += np
        if trn:
            trn_ps += np
        res += f"{typ:<20} {_print_shapes(sz, bs)[:19]:<20} {np:<10,} {str(trn):<10}\n"
        res += "_" * n + "\n"
    res += f"\nTotal params: {ps:,}\n"
    res += f"Total trainable params: {trn_ps:,}\n"
    res += f"Total non-trainable params: {ps - trn_ps:,}\n\n"
    return PrettyString(res)


# export
@patch
def summary(self: Learner):
    "Print a summary of the model, optimizer and loss function."
    xb = self.dls.train.one_batch()[:self.dls.train.n_inp]
    res = module_summary(self, *xb)
    res += f"Optimizer used: {self.opt_func}\nLoss function: {self.loss_func}\n\n"
    if self.opt is not None:
        res += f"Model " + ("unfrozen\n\n" if self.opt.frozen_idx == 0 else f"frozen up to parameter group #{self.opt.frozen_idx}\n\n")
    res += "Callbacks:\n" + '\n'.join(f"  - {cb}" for cb in sort_by_run(self.cbs))
    return PrettyString(res)


learn = synth_learner(model=_m())
learn.summary()

# hide
# cuda
learn = synth_learner(model=_m(), cuda=True)
learn.summary()


# +
# hide
# Test for multiple output
class _NOutModel(Module):
    def __init__(self): self.lin = nn.Linear(5, 6)

    def forward(self, x1):
        x = torch.randn((10, 5))
        return x, self.lin(x)


learn = synth_learner(model=_NOutModel())
learn.summary()  # Output Shape should be (50, 16, 256), (1, 16, 256)


# -

# ## Activation graphs

# This is an example of a `HookCallback`, that stores the mean, stds and histograms of activations that go through the network.

# exports
@delegates()
class ActivationStats(HookCallback):
    "Callback that record the mean and std of activations."
    run_before = TrainEvalCallback

    def __init__(self, with_hist=False, **kwargs):
        super().__init__(**kwargs)
        self.with_hist = with_hist

    def before_fit(self):
        "Initialize stats."
        super().before_fit()
        self.stats = L()

    def hook(self, m, i, o):
        o = o.float()
        res = {'mean': o.mean().item(), 'std': o.std().item(),
               'near_zero': (o <= 0.05).long().sum().item() / o.numel()}
        if self.with_hist:
            res['hist'] = o.histc(40, 0, 10)
        return res

    def after_batch(self):
        "Take the stored results and puts it in `self.stats`"
        if self.training and (self.every is None or self.train_iter % self.every == 0):
            self.stats.append(self.hooks.stored)
        super().after_batch()

    def layer_stats(self, idx):
        lstats = self.stats.itemgot(idx)
        return L(lstats.itemgot(o) for o in ('mean', 'std', 'near_zero'))

    def hist(self, idx):
        res = self.stats.itemgot(idx).itemgot('hist')
        return torch.stack(tuple(res)).t().float().log1p()

    def color_dim(self, idx, figsize=(10, 5), ax=None):
        "The 'colorful dimension' plot"
        res = self.hist(idx)
        if ax is None:
            ax = subplots(figsize=figsize)[1][0]
        ax.imshow(res, origin='lower')
        ax.axis('off')

    def plot_layer_stats(self, idx):
        _, axs = subplots(1, 3, figsize=(12, 3))
        for o, ax, title in zip(self.layer_stats(idx), axs, ('mean', 'std', '% near zero')):
            ax.plot(o)
            ax.set_title(title)


learn = synth_learner(n_trn=5, cbs=ActivationStats(every=4))
learn.fit(1)

learn.activation_stats.stats

# The first line contains the means of the outputs of the model for each batch in the training set, the second line their standard deviations.


# +
def test_every(n_tr, every):
    "create a learner, fit, then check number of stats collected"
    learn = synth_learner(n_trn=n_tr, cbs=ActivationStats(every=every))
    learn.fit(1)
    expected_stats_len = math.ceil(n_tr / every)
    test_eq(expected_stats_len, len(learn.activation_stats.stats))


for n_tr in [11, 12, 13]:
    test_every(n_tr, 4)
    test_every(n_tr, 1)


# +
# hide
class TstCallback(HookCallback):
    def hook(self, m, i, o): return o

    def before_fit(self):
        super().before_fit()
        self.means, self.stds = [], []

    def after_batch(self):
        if self.training:
            self.means.append(self.hooks.stored[0].mean().item())
            self.stds.append(self.hooks.stored[0].std() .item())


learn = synth_learner(n_trn=5, cbs=[TstCallback(), ActivationStats()])
learn.fit(1)
test_eq(learn.activation_stats.stats.itemgot(0).itemgot("mean"), learn.tst.means)
test_eq(learn.activation_stats.stats.itemgot(0).itemgot("std"), learn.tst.stds)
# -

# ## Export -

# hide
notebook2script()
