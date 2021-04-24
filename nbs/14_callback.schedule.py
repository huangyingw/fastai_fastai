# -*- coding: utf-8 -*-
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
from fastai.vision.all import *
from fastai.test_utils import *
from nbdev.showdoc import *
from fastai.basics import *
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# export

# hide

# +
# default_exp callback.schedule
# -

# # Hyperparam schedule
#
# > Callback and helper functions to schedule any hyper-parameter


# ## Annealing

# export
class _Annealer:
    def __init__(self, f, start, end): store_attr('f,start,end')
    def __call__(self, pos): return self.f(self.start, self.end, pos)


# export
def annealer(f):
    "Decorator to make `f` return itself partially applied."
    @functools.wraps(f)
    def _inner(start, end): return _Annealer(f, start, end)
    return _inner


# This is the decorator we will use for all of our scheduling functions, as it transforms a function taking `(start, end, pos)` to something taking `(start, end)` and return a function depending of `pos`.

# +
# export
# TODO Jeremy, make this pickle
# @annealer
#def SchedLin(start, end, pos): return start + pos*(end-start)
# @annealer
#def SchedCos(start, end, pos): return start + (1 + math.cos(math.pi*(1-pos))) * (end-start) / 2
# @annealer
#def SchedNo (start, end, pos): return start
# @annealer
#def SchedExp(start, end, pos): return start * (end/start) ** pos
#
#SchedLin.__doc__ = "Linear schedule function from `start` to `end`"
#SchedCos.__doc__ = "Cosine schedule function from `start` to `end`"
#SchedNo .__doc__ = "Constant schedule function with `start` value"
#SchedExp.__doc__ = "Exponential schedule function from `start` to `end`"

# +
# export
def sched_lin(start, end, pos): return start + pos * (end - start)


def sched_cos(start, end, pos): return start + (1 + math.cos(math.pi * (1 - pos))) * (end - start) / 2


def sched_no(start, end, pos): return start


def sched_exp(start, end, pos): return start * (end / start) ** pos


def SchedLin(start, end): return _Annealer(sched_lin, start, end)


def SchedCos(start, end): return _Annealer(sched_cos, start, end)


def SchedNo(start, end): return _Annealer(sched_no, start, end)


def SchedExp(start, end): return _Annealer(sched_exp, start, end)


SchedLin.__doc__ = "Linear schedule function from `start` to `end`"
SchedCos.__doc__ = "Cosine schedule function from `start` to `end`"
SchedNo .__doc__ = "Constant schedule function with `start` value"
SchedExp.__doc__ = "Exponential schedule function from `start` to `end`"
# -

# hide
tst = pickle.dumps(SchedCos(0, 5))

annealings = "NO LINEAR COS EXP".split()
p = torch.linspace(0., 1, 100)
fns = [SchedNo, SchedLin, SchedCos, SchedExp]


# export
def SchedPoly(start, end, power):
    "Polynomial schedule (of `power`) function from `start` to `end`"
    def _inner(pos): return start + (end - start) * pos ** power
    return _inner


for fn, t in zip(fns, annealings):
    plt.plot(p, [fn(2, 1e-2)(o) for o in p], label=t)
f = SchedPoly(2, 1e-2, 0.5)
plt.plot(p, [f(o) for o in p], label="POLY(0.5)")
plt.legend()

show_doc(SchedLin)

sched = SchedLin(0, 2)
test_eq(L(map(sched, [0., 0.25, 0.5, 0.75, 1.])), [0., 0.5, 1., 1.5, 2.])

show_doc(SchedCos)

sched = SchedCos(0, 2)
test_close(L(map(sched, [0., 0.25, 0.5, 0.75, 1.])), [0., 0.29289, 1., 1.70711, 2.])

show_doc(SchedNo)

sched = SchedNo(0, 2)
test_close(L(map(sched, [0., 0.25, 0.5, 0.75, 1.])), [0., 0., 0., 0., 0.])

show_doc(SchedExp)

sched = SchedExp(1, 2)
test_close(L(map(sched, [0., 0.25, 0.5, 0.75, 1.])), [1., 1.18921, 1.41421, 1.68179, 2.])

show_doc(SchedPoly)

sched = SchedPoly(0, 2, 2)
test_close(L(map(sched, [0., 0.25, 0.5, 0.75, 1.])), [0., 0.125, 0.5, 1.125, 2.])

# +
p = torch.linspace(0., 1, 100)

pows = [0.5, 1., 2.]
for e in pows:
    f = SchedPoly(2, 0, e)
    plt.plot(p, [f(o) for o in p], label=f'power {e}')
plt.legend()


# -

# export
def combine_scheds(pcts, scheds):
    "Combine `scheds` according to `pcts` in one function"
    assert sum(pcts) == 1.
    pcts = tensor([0] + L(pcts))
    assert torch.all(pcts >= 0)
    pcts = torch.cumsum(pcts, 0)

    def _inner(pos):
        if int(pos) == 1:
            return scheds[-1](1.)
        idx = (pos >= pcts).nonzero().max()
        actual_pos = (pos - pcts[idx]) / (pcts[idx + 1] - pcts[idx])
        return scheds[idx](actual_pos.item())
    return _inner


# `pcts` must be a list of positive numbers that add up to 1 and is the same length as `scheds`. The generated function will use `scheds[0]` from 0 to `pcts[0]` then `scheds[1]` from `pcts[0]` to `pcts[0]+pcts[1]` and so forth.

p = torch.linspace(0., 1, 100)
f = combine_scheds([0.3, 0.7], [SchedCos(0.3, 0.6), SchedCos(0.6, 0.2)])
plt.plot(p, [f(o) for o in p])

p = torch.linspace(0., 1, 100)
f = combine_scheds([0.3, 0.2, 0.5], [SchedLin(0., 1.), SchedNo(1., 1.), SchedCos(1., 0.)])
plt.plot(p, [f(o) for o in p])

# hide
test_close([f(0.), f(0.15), f(0.3), f(0.4), f(0.5), f(0.7), f(1.)],
           [0., 0.5, 1., 1., 1., 0.65451, 0.])


# export
def combined_cos(pct, start, middle, end):
    "Return a scheduler with cosine annealing from `start`→`middle` & `middle`→`end`"
    return combine_scheds([pct, 1 - pct], [SchedCos(start, middle), SchedCos(middle, end)])


# This is a useful helper function for the [1cycle policy](https://sgugger.github.io/the-1cycle-policy.html). `pct` is used for the `start` to `middle` part, `1-pct` for the `middle` to `end`. Handles floats or collection of floats. For example:

f = combined_cos(0.25, 0.5, 1., 0.)
plt.plot(p, [f(o) for o in p])

# hide
test_close([f(0.), f(0.1), f(0.25), f(0.5), f(1.)], [0.5, 0.67275, 1., 0.75, 0.])
f = combined_cos(0.25, np.array([0.25, 0.5]), np.array([0.5, 1.]), np.array([0., 0.]))
for a, b in zip([f(0.), f(0.1), f(0.25), f(0.5), f(1.)],
                [[0.25, 0.5], [0.33638, 0.67275], [0.5, 1.], [0.375, 0.75], [0., 0.]]):
    test_close(a, b)


# ## ParamScheduler -

# export
@docs
class ParamScheduler(Callback):
    "Schedule hyper-parameters according to `scheds`"
    order, run_valid = 60, False

    def __init__(self, scheds): self.scheds = scheds
    def before_fit(self): self.hps = {p: [] for p in self.scheds.keys()}
    def before_batch(self): self._update_val(self.pct_train)

    def _update_val(self, pct):
        for n, f in self.scheds.items():
            self.opt.set_hyper(n, f(pct))

    def after_batch(self):
        for p in self.scheds.keys():
            self.hps[p].append(self.opt.hypers[-1][p])

    def after_fit(self):
        if hasattr(self.learn, 'recorder') and hasattr(self, 'hps'):
            self.recorder.hps = self.hps

    _docs = {"before_fit": "Initialize container for hyper-parameters",
             "before_batch": "Set the proper hyper-parameters in the optimizer",
             "after_batch": "Record hyper-parameters of this batch",
             "after_fit": "Save the hyper-parameters in the recorder if there is one"}


# `scheds` is a dictionary with one key for each hyper-parameter you want to schedule, with either a scheduler or a list of schedulers as values (in the second case, the list must have the same length as the the number of parameters groups of the optimizer).

learn = synth_learner()
sched = {'lr': SchedLin(1e-3, 1e-2)}
learn.fit(1, cbs=ParamScheduler(sched))
n = len(learn.dls.train)
test_close(learn.recorder.hps['lr'], [1e-3 + (1e-2 - 1e-3) * i / n for i in range(n)])


# hide
# test discriminative lrs
def _splitter(m): return [[m.a], [m.b]]


learn = synth_learner(splitter=_splitter)
sched = {'lr': combined_cos(0.5, np.array([1e-4, 1e-3]), np.array([1e-3, 1e-2]), np.array([1e-5, 1e-4]))}
learn.fit(1, cbs=ParamScheduler(sched))

show_doc(ParamScheduler.before_fit)

show_doc(ParamScheduler.before_batch)

show_doc(ParamScheduler.after_batch)

show_doc(ParamScheduler.after_fit)


# export
@patch
def fit_one_cycle(self: Learner, n_epoch, lr_max=None, div=25., div_final=1e5, pct_start=0.25, wd=None,
                  moms=None, cbs=None, reset_opt=False):
    "Fit `self.model` for `n_epoch` using the 1cycle policy."
    if self.opt is None:
        self.create_opt()
    self.opt.set_hyper('lr', self.lr if lr_max is None else lr_max)
    lr_max = np.array([h['lr'] for h in self.opt.hypers])
    scheds = {'lr': combined_cos(pct_start, lr_max / div, lr_max, lr_max / div_final),
              'mom': combined_cos(pct_start, *(self.moms if moms is None else moms))}
    self.fit(n_epoch, cbs=ParamScheduler(scheds) + L(cbs), reset_opt=reset_opt, wd=wd)


# The 1cycle policy was introduced by Leslie N. Smith et al. in [Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates](https://arxiv.org/abs/1708.07120). It schedules the learning rate with a cosine annealing from `lr_max/div` to `lr_max` then `lr_max/div_final` (pass an array to `lr_max` if you want to use differential learning rates) and the momentum with cosine annealing according to the values in `moms`. The first phase takes `pct_start` of the training. You can optionally pass additional `cbs` and `reset_opt`.

# Integration test: training a few epochs should make the model better
learn = synth_learner(lr=1e-2)
xb, yb = learn.dls.one_batch()
init_loss = learn.loss_func(learn.model(xb), yb)
learn.fit_one_cycle(2)
xb, yb = learn.dls.one_batch()
final_loss = learn.loss_func(learn.model(xb), yb)
assert final_loss < init_loss

# Scheduler test
lrs, moms = learn.recorder.hps['lr'], learn.recorder.hps['mom']
test_close(lrs, [combined_cos(0.25, 1e-2 / 25, 1e-2, 1e-7)(i / 20) for i in range(20)])
test_close(moms, [combined_cos(0.25, 0.95, 0.85, 0.95)(i / 20) for i in range(20)])


# export
@patch
def plot_sched(self: Recorder, keys=None, figsize=None):
    keys = self.hps.keys() if keys is None else L(keys)
    rows, cols = (len(keys) + 1) // 2, min(2, len(keys))
    figsize = figsize or (6 * cols, 4 * rows)
    _, axs = plt.subplots(rows, cols, figsize=figsize)
    axs = axs.flatten() if len(keys) > 1 else L(axs)
    for p, ax in zip(keys, axs):
        ax.plot(self.hps[p])
        ax.set_ylabel(p)


# hide
# test discriminative lrs
def _splitter(m): return [[m.a], [m.b]]


learn = synth_learner(splitter=_splitter)
learn.fit_one_cycle(1, lr_max=slice(1e-3, 1e-2))
#n = len(learn.dls.train)
#est_close(learn.recorder.hps['lr'], [1e-3 + (1e-2-1e-3) * i/n for i in range(n)])

learn = synth_learner()
learn.fit_one_cycle(2)

learn.recorder.plot_sched()


# export
@patch
def fit_flat_cos(self: Learner, n_epoch, lr=None, div_final=1e5, pct_start=0.75, wd=None,
                 cbs=None, reset_opt=False):
    "Fit `self.model` for `n_epoch` at flat `lr` before a cosine annealing."
    if self.opt is None:
        self.create_opt()
    self.opt.set_hyper('lr', self.lr if lr is None else lr)
    lr = np.array([h['lr'] for h in self.opt.hypers])
    scheds = {'lr': combined_cos(pct_start, lr, lr, lr / div_final)}
    self.fit(n_epoch, cbs=ParamScheduler(scheds) + L(cbs), reset_opt=reset_opt, wd=wd)


learn = synth_learner()
learn.fit_flat_cos(2)

learn.recorder.plot_sched()


# export
@patch
def fit_sgdr(self: Learner, n_cycles, cycle_len, lr_max=None, cycle_mult=2, cbs=None, reset_opt=False, wd=None):
    "Fit `self.model` for `n_cycles` of `cycle_len` using SGDR."
    if self.opt is None:
        self.create_opt()
    self.opt.set_hyper('lr', self.lr if lr_max is None else lr_max)
    lr_max = np.array([h['lr'] for h in self.opt.hypers])
    n_epoch = cycle_len * (cycle_mult**n_cycles - 1) // (cycle_mult - 1)
    pcts = [cycle_len * cycle_mult**i / n_epoch for i in range(n_cycles)]
    scheds = [SchedCos(lr_max, 0) for _ in range(n_cycles)]
    scheds = {'lr': combine_scheds(pcts, scheds)}
    self.fit(n_epoch, cbs=ParamScheduler(scheds) + L(cbs), reset_opt=reset_opt, wd=wd)


# This schedule was introduced by Ilya Loshchilov et al. in [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983). It consists of `n_cycles` that are cosine annealings from `lr_max` (defaults to the `Learner` lr) to 0, with a length of `cycle_len * cycle_mult**i` for the `i`-th cycle (first one is `cycle_len`-long, then we multiply the length by `cycle_mult` at each epoch). You can optionally pass additional `cbs` and `reset_opt`.

# +
# slow
learn = synth_learner()
with learn.no_logging():
    learn.fit_sgdr(3, 1)
test_eq(learn.n_epoch, 7)
iters = [k * len(learn.dls.train) for k in [0, 1, 3, 7]]
for i in range(3):
    n = iters[i + 1] - iters[i]
    # The start of a cycle can be mixed with the 0 of the previous cycle with rounding errors, so we test at +1
    test_close(learn.recorder.lrs[iters[i] + 1:iters[i + 1]], [SchedCos(learn.lr, 0)(k / n) for k in range(1, n)])

learn.recorder.plot_sched()


# -

# export
@patch
@delegates(Learner.fit_one_cycle)
def fine_tune(self: Learner, epochs, base_lr=2e-3, freeze_epochs=1, lr_mult=100,
              pct_start=0.3, div=5.0, **kwargs):
    "Fine tune with `freeze` for `freeze_epochs` then with `unfreeze` from `epochs` using discriminative LR"
    self.freeze()
    self.fit_one_cycle(freeze_epochs, slice(base_lr), pct_start=0.99, **kwargs)
    base_lr /= 2
    self.unfreeze()
    self.fit_one_cycle(epochs, slice(base_lr / lr_mult, base_lr), pct_start=pct_start, div=div, **kwargs)


learn.fine_tune(1)


# ## LRFind -

# export
@docs
class LRFinder(ParamScheduler):
    "Training with exponentially growing learning rate"

    def __init__(self, start_lr=1e-7, end_lr=10, num_it=100, stop_div=True):
        if is_listy(start_lr):
            self.scheds = {'lr': [SchedExp(s, e) for (s, e) in zip(start_lr, end_lr)]}
        else:
            self.scheds = {'lr': SchedExp(start_lr, end_lr)}
        self.num_it, self.stop_div = num_it, stop_div

    def before_fit(self):
        super().before_fit()
        self.learn.save('_tmp')
        self.best_loss = float('inf')

    def before_batch(self):
        self._update_val(self.train_iter / self.num_it)

    def after_batch(self):
        super().after_batch()
        if self.smooth_loss < self.best_loss:
            self.best_loss = self.smooth_loss
        if self.smooth_loss > 4 * self.best_loss and self.stop_div:
            raise CancelFitException()
        if self.train_iter >= self.num_it:
            raise CancelFitException()

    def before_validate(self): raise CancelValidException()

    def after_fit(self):
        self.learn.opt.zero_grad()  # Need to zero the gradients of the model before detaching the optimizer for future fits
        tmp_f = self.path / self.model_dir / '_tmp.pth'
        if tmp_f.exists():
            self.learn.load('_tmp', with_opt=True)
            os.remove(tmp_f)

    _docs = {"before_fit": "Initialize container for hyper-parameters and save the model",
             "before_batch": "Set the proper hyper-parameters in the optimizer",
             "after_batch": "Record hyper-parameters of this batch and potentially stop training",
             "after_fit": "Save the hyper-parameters in the recorder if there is one and load the original model",
             "before_validate": "Skip the validation part of training"}


# +
set_seed(99, True)
path = untar_data(URLs.PETS) / 'images'

image_files = get_image_files(path)
if sys.platform == "win32" and IN_NOTEBOOK:
    image_files = random.choices(image_files, k=int(len(image_files) / 8))
    print("Randomly select 1/8 files in NOTEBOOK on Windows to save time")

# pickle can't serializer lamda function.


def _label_func(x):
    return x[0].isupper()


dls = ImageDataLoaders.from_name_func(
    path, image_files, valid_pct=0.2,
    label_func=_label_func, item_tfms=Resize(224))

learn = cnn_learner(dls, resnet18)
learn.fit(1)
learn.opt.state_dict()['state'][1]['grad_avg']
# -

# slow
learn.lr_find()
learn.opt.state_dict()['state'][1]['grad_avg']

# slow
learn.lr_find()
learn.opt.state_dict()['state'][1]['grad_avg']

# slow
with tempfile.TemporaryDirectory() as d:
    learn = synth_learner(path=Path(d))
    init_a, init_b = learn.model.a, learn.model.b
    with learn.no_logging():
        learn.fit(20, cbs=LRFinder(num_it=100))
    assert len(learn.recorder.lrs) <= 100
    test_eq(len(learn.recorder.lrs), len(learn.recorder.losses))
    # Check stop if diverge
    if len(learn.recorder.lrs) < 100:
        assert learn.recorder.losses[-1] > 4 * min(learn.recorder.losses)
    # Test schedule
    test_eq(learn.recorder.lrs, [SchedExp(1e-7, 10)(i / 100) for i in range_of(learn.recorder.lrs)])
    # No validation data
    test_eq([len(v) for v in learn.recorder.values], [1 for _ in range_of(learn.recorder.values)])
    # Model loaded back properly
    test_eq(learn.model.a, init_a)
    test_eq(learn.model.b, init_b)
    test_eq(learn.opt.state_dict()['state'], [{}, {}])

show_doc(LRFinder.before_fit)

show_doc(LRFinder.before_batch)

show_doc(LRFinder.after_batch)

show_doc(LRFinder.before_validate)


# export
@patch
def plot_lr_find(self: Recorder, skip_end=5):
    "Plot the result of an LR Finder test (won't work if you didn't do `learn.lr_find()` before)"
    lrs = self.lrs if skip_end == 0 else self.lrs[:-skip_end]
    losses = self.losses if skip_end == 0 else self.losses[:-skip_end]
    fig, ax = plt.subplots(1, 1)
    ax.plot(lrs, losses)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Learning Rate")
    ax.set_xscale('log')


# export
SuggestedLRs = collections.namedtuple('SuggestedLRs', ['lr_min', 'lr_steep'])


# export
@patch
def lr_find(self: Learner, start_lr=1e-7, end_lr=10, num_it=100, stop_div=True, show_plot=True, suggestions=True):
    "Launch a mock training to find a good learning rate, return lr_min, lr_steep if `suggestions` is True"
    n_epoch = num_it // len(self.dls.train) + 1
    cb = LRFinder(start_lr=start_lr, end_lr=end_lr, num_it=num_it, stop_div=stop_div)
    with self.no_logging():
        self.fit(n_epoch, cbs=cb)
    if show_plot:
        self.recorder.plot_lr_find()
    if suggestions:
        lrs, losses = tensor(self.recorder.lrs[num_it // 10:-5]), tensor(self.recorder.losses[num_it // 10:-5])
        if len(losses) == 0:
            return
        lr_min = lrs[losses.argmin()].item()
        grads = (losses[1:] - losses[:-1]) / (lrs[1:].log() - lrs[:-1].log())
        lr_steep = lrs[grads.argmin()].item()
        return SuggestedLRs(lr_min / 10., lr_steep)


# First introduced by Leslie N. Smith in [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/pdf/1506.01186.pdf), the LR Finder trains the model with exponentially growing learning rates from `start_lr` to `end_lr` for `num_it` and stops in case of divergence (unless `stop_div=False`) then plots the losses vs the learning rates with a log scale.
#
# A good value for the learning rates is then either:
# - one tenth of the minimum before the divergence
# - when the slope is the steepest
#
# Those two values are returned by default by the Learning Rate Finder.
# slow
with tempfile.TemporaryDirectory() as d:
    learn = synth_learner(path=Path(d))
    weights_pre_lr_find = L(learn.model.parameters())
    lr_min, lr_steep = learn.lr_find()
    weights_post_lr_find = L(learn.model.parameters())
test_eq(weights_pre_lr_find, weights_post_lr_find)
print(f"Minimum/10: {lr_min:.2e}, steepest point: {lr_steep:.2e}")

# ## Export -

# hide
notebook2script()
