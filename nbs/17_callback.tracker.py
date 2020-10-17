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
from fastai.callback.schedule import *
from fastai.test_utils import *
from nbdev.showdoc import *
from fastai.callback.fp16 import MixedPrecision
from fastai.callback.progress import *
from fastai.basics import *
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# +
# default_exp callback.tracker
# -

# export

# hide


# # Tracking callbacks
#
# > Callbacks that make decisions depending how a monitored metric/loss behaves

# ## TerminateOnNaNCallback -

# export
class TerminateOnNaNCallback(Callback):
    "A `Callback` that terminates training if loss is NaN."
    run_before = Recorder

    def after_batch(self):
        "Test if `last_loss` is NaN and interrupts training."
        if torch.isinf(self.loss) or torch.isnan(self.loss):
            raise CancelFitException


learn = synth_learner()
learn.fit(10, lr=100, cbs=TerminateOnNaNCallback())

assert len(learn.recorder.losses) < 10 * len(learn.dls.train)
for l in learn.recorder.losses:
    assert not torch.isinf(l) and not torch.isnan(l)


# ## TrackerCallback -

# export
class TrackerCallback(Callback):
    "A `Callback` that keeps track of the best value in `monitor`."
    remove_on_fetch, run_after = True, Recorder

    def __init__(self, monitor='valid_loss', comp=None, min_delta=0., reset_on_fit=True):
        if comp is None:
            comp = np.less if 'loss' in monitor or 'error' in monitor else np.greater
        if comp == np.less:
            min_delta *= -1
        self.monitor, self.comp, self.min_delta, self.reset_on_fit, self.best = monitor, comp, min_delta, reset_on_fit, None

    def before_fit(self):
        "Prepare the monitored value"
        self.run = not hasattr(self, "lr_finder") and not hasattr(self, "gather_preds")
        if self.reset_on_fit or self.best is None:
            self.best = float('inf') if self.comp == np.less else -float('inf')
        assert self.monitor in self.recorder.metric_names[1:]
        self.idx = list(self.recorder.metric_names[1:]).index(self.monitor)

    def after_epoch(self):
        "Compare the last value to the best up to now"
        val = self.recorder.values[-1][self.idx]
        if self.comp(val - self.min_delta, self.best):
            self.best, self.new_best = val, True
        else:
            self.new_best = False

    def after_fit(self): self.run = True


# When implementing a `Callback` that has behavior that depends on the best value of a metric or loss, subclass this `Callback` and use its `best` (for best value so far) and `new_best` (there was a new best value this epoch) attributes. If you want to maintain `best` over subsequent calls to `fit` (e.g., `Learner.fit_one_cycle`), set `reset_on_fit` = True.
#
# `comp` is the comparison operator used to determine if a value is best than another (defaults to `np.less` if 'loss' is in the name passed in `monitor`, `np.greater` otherwise) and `min_delta` is an optional float that requires a new value to go over the current best (depending on `comp`) by at least that amount.

# +
# hide
class FakeRecords(Callback):
    run_after = Recorder
    run_before = TrackerCallback

    def __init__(self, monitor, values): self.monitor, self.values = monitor, values

    def before_fit(self): self.idx = list(self.recorder.metric_names[1:]).index(self.monitor)
    def after_epoch(self): self.recorder.values[-1][self.idx] = self.values[self.epoch]

class TestTracker(Callback):
    run_after = TrackerCallback
    def before_fit(self): self.bests, self.news = [], []
    def after_epoch(self):
        self.bests.append(self.tracker.best)
        self.news.append(self.tracker.new_best)


# +
# hide
learn = synth_learner(n_trn=2, cbs=TestTracker())
cbs = [TrackerCallback(monitor='valid_loss'), FakeRecords('valid_loss', [0.2, 0.1])]
with learn.no_logging():
    learn.fit(2, cbs=cbs)
test_eq(learn.test_tracker.bests, [0.2, 0.1])
test_eq(learn.test_tracker.news, [True, True])

# With a min_delta
cbs = [TrackerCallback(monitor='valid_loss', min_delta=0.15), FakeRecords('valid_loss', [0.2, 0.1])]
with learn.no_logging():
    learn.fit(2, cbs=cbs)
test_eq(learn.test_tracker.bests, [0.2, 0.2])
test_eq(learn.test_tracker.news, [True, False])


# +
# hide
# By default metrics have to be bigger at each epoch.
def tst_metric(out, targ): return F.mse_loss(out, targ)
learn = synth_learner(n_trn=2, cbs=TestTracker(), metrics=tst_metric)
cbs = [TrackerCallback(monitor='tst_metric'), FakeRecords('tst_metric', [0.2, 0.1])]
with learn.no_logging():
    learn.fit(2, cbs=cbs)
test_eq(learn.test_tracker.bests, [0.2, 0.2])
test_eq(learn.test_tracker.news, [True, False])

# This can be overwritten by passing `comp=np.less`.
learn = synth_learner(n_trn=2, cbs=TestTracker(), metrics=tst_metric)
cbs = [TrackerCallback(monitor='tst_metric', comp=np.less), FakeRecords('tst_metric', [0.2, 0.1])]
with learn.no_logging():
    learn.fit(2, cbs=cbs)
test_eq(learn.test_tracker.bests, [0.2, 0.1])
test_eq(learn.test_tracker.news, [True, True])
# -

# hide
# Setting reset_on_fit=True will maintain the "best" value over subsequent calls to fit
learn = synth_learner(n_val=2, cbs=TrackerCallback(monitor='tst_metric', reset_on_fit=False), metrics=tst_metric)
tracker_cb = learn.cbs.filter(lambda cb: isinstance(cb, TrackerCallback))[0]
with learn.no_logging():
    learn.fit(1)
first_best = tracker_cb.best
with learn.no_logging():
    learn.fit(1)
test_eq(tracker_cb.best, first_best)

# hide
# A tracker callback is not run during an lr_find
learn = synth_learner(n_trn=2, cbs=TrackerCallback(monitor='tst_metric'), metrics=tst_metric)
learn.lr_find(num_it=15, show_plot=False)
assert not hasattr(learn, 'new_best')


# ## EarlyStoppingCallback -

# export
@log_args
class EarlyStoppingCallback(TrackerCallback):
    "A `TrackerCallback` that terminates training when monitored quantity stops improving."
    def __init__(self, monitor='valid_loss', comp=None, min_delta=0., patience=1, reset_on_fit=True):
        super().__init__(monitor=monitor, comp=comp, min_delta=min_delta, reset_on_fit=reset_on_fit)
        self.patience = patience

    def before_fit(self): self.wait = 0
    super().before_fit()
    def after_epoch(self):
        "Compare the value monitored to its best score and maybe stop training."
        super().after_epoch()
        if self.new_best:
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f'No improvement since epoch {self.epoch-self.wait}: early stopping')
                raise CancelFitException()


# `comp` is the comparison operator used to determine if a value is best than another (defaults to `np.less` if 'loss' is in the name passed in `monitor`, `np.greater` otherwise) and `min_delta` is an optional float that requires a new value to go over the current best (depending on `comp`) by at least that amount. `patience` is the number of epochs you're willing to wait without improvement.

learn = synth_learner(n_trn=2, metrics=F.mse_loss)
learn.fit(n_epoch=200, lr=1e-7, cbs=EarlyStoppingCallback(monitor='mse_loss', min_delta=0.1, patience=2))

learn.validate()

learn = synth_learner(n_trn=2)
learn.fit(n_epoch=200, lr=1e-7, cbs=EarlyStoppingCallback(monitor='valid_loss', min_delta=0.1, patience=2))

# hide
test_eq(len(learn.recorder.values), 3)


# ## SaveModelCallback -

# export
@log_args
class SaveModelCallback(TrackerCallback):
    "A `TrackerCallback` that saves the model's best during training and loads it at the end."
    def __init__(self, monitor='valid_loss', comp=None, min_delta=0., fname='model', every_epoch=False, with_opt=False, reset_on_fit=True):
        super().__init__(monitor=monitor, comp=comp, min_delta=min_delta, reset_on_fit=reset_on_fit)
        # keep track of file path for loggers
        self.last_saved_path = None
        store_attr('fname,every_epoch,with_opt')

    def _save(self, name):
        self.last_saved_path = self.learn.save(name, with_opt=self.with_opt)

    def after_epoch(self):
        "Compare the value monitored to its best score and save if best."
        if self.every_epoch:
            self._save(f'{self.fname}_{self.epoch}')
        else:  # every improvement
            super().after_epoch()
            if self.new_best:
                print(f'Better model found at epoch {self.epoch} with {self.monitor} value: {self.best}.')
                self._save(f'{self.fname}')

    def after_fit(self, **kwargs):
        "Load the best model."
        if not self.every_epoch:
            self.learn.load(f'{self.fname}')


# `comp` is the comparison operator used to determine if a value is best than another (defaults to `np.less` if 'loss' is in the name passed in `monitor`, `np.greater` otherwise) and `min_delta` is an optional float that requires a new value to go over the current best (depending on `comp`) by at least that amount. Model will be saved in `learn.path/learn.model_dir/name.pth`, maybe `every_epoch` or at each improvement of the monitored quantity.

learn = synth_learner(n_trn=2, path=Path.cwd() / 'tmp')
learn.fit(n_epoch=2, cbs=SaveModelCallback())
assert (Path.cwd() / 'tmp/models/model.pth').exists()
learn.fit(n_epoch=2, cbs=SaveModelCallback(every_epoch=True))
for i in range(2):
    assert (Path.cwd() / f'tmp/models/model_{i}.pth').exists()
shutil.rmtree(Path.cwd() / 'tmp')


# ## ReduceLROnPlateau

# export
@log_args
class ReduceLROnPlateau(TrackerCallback):
    "A `TrackerCallback` that reduces learning rate when a metric has stopped improving."
    def __init__(self, monitor='valid_loss', comp=None, min_delta=0., patience=1, factor=10., min_lr=0, reset_on_fit=True):
        super().__init__(monitor=monitor, comp=comp, min_delta=min_delta, reset_on_fit=reset_on_fit)
        self.patience, self.factor, self.min_lr = patience, factor, min_lr

    def before_fit(self): self.wait = 0
    super().before_fit()
    def after_epoch(self):
        "Compare the value monitored to its best score and reduce LR by `factor` if no improvement."
        super().after_epoch()
        if self.new_best:
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = self.opt.hypers[-1]['lr']
                for h in self.opt.hypers:
                    h['lr'] = max(h['lr'] / self.factor, self.min_lr)
                self.wait = 0
                if self.opt.hypers[-1]["lr"] < old_lr:
                    print(f'Epoch {self.epoch}: reducing lr to {self.opt.hypers[-1]["lr"]}')


learn = synth_learner(n_trn=2)
learn.fit(n_epoch=4, lr=1e-7, cbs=ReduceLROnPlateau(monitor='valid_loss', min_delta=0.1, patience=2))

# hide
test_eq(learn.opt.hypers[-1]['lr'], 1e-8)

learn = synth_learner(n_trn=2)
learn.fit(n_epoch=6, lr=5e-8, cbs=ReduceLROnPlateau(monitor='valid_loss', min_delta=0.1, patience=2, min_lr=1e-8))

# hide
test_eq(learn.opt.hypers[-1]['lr'], 1e-8)

# ## Export -

# hide
notebook2script()
