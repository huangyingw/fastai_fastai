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
from fastai.callback.fp16 import *
from fastai.callback.progress import *
from fastai.basics import *
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# +
# default_exp callback.training
# -

# export

# hide


# # Training callbacks
#
# > Various callbacks to customize training behavior

# ## ShortEpochCallback -

# export
@log_args
class ShortEpochCallback(Callback):
    "Fit just `pct` of an epoch, then stop"

    def __init__(self, pct=0.01, short_valid=True): self.pct, self.short_valid = pct, short_valid

    def after_batch(self):
        if self.iter / self.n_iter < self.pct:
            return
        if self.training:
            raise CancelTrainException
        if self.short_valid:
            raise CancelValidException


learn = synth_learner()
learn.fit(1, cbs=ShortEpochCallback())

learn = synth_learner()
learn.fit(1, cbs=ShortEpochCallback(short_valid=False))


# ## GradientAccumulation -

# export
@log_args
class GradientAccumulation(Callback):
    "Accumulate gradients before updating weights"
    toward_end, run_before = True, MixedPrecision

    def __init__(self, n_acc=32): store_attr('n_acc')
    def before_fit(self): self.count = 0

    def after_backward(self):
        self.count += find_bs(self.learn.yb)
        if self.count < self.n_acc:
            raise CancelBatchException()  # skip weight update
        else:
            self.count = 0

    _docs = dict(before_fit="Set counter to 0",
                 after_backward="Skip weight update if we have not seen enough items")


# +
learn = synth_learner()

learn.fit(2, lr=0.01, cbs=GradientAccumulation(n_acc=2 * learn.dls.bs))
# ensure train_loss decreased
assert learn.recorder.values[-1][0] < learn.recorder.values[0][0]

learn.fit(2, lr=0.01, cbs=GradientAccumulation(n_acc=1e6))
# ensure valid_loss didn't change (same weights)
assert learn.recorder.values[-1][1] == learn.recorder.values[0][1]
# -

# ## BnFreeze

# +
# export
bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def set_bn_eval(m: nn.Module, use_eval=True) -> None:
    "Set bn layers in eval mode for all recursive children of `m`."
    for l in m.children():
        if isinstance(l, bn_types) and not next(l.parameters()).requires_grad:
            if use_eval:
                l.eval()
            else:
                l.train()
        set_bn_eval(l)


class BnFreeze(Callback):
    "Freeze moving average statistics in all non-trainable batchnorm layers."

    def before_epoch(self):
        set_bn_eval(self.model)


# -

# `BnFreeze` is useful when you'd like to train two separate models that have a common feature extractor / body. The only part of the model that's different is the head that you attach for transfer learning. <br>
#
# `Learner.freeze()` doesn't suffice here as the `BatchNorm` layers are trainable by default, and running mean and std of batches are tracked. For feature extractors to fully match, you need to set `train_bn=False` and these stats need to be frozen as well, which is precisely the function of `BnFreeze`.


# slow
path = untar_data(URLs.MNIST_TINY)
dls = ImageDataLoaders.from_folder(path, valid_pct=0.2)

# We first demonstrate the mismatch of the running stats when using only `train_bn=False`, by creating a `Learner`...:

# slow
learn1 = cnn_learner(deepcopy(dls), resnet18, pretrained=True, train_bn=False)

# ...and grab the first `BatchNorm` layer, and store its running mean:

# slow
m = learn1.model[0][1].running_mean.clone()

# You can see that now that running mean has changed:

# slow
learn1.fit(1, lr=0.02)
test_ne(learn1.model[0][1].running_mean, m)

# When we use the `BnFreeze` callback, the running statistics will not be changed during training. This is often important for getting good results from transfer learning.

# slow
learn1 = cnn_learner(deepcopy(dls), resnet18, pretrained=True, train_bn=False, cbs=BnFreeze)
m = learn1.model[0][1].running_mean.clone()
learn1.fit(1, lr=0.02)
test_eq(learn1.model[0][1].running_mean, m)

# ## Export -

# hide
notebook2script()
