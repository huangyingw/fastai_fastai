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
class GradientAccumulation(Callback):
    "Accumulate gradients before updating weights"
    order, run_valid = MixedPrecision.order - 4, False
    def __init__(self, n_acc=32): store_attr()
    def before_fit(self): self.count = 0
    def after_loss(self): self.learn.loss_grad /= self.n_acc / find_bs(self.learn.yb)

    def before_step(self):
        "Skip weight update if we have not seen enough items"
        self.learn.loss_grad *= self.n_acc / find_bs(self.learn.yb)  # log correct loss
        self.count += find_bs(self.learn.yb)
        if self.count < self.n_acc:
            raise CancelBatchException()  # skip step/zero_grad
        else:
            self.count = 0


# +
# hide
class GetGrads(Callback):
    run_valid, order = False, GradientAccumulation.order + 1
    def before_step(self): self.grads = to_detach(L([p.grad.clone() for p in self.model.parameters()]))


def _test_acc(bs, n, cbs=None, cuda=False):
    with no_random(99):
        db = synth_dbunch(bs=bs, n_train=n, n_valid=n, cuda=cuda)
        learn = synth_learner(data=db, cbs=[GetGrads] + L(cbs))
        learn.fit(1, lr=0.01)
        train, valid = learn.recorder.values[-1]
        return train, valid, learn.get_grads.grads


acc_cb = GradientAccumulation(n_acc=8)

train1, valid1, grads1 = _test_acc(8, 1)
train2, valid2, grads2 = _test_acc(1, 8, acc_cb)

# grads should be same, valid loss same, train loss different
test_close(grads2, grads1)
test_close(valid2, valid1)
test_ne(train2, train1)
# -

# hide
# cuda
fp16_cb = MixedPrecision(init_scale=1024)
train1, valid1, grads1 = _test_acc(8, 1, fp16_cb, cuda=True)
train2, valid2, grads2 = _test_acc(1, 8, [acc_cb, fp16_cb], cuda=True)
test_close(grads2, grads1, eps=0.01)
test_close(valid2, valid1)
test_ne(train2, train1)

# When the number of steps per accumulation is higher than the number of batches, the parameters (and therefore validation loss) don't change at all:

learn = synth_learner()
learn.fit(1, lr=0.01, cbs=GradientAccumulation(n_acc=1000))
# ensure valid_loss didn't change
assert learn.recorder.values[-1][1] == learn.recorder.values[0][1]


# ## GradientClip -

# export
class GradientClip(Callback):
    "Clip norm of gradients"
    order = MixedPrecision.order + 1
    def __init__(self, max_norm: float=1., norm_type: float=2.0): store_attr()
    def before_step(self): nn.utils.clip_grad_norm_(self.parameters(), self.max_norm, self.norm_type)


# Normally if we use a learning rate that is too high, our training will diverge. This even happens if we use mixed precision training, which avoid infinities by using dynamic loss scaling, but still diverges:

fp16 = MixedPrecision()

set_seed(99)
learn = synth_learner(lr=1.1, cuda=True)
learn.fit(3, cbs=fp16)

# By adding the `GradientClip` callback, the gradient `norm_type` (default:2) norm is clipped to at most `max_norm` (default:1) using `nn.utils.clip_grad_norm_`, which can avoid loss divergence:

set_seed(99)
learn = synth_learner(lr=1.1, cuda=True)
learn.fit(3, cbs=[GradientClip, fp16])

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
    run_after = TrainEvalCallback
    "Freeze moving average statistics in all non-trainable batchnorm layers."

    def before_train(self):
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
test_ne(to_detach(learn1.model[0][1].running_mean), m)

# When we use the `BnFreeze` callback, the running statistics will not be changed during training. This is often important for getting good results from transfer learning.

# slow
learn1 = cnn_learner(deepcopy(dls), resnet18, pretrained=True, train_bn=False, cbs=BnFreeze)
m = learn1.model[0][1].running_mean.detach().clone()
learn1.fit(1, lr=0.02)
test_eq(to_detach(learn1.model[0][1].running_mean), m)

# ## Export -

# hide
notebook2script()
