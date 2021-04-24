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
from nbdev.export import *
from nbdev.showdoc import *
from fastai.layers import *
from fastai.torch_core import *
from fastai.torch_imports import *
from fastai.imports import *
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# +
# default_exp losses
# default_cls_lvl 3
# -

# export

# hide


# # Loss Functions
# > Custom fastai loss functions

# export
class BaseLoss():
    "Same as `loss_cls`, but flattens input and target."
    activation = decodes = noops

    def __init__(self, loss_cls, *args, axis=-1, flatten=True, floatify=False, is_2d=True, **kwargs):
        store_attr("axis,flatten,floatify,is_2d")
        self.func = loss_cls(*args, **kwargs)
        functools.update_wrapper(self, self.func)

    def __repr__(self): return f"FlattenedLoss of {self.func}"
    @property
    def reduction(self): return self.func.reduction
    @reduction.setter
    def reduction(self, v): self.func.reduction = v

    def _contiguous(self, x):
        return TensorBase(x.transpose(self.axis, -1).contiguous()) if isinstance(x, torch.Tensor) else x

    def __call__(self, inp, targ, **kwargs):
        inp, targ = map(self._contiguous, (inp, targ))
        if self.floatify and targ.dtype != torch.float16:
            targ = targ.float()
        if targ.dtype in [torch.int8, torch.int16, torch.int32]:
            targ = targ.long()
        if self.flatten:
            inp = inp.view(-1, inp.shape[-1]) if self.is_2d else inp.view(-1)
        return self.func.__call__(inp, targ.view(-1) if self.flatten else targ, **kwargs)


# Wrapping a general loss function inside of `BaseLoss` provides extra functionalities to your loss functions:
# - flattens the tensors before trying to take the losses since it's more convenient (with a potential tranpose to put `axis` at the end)
# - a potential `activation` method that tells the library if there is an activation fused in the loss (useful for inference and methods such as `Learner.get_preds` or `Learner.predict`)
# - a potential <code>decodes</code> method that is used on predictions in inference (for instance, an argmax in classification)

# The `args` and `kwargs` will be passed to `loss_cls` during the initialization to instantiate a loss function. `axis` is put at the end for losses like softmax that are often performed on the last axis. If `floatify=True`, the `targs` will be converted to floats (useful for losses that only accept float targets like `BCEWithLogitsLoss`), and `is_2d` determines if we flatten while keeping the first dimension (batch size) or completely flatten the input. We want the first for losses like Cross Entropy, and the second for pretty much anything else.

# export
@delegates()
class CrossEntropyLossFlat(BaseLoss):
    "Same as `nn.CrossEntropyLoss`, but flattens input and target."
    y_int = True
    @use_kwargs_dict(keep=True, weight=None, ignore_index=-100, reduction='mean')
    def __init__(self, *args, axis=-1, **kwargs): super().__init__(nn.CrossEntropyLoss, *args, axis=axis, **kwargs)
    def decodes(self, x): return x.argmax(dim=self.axis)
    def activation(self, x): return F.softmax(x, dim=self.axis)


# +
tst = CrossEntropyLossFlat()
output = torch.randn(32, 5, 10)
target = torch.randint(0, 10, (32, 5))
# nn.CrossEntropy would fail with those two tensors, but not our flattened version.
_ = tst(output, target)
test_fail(lambda x: nn.CrossEntropyLoss()(output, target))

# Associated activation is softmax
test_eq(tst.activation(output), F.softmax(output, dim=-1))
# This loss function has a decodes which is argmax
test_eq(tst.decodes(output), output.argmax(dim=-1))

# +
# In a segmentation task, we want to take the softmax over the channel dimension
tst = CrossEntropyLossFlat(axis=1)
output = torch.randn(32, 5, 128, 128)
target = torch.randint(0, 5, (32, 128, 128))
_ = tst(output, target)

test_eq(tst.activation(output), F.softmax(output, dim=1))
test_eq(tst.decodes(output), output.argmax(dim=1))


# -

# [Focal Loss](https://arxiv.org/pdf/1708.02002.pdf) is the same as cross entropy except easy-to-classify observations are down-weighted in the loss calculation. The strength of down-weighting is proportional to the size of the `gamma` parameter. Put another way, the larger `gamma` the less the easy-to-classify observations contribute to the loss.

# export
class FocalLossFlat(CrossEntropyLossFlat):
    """
    Same as CrossEntropyLossFlat but with focal paramter, `gamma`. Focal loss is introduced by Lin et al. 
    https://arxiv.org/pdf/1708.02002.pdf. Note the class weighting factor in the paper, alpha, can be 
    implemented through pytorch `weight` argument in nn.CrossEntropyLoss.
    """
    y_int = True
    @use_kwargs_dict(keep=True, weight=None, ignore_index=-100, reduction='mean')
    def __init__(self, *args, gamma=2, axis=-1, **kwargs):
        self.gamma = gamma
        self.reduce = kwargs.pop('reduction') if 'reduction' in kwargs else 'mean'
        super().__init__(*args, reduction='none', axis=axis, **kwargs)

    def __call__(self, inp, targ, **kwargs):
        ce_loss = super().__call__(inp, targ, **kwargs)
        pt = torch.exp(-ce_loss)
        fl_loss = (1 - pt)**self.gamma * ce_loss
        return fl_loss.mean() if self.reduce == 'mean' else fl_loss.sum() if self.reduce == 'sum' else fl_loss


# Compare focal loss with gamma = 0 to cross entropy
fl = FocalLossFlat(gamma=0)
ce = CrossEntropyLossFlat()
output = torch.randn(32, 5, 10)
target = torch.randint(0, 10, (32, 5))
test_close(fl(output, target), ce(output, target))
# Test focal loss with gamma > 0 is different than cross entropy
fl = FocalLossFlat(gamma=2)
test_ne(fl(output, target), ce(output, target))

# In a segmentation task, we want to take the softmax over the channel dimension
fl = FocalLossFlat(gamma=0, axis=1)
ce = CrossEntropyLossFlat(axis=1)
output = torch.randn(32, 5, 128, 128)
target = torch.randint(0, 5, (32, 128, 128))
test_close(fl(output, target), ce(output, target), eps=1e-4)
test_eq(fl.activation(output), F.softmax(output, dim=1))
test_eq(fl.decodes(output), output.argmax(dim=1))


# export
@delegates()
class BCEWithLogitsLossFlat(BaseLoss):
    "Same as `nn.BCEWithLogitsLoss`, but flattens input and target."
    @use_kwargs_dict(keep=True, weight=None, reduction='mean', pos_weight=None)
    def __init__(self, *args, axis=-1, floatify=True, thresh=0.5, **kwargs):
        if kwargs.get('pos_weight', None) is not None and kwargs.get('flatten', None) is True:
            raise ValueError("`flatten` must be False when using `pos_weight` to avoid a RuntimeError due to shape mismatch")
        if kwargs.get('pos_weight', None) is not None:
            kwargs['flatten'] = False
        super().__init__(nn.BCEWithLogitsLoss, *args, axis=axis, floatify=floatify, is_2d=False, **kwargs)
        self.thresh = thresh

    def decodes(self, x): return x > self.thresh
    def activation(self, x): return torch.sigmoid(x)


# +
tst = BCEWithLogitsLossFlat()
output = torch.randn(32, 5, 10)
target = torch.randn(32, 5, 10)
# nn.BCEWithLogitsLoss would fail with those two tensors, but not our flattened version.
_ = tst(output, target)
test_fail(lambda x: nn.BCEWithLogitsLoss()(output, target))
output = torch.randn(32, 5)
target = torch.randint(0, 2, (32, 5))
# nn.BCEWithLogitsLoss would fail with int targets but not our flattened version.
_ = tst(output, target)
test_fail(lambda x: nn.BCEWithLogitsLoss()(output, target))

tst = BCEWithLogitsLossFlat(pos_weight=torch.ones(10))
output = torch.randn(32, 5, 10)
target = torch.randn(32, 5, 10)
_ = tst(output, target)
test_fail(lambda x: nn.BCEWithLogitsLoss()(output, target))

# Associated activation is sigmoid
test_eq(tst.activation(output), torch.sigmoid(output))


# -

# export
@use_kwargs_dict(weight=None, reduction='mean')
def BCELossFlat(*args, axis=-1, floatify=True, **kwargs):
    "Same as `nn.BCELoss`, but flattens input and target."
    return BaseLoss(nn.BCELoss, *args, axis=axis, floatify=floatify, is_2d=False, **kwargs)


tst = BCELossFlat()
output = torch.sigmoid(torch.randn(32, 5, 10))
target = torch.randint(0, 2, (32, 5, 10))
_ = tst(output, target)
test_fail(lambda x: nn.BCELoss()(output, target))


# export
@use_kwargs_dict(reduction='mean')
def MSELossFlat(*args, axis=-1, floatify=True, **kwargs):
    "Same as `nn.MSELoss`, but flattens input and target."
    return BaseLoss(nn.MSELoss, *args, axis=axis, floatify=floatify, is_2d=False, **kwargs)


tst = MSELossFlat()
output = torch.sigmoid(torch.randn(32, 5, 10))
target = torch.randint(0, 2, (32, 5, 10))
_ = tst(output, target)
test_fail(lambda x: nn.MSELoss()(output, target))

# hide
# cuda
# Test losses work in half precision
if torch.cuda.is_available():
    output = torch.sigmoid(torch.randn(32, 5, 10)).half().cuda()
    target = torch.randint(0, 2, (32, 5, 10)).half().cuda()
    for tst in [BCELossFlat(), MSELossFlat()]:
        _ = tst(output, target)


# export
@use_kwargs_dict(reduction='mean')
def L1LossFlat(*args, axis=-1, floatify=True, **kwargs):
    "Same as `nn.L1Loss`, but flattens input and target."
    return BaseLoss(nn.L1Loss, *args, axis=axis, floatify=floatify, is_2d=False, **kwargs)


# export
class LabelSmoothingCrossEntropy(Module):
    y_int = True

    def __init__(self, eps: float=0.1, weight=None, reduction='mean'):
        store_attr()

    def forward(self, output, target):
        c = output.size()[1]
        log_preds = F.log_softmax(output, dim=1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=1)  # We divide by that size at the return line so sum and not mean
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * F.nll_loss(log_preds, target.long(), weight=self.weight, reduction=self.reduction)

    def activation(self, out): return F.softmax(out, dim=-1)
    def decodes(self, out): return out.argmax(dim=-1)


lmce = LabelSmoothingCrossEntropy()
output = torch.randn(32, 5, 10)
target = torch.randint(0, 10, (32, 5))
test_eq(lmce(output.flatten(0, 1), target.flatten()), lmce(output.transpose(-1, -2), target))


# On top of the formula we define:
# - a `reduction` attribute, that will be used when we call `Learner.get_preds`
# - `weight` attribute to pass to BCE.
# - an `activation` function that represents the activation fused in the loss (since we use cross entropy behind the scenes). It will be applied to the output of the model when calling `Learner.get_preds` or `Learner.predict`
# - a <code>decodes</code> function that converts the output of the model to a format similar to the target (here indices). This is used in `Learner.predict` and `Learner.show_results` to decode the predictions

# export
@delegates()
class LabelSmoothingCrossEntropyFlat(BaseLoss):
    "Same as `LabelSmoothingCrossEntropy`, but flattens input and target."
    y_int = True
    @use_kwargs_dict(keep=True, eps=0.1, reduction='mean')
    def __init__(self, *args, axis=-1, **kwargs): super().__init__(LabelSmoothingCrossEntropy, *args, axis=axis, **kwargs)
    def activation(self, out): return F.softmax(out, dim=-1)
    def decodes(self, out): return out.argmax(dim=-1)


# ## Export -

# hide
notebook2script()
