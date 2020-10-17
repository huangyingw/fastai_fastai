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
from fastai.test_utils import *
from nbdev.showdoc import *
from torch.distributions.beta import Beta
from fastai.vision.models.xresnet import *
from fastai.vision.core import *
from fastai.callback.progress import *
from fastai.basics import *
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# +
# default_exp callback.mixup

# +
# export

# -

# hide


# # Mixup callback
#
# > Callback to apply MixUp data augmentation to your training

# ## MixupCallback -

# export
def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


# export
@log_args
class MixUp(Callback):
    run_after, run_valid = [Normalize], False
    def __init__(self, alpha=0.4): self.distrib = Beta(tensor(alpha), tensor(alpha))
    def before_fit(self):
        self.stack_y = getattr(self.learn.loss_func, 'y_int', False)
        if self.stack_y:
            self.old_lf, self.learn.loss_func = self.learn.loss_func, self.lf

    def after_fit(self):
        if self.stack_y:
            self.learn.loss_func = self.old_lf

    def before_batch(self):
        lam = self.distrib.sample((self.y.size(0),)).squeeze().to(self.x.device)
        lam = torch.stack([lam, 1 - lam], 1)
        self.lam = lam.max(1)[0]
        shuffle = torch.randperm(self.y.size(0)).to(self.x.device)
        xb1, self.yb1 = tuple(L(self.xb).itemgot(shuffle)), tuple(L(self.yb).itemgot(shuffle))
        nx_dims = len(self.x.size())
        self.learn.xb = tuple(L(xb1, self.xb).map_zip(torch.lerp, weight=unsqueeze(self.lam, n=nx_dims - 1)))
        if not self.stack_y:
            ny_dims = len(self.y.size())
            self.learn.yb = tuple(L(self.yb1, self.yb).map_zip(torch.lerp, weight=unsqueeze(self.lam, n=ny_dims - 1)))

    def lf(self, pred, *yb):
        if not self.training:
            return self.old_lf(pred, *yb)
        with NoneReduce(self.old_lf) as lf:
            loss = torch.lerp(lf(pred, *self.yb1), lf(pred, *yb), self.lam)
        return reduce_loss(loss, getattr(self.old_lf, 'reduction', 'mean'))



path = untar_data(URLs.MNIST_TINY)
items = get_image_files(path)
tds = Datasets(items, [PILImageBW.create, [parent_label, Categorize()]], splits=GrandparentSplitter()(items))
dls = tds.dataloaders(after_item=[ToTensor(), IntToFloatTensor()])

# +
mixup = MixUp(0.5)
with Learner(dls, nn.Linear(3, 4), loss_func=CrossEntropyLossFlat(), cbs=mixup) as learn:
    learn.epoch, learn.training = 0, True
    learn.dl = dls.train
    b = dls.one_batch()
    learn._split(b)
    learn('before_batch')

_, axs = plt.subplots(3, 3, figsize=(9, 9))
dls.show_batch(b=(mixup.x, mixup.y), ctxs=axs.flatten())
# -

# ## Export -

# hide
notebook2script()
