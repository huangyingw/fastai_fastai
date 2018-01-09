
# coding: utf-8

# ## Dogs v Cats

get_ipython().magic(u'reload_ext autoreload')
get_ipython().magic(u'autoreload 2')
get_ipython().magic(u'matplotlib inline')

from fastai.imports import *

from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *


PATH = "data/dogscats/"
sz = 224
arch = resnet34
bs = 64


m = arch(True)


m


m = nn.Sequential(*children(m)[:-2],
                  nn.Conv2d(512, 2, 3, padding=1),
                  nn.AdaptiveAvgPool2d(1), Flatten(),
                  nn.LogSoftmax())


tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
data = ImageClassifierData.from_paths(PATH, tfms=tfms, bs=bs)


learn = ConvLearner.from_model_data(m, data)


learn.freeze_to(-4)


m[-1].trainable


m[-4].trainable


learn.fit(0.01, 1)


learn.fit(0.01, 1, cycle_len=1)


# ## CAM

class SaveFeatures():
    features = None

    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output): self.features = to_np(output)

    def remove(self): self.hook.remove()


x, y = next(iter(data.val_dl))
x, y = x[None, 1], y[None, 1]

vx = Variable(x.cuda(), requires_grad=True)


dx = data.val_ds.denorm(x)[0]
plt.imshow(dx)


sf = SaveFeatures(m[-4])
py = m(Variable(x.cuda()))
sf.remove()

py = np.exp(to_np(py)[0])
py


feat = np.maximum(0, sf.features[0])
feat.shape


f2 = np.dot(np.rollaxis(feat, 0, 3), py)
f2 -= f2.min()
f2 /= f2.max()
f2


plt.imshow(dx)
plt.imshow(scipy.misc.imresize(f2, dx.shape), alpha=0.5, cmap='hot')


# ## Model

learn.unfreeze()
learn.bn_freeze(True)


lr = np.array([1e-6, 1e-4, 1e-2])


learn.fit(lr, 2, cycle_len=1)


accuracy(*learn.TTA())


learn.fit(lr, 2, cycle_len=1)


accuracy(*learn.TTA())
