
# coding: utf-8

# ## Image classification with Convolutional Neural Networks

# Put these at the top of every notebook, to get automatic reloading and
# inline plotting
get_ipython().magic(u'reload_ext autoreload')
get_ipython().magic(u'autoreload 2')
get_ipython().magic(u'matplotlib inline')


# This file contains all the main external libs we'll use
from fastai.imports import *

from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *


PATH = "data/dogscats/"
sz = 224
arch = vgg16
bs = 64


# Uncomment the below if you need to reset your precomputed activations
# !rm -rf {PATH}tmp


data = ImageClassifierData.from_paths(PATH, tfms=tfms_from_model(arch, sz))


learn = ConvLearner.pretrained(arch, data, precompute=True)


learn.fit(0.01, 3, cycle_len=1)


tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)


data = ImageClassifierData.from_paths(PATH, tfms=tfms, bs=bs, num_workers=4)
learn = ConvLearner.pretrained(arch, data, precompute=True)


learn.fit(1e-2, 2)


learn.precompute = False


learn.fit(1e-2, 1, cycle_len=1)


learn.unfreeze()


lr = np.array([1e-4, 1e-3, 1e-2])


learn.fit(lr, 1, cycle_len=1)


learn.fit(lr, 3, cycle_len=1, cycle_mult=2)


learn.fit(lr, 3, cycle_len=3)


log_preds, y = learn.TTA()
accuracy(log_preds, y)
