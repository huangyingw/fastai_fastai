
# coding: utf-8

# ## Image classification with Convolutional Neural Networks

# Put these at the top of every notebook, to get automatic reloading and inline plotting
# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'inline')


# This file contains all the main external libs we'll use
from fastai.imports import *

from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *


os.chdir(os.path.dirname(os.path.realpath(__file__)))
PATH = "data/dogscats/"
sz = 224
arch = vgg16
bs = 64


# Uncomment the below if you need to reset your precomputed activations
# !rm -rf {PATH}tmp


data = ImageClassifierData.from_paths(PATH, bs=bs, tfms=tfms_from_model(arch, sz))


learn = ConvLearner.pretrained(arch, data, precompute=True)


learn.fit(0.01, 3, cycle_len=1, saved_model_name='lesson1-vgg_1')


tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)


data = ImageClassifierData.from_paths(PATH, tfms=tfms, bs=bs, num_workers=4)
learn = ConvLearner.pretrained(arch, data, precompute=True)


learn.fit(1e-2, 2, saved_model_name='lesson1-vgg_2')


learn.precompute = False


learn.fit(1e-2, 1, cycle_len=1, saved_model_name='lesson1-vgg_3')


learn.unfreeze()


lr = np.array([1e-4, 1e-3, 1e-2])


learn.fit(lr, 1, cycle_len=1, saved_model_name='lesson1-vgg_4')


learn.fit(lr, 3, cycle_len=1, cycle_mult=2, saved_model_name='lesson1-vgg_5')


learn.fit(lr, 3, cycle_len=3, saved_model_name='lesson1-vgg_6')


log_preds, y = learn.TTA()
probs = np.mean(np.exp(log_preds), 0)accuracy_np(probs, y)
