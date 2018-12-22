# coding: utf-8
# # NasNet Dogs v Cats
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
from fastai.conv_learner import *
PATH = "data/dogscats/"
sz = 224; bs = 48
def nasnet(pre): return nasnetalarge(pretrained='imagenet' if pre else None)
model_features[nasnet] = 4032 * 2
stats = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
tfms = tfms_from_stats(stats, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
data = ImageClassifierData.from_paths(PATH, tfms=tfms, bs=bs)
learn = ConvLearner.pretrained(nasnet, data, precompute=True, xtra_fc=[], ps=0.5)
get_ipython().run_line_magic('time', 'learn.fit(1e-2, 2)')
learn.precompute = False
learn.bn_freeze = True
get_ipython().run_line_magic('time', 'learn.fit(1e-2, 1, cycle_len=1)')
learn.save('nas_pre')
def freeze_to(m, n):
    c = children(m[0])
    for l in c: set_trainable(l, False)
    for l in c[n:]: set_trainable(l, True)
freeze_to(learn.model, 17)
learn.fit([1e-5, 1e-4, 1e-2], 1, cycle_len=1)
learn.save('nas')
