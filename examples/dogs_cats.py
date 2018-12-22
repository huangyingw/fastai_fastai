# coding: utf-8
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from fastai.vision import *
# # Dogs and cats
# ## Resnet 34
path = untar_data(URLs.DOGS)
path
data = ImageDataBunch.from_folder(path, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
data.show_batch(rows=4)
learn = create_cnn(data, models.resnet34, metrics=accuracy)
learn.fit_one_cycle(1)
learn.unfreeze()
learn.fit_one_cycle(6, slice(1e-5, 3e-4), pct_start=0.05)
accuracy(*learn.TTA())
# ## rn50
learn = create_cnn(data, models.resnet50, metrics=accuracy)
learn.fit_one_cycle(1)
learn.unfreeze()
learn.fit_one_cycle(6, slice(1e-5, 3e-4), pct_start=0.05)
accuracy(*learn.TTA())
