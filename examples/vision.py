# coding: utf-8
from fastai.vision import *   # Quick access to computer vision functionality
# # Vision example
# Images can be in labeled folders, or a single folder with a CSV.
path = untar_data(URLs.MNIST_SAMPLE)
path
# ### Image folder version
# Create a `DataBunch`, optionally with transforms:
data = ImageDataBunch.from_folder(path, ds_tfms=(rand_pad(2, 28), []), bs=64)
data.normalize(imagenet_stats)
img, label = data.train_ds[0]
img
# Create and fit a `Learner`:
learn = create_cnn(data, models.resnet18, metrics=accuracy)
learn.fit_one_cycle(1, 0.01)
accuracy(*learn.get_preds())
# ### CSV version
# Same as above, using CSV instead of folder name for labels
data = ImageDataBunch.from_csv(path, ds_tfms=(rand_pad(2, 28), []), bs=64)
data.normalize(imagenet_stats)
img, label = data.train_ds[0]
img
learn = create_cnn(data, models.resnet18, metrics=accuracy)
learn.fit_one_cycle(1, 0.01)
