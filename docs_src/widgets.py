
# coding: utf-8

# # Widgets

from fastai import *
from fastai.vision import *
from fastai.widgets import DatasetFormatter, ImageCleaner


# fastai offers several widgets to support the workflow of a deep learning practitioner. The purpose of the widgets are to help you organize, clean, and prepare your data for your model. Widgets are separated by data type.

# ## Images

# ### DatasetFormatter
# The [`DatasetFormatter`](/widgets.image_cleaner.html#DatasetFormatter) class prepares your image dataset for widgets by returning a formatted [`DatasetTfm`](/vision.data.html#DatasetTfm) based on the [`DatasetType`](/basic_data.html#DatasetType) specified. Use `from_toplosses` to grab the most problematic images directly from your learner. Optionally, you can restrict the formatted dataset returned to `n_imgs`.
#
# Specify the [`DatasetType`](/basic_data.html#DatasetType) you'd like to process:
# - DatasetType.Train
# - DatasetType.Valid
# - DatasetType.Test

path = untar_data(URLs.MNIST_SAMPLE)
data = ImageDataBunch.from_folder(path)


learner = create_cnn(data, models.resnet18, metrics=[accuracy])
ds, idxs = DatasetFormatter().from_toplosses(learner, ds_type=DatasetType.Valid)


# ### ImageCleaner

# [`ImageDeleter`](/widgets.image_cleaner.html#ImageDeleter) is for cleaning up images that don't belong in your dataset. It renders images in a row and gives you the opportunity to delete the file from your file system.

ImageCleaner(ds, idxs)
