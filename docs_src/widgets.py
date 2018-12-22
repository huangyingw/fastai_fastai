# coding: utf-8
# # Widgets
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from fastai.vision import *
from fastai.widgets import DatasetFormatter, ImageCleaner
# fastai offers several widgets to support the workflow of a deep learning practitioner. The purpose of the widgets are to help you organize, clean, and prepare your data for your model. Widgets are separated by data type.
# ## Images
# ### DatasetFormatter
# The [`DatasetFormatter`](/widgets.image_cleaner.html#DatasetFormatter) class prepares your image dataset for widgets by returning a formatted [`DatasetTfm`](/vision.data.html#DatasetTfm) based on the [`DatasetType`](/basic_data.html#DatasetType) specified. Use `from_toplosses` to grab the most problematic images directly from your learner. Optionally, you can restrict the formatted dataset returned to `n_imgs`.
path = untar_data(URLs.MNIST_SAMPLE)
data = ImageDataBunch.from_folder(path)
learn = create_cnn(data, models.resnet18, metrics=error_rate)
learn.fit_one_cycle(2)
learn.save('stage-1')
# We create a databunch with all the data in the training set and no validation set (DatasetFormatter uses only the training set)
db = (ImageItemList.from_folder(path)
                   .no_split()
                   .label_from_folder()
                   .databunch())
learn = create_cnn(db, models.resnet18, metrics=[accuracy])
learn.load('stage-1');
# ### ImageCleaner
# [`ImageCleaner`](/widgets.image_cleaner.html#ImageCleaner) is for cleaning up images that don't belong in your dataset. It renders images in a row and gives you the opportunity to delete the file from your file system. To use [`ImageCleaner`](/widgets.image_cleaner.html#ImageCleaner) we must first use `DatasetFormatter().from_toplosses` to get the suggested indices for misclassified images.
ds, idxs = DatasetFormatter().from_toplosses(learn)
ImageCleaner(ds, idxs, path)
