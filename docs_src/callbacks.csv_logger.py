
# coding: utf-8

# # CSV Logger

from fastai import *
from fastai.vision import *
from fastai.gen_doc.nbdoc import *
from fastai.callbacks import *


show_doc(CSVLogger)


# First let's show an example of use, with a training on the usual MNIST dataset.

path = untar_data(URLs.MNIST_TINY)
data = ImageDataBunch.from_folder(path)
learn = Learner(data, simple_cnn((3, 16, 16, 2)), metrics=[accuracy, error_rate], callback_fns=[CSVLogger])


learn.fit(3)


# Training details have been saved in 'history.csv'.

learn.path.ls()


# Note that, as with all [`LearnerCallback`](/basic_train.html#LearnerCallback), you can access the object as an attribute of `learn` after it has been created. Here it's `learn.csv_logger`.

show_doc(CSVLogger.read_logged_file)


learn.csv_logger.read_logged_file()


# ## Undocumented Methods - Methods moved below this line will intentionally be hidden

show_doc(CSVLogger.on_train_end)


show_doc(CSVLogger.on_epoch_end)


show_doc(CSVLogger.on_train_begin)


# ## New Methods - Please document or move to the undocumented section
