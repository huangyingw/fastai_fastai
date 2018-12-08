
# coding: utf-8

# # Get your data ready for training

# This module defines the basic [`DataBunch`](/basic_data.html#DataBunch) object that is used inside [`Learner`](/basic_train.html#Learner) to train a model. This is the generic class, that can take any kind of fastai [`Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) or [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader). You'll find helpful functions in the data module of every application to directly create this [`DataBunch`](/basic_data.html#DataBunch) for you.

from fastai.gen_doc.nbdoc import *
from fastai import *


show_doc(DataBunch)


# It also ensure all the dataloaders are on `device` and apply to them `tfms` as batch are drawn (like normalization). `path` is used internally to store temporary files, `collate_fn` is passed to the pytorch `Dataloader` (replacing the one there) to explain how to collate the samples picked for a batch. By default, it applies data to the object sent (see in [`vision.image`](/vision.image.html#vision.image) or the [data block API](/data_block.html) why this can be important).
#
# `train_dl`, `valid_dl` and optionally `test_dl` will be wrapped in [`DeviceDataLoader`](/basic_data.html#DeviceDataLoader).

# ### Factory method

show_doc(DataBunch.create)


# `num_workers` is the number of CPUs to use, `tfms`, `device` and `collate_fn` are passed to the init method.

# ### Visualization

show_doc(DataBunch.show_batch)


# ### Grabbing some data

show_doc(DataBunch.dl)


show_doc(DataBunch.one_batch)


show_doc(DataBunch.one_item)


# ### Empty [`DataBunch`](/basic_data.html#DataBunch) for inference

show_doc(DataBunch.export)


show_doc(DataBunch.load_empty, full_name='load_empty')


# This method should be used to create a [`DataBunch`](/basic_data.html#DataBunch) at inference, see the corresponding [tutorial](/tutorial.inference.html).

# ### Dataloader transforms

show_doc(DataBunch.add_tfm)


# Adds a transform to all dataloaders.

show_doc(DeviceDataLoader)


# Put the batches of `dl` on `device` after applying an optional list of `tfms`. `collate_fn` will replace the one of `dl`. All dataloaders of a [`DataBunch`](/basic_data.html#DataBunch) are of this type.

# ### Factory method

show_doc(DeviceDataLoader.create)


# The given `collate_fn` will be used to put the samples together in one batch (by default it grabs their data attribute). `shuffle` means the dataloader will take the samples randomly if that flag is set to `True`, or in the right order otherwise. `tfms` are passed to the init method. All `kwargs` are passed to the pytorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) class initialization.

# ### Methods

show_doc(DeviceDataLoader.add_tfm)


show_doc(DeviceDataLoader.remove_tfm)


show_doc(DeviceDataLoader.new)


show_doc(DatasetType, doc_string=False)


# Internal enumerator to name the training, validation and test dataset/dataloader.

# ## Undocumented Methods - Methods moved below this line will intentionally be hidden

show_doc(DeviceDataLoader.proc_batch)


show_doc(DeviceDataLoader.collate_fn)


# ## New Methods - Please document or move to the undocumented section
