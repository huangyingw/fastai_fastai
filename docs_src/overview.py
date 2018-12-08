
# coding: utf-8

from fastai.gen_doc.nbdoc import *


jekyll_note('To get started with fastai, have a look at the <a href="/training">training overview</a>. The documentation below covers some lower-level details.')


# # Core modules of fastai

# The basic foundations needed in several parts of the library are provided by these modules:
#
# ## [`basic_data`](/basic_data.html#basic_data)
#
# This module defines the basic [`DataBunch`](/basic_data.html#DataBunch) class which is what will be needed to create a [`Learner`](/basic_train.html#Learner) object with a model. It also defines the [`DeviceDataLoader`](/basic_data.html#DeviceDataLoader), a class that wraps a pytorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) to put batches on the right device.

# ## [`layers`](/layers.html#layers)
#
# This module contains the definitions of basic custom layers we need in most of our models, as well as a few helper functions to create simple blocks.

# Most of the documentation of the following two modules can be skipped at a first read, unless you specifically want to know what a certain function is doing.

# ## [`core`](/core.html#core)
#
# This module contains the most basic functions and imports, notably:
# - pandas as pd
# - numpy as np
# - matplotlib.pyplot as plt

# ## [`torch_core`](/torch_core.html#torch_core)
#
# This module contains the most basic functions and imports that use pytorch. We follow pytorch naming conventions, mainly:
# - torch.nn as nn
# - torch.optim as optim
# - torch.nn.functional as F
