# coding: utf-8
# # Application fields
from fastai.gen_doc.nbdoc import *
# The fastai library allows you to train a [`Model`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) on a certain [`DataBunch`](/basic_data.html#DataBunch) very easily by binding them together inside a [`Learner`](/basic_train.html#Learner) object. This module regroups the tools the library provides to help you preprocess and group your data in this format.
#
# ## [`collab`](/collab.html#collab)
#
# This submodule handles the collaborative filtering problems.
#
# ## [`tabular`](/tabular.html#tabular)
#
# This sub-package deals with tabular (or structured) data.
#
# ## [`text`](/text.html#text)
#
# This sub-package contains everything you need for Natural Language Processing.
#
# ## [`vision`](/vision.html#vision)
#
# This sub-package contains the classes that deal with Computer Vision.
#
# ## Module structure
#
# In each case (except for [`collab`](/collab.html#collab)), the module is organized this way:
#
# ### [`transform`](/tabular.transform.html#tabular.transform)
#
# This sub-module deals with the pre-processing (data augmentation for images, cleaning for tabular data, tokenizing and numericalizing for text).
#
# ### [`data`](/data.html#data)
#
# This sub-module defines the dataset class(es) to deal with this kind of data.
#
# ### [`models`](/models.html#models)
#
# This sub-module defines the specific models used for this kind of data.
#
# ### [`learner`](/text.learner.html#text.learner)
#
# When it exists, this sub-module contains functions will directly bind this data with a suitable model and add the necessary callbacks.
