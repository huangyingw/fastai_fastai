
# coding: utf-8

# # Torch Core

# This module contains all the basic functions we need in other modules of the fastai library (split with [`core`](/core.html#core) that contains the ones not requiring pytorch). Its documentation can easily be skipped at a first read, unless you want to know what a given fuction does.

from fastai.gen_doc.nbdoc import *
from fastai.torch_core import *


# ## Global constants

# `AdamW = partial(optim.Adam, betas=(0.9,0.99))` <div style="text-align: right"><a href="https://github.com/fastai/fastai/blob/master/fastai/torch_core.py#L43">[source]</a></div>

# `bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)` <div style="text-align: right"><a href="https://github.com/fastai/fastai/blob/master/fastai/torch_core.py#L41">[source]</a></div>

# `default_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')` <div style="text-align: right"><a href="https://github.com/fastai/fastai/blob/master/fastai/torch_core.py#L42">[source]</a></div>

# ## Functions that operate conversions

show_doc(flatten_model, full_name='flatten')


# Flattens all the layers of `m`.

show_doc(model2half)


show_doc(np2model_tensor)


show_doc(requires_grad, doc_string=False)


# If `b` is None, returns the [`requires_grad`](/torch_core.html#requires_grad) state of the first layer of `m`. Otherwise, sets `requires_grad=b` in all children of `m`.

show_doc(tensor)


# Ensures `x` is a torch `Tensor`.

show_doc(to_data)


show_doc(to_detach)


show_doc(to_device)


show_doc(to_half, doc_string=False)


# Put the input of the batch `b` in half precision.

show_doc(to_np)


# Convert `x` to a numpy array.

# ## Functions to deal with model initialization

show_doc(apply_init)


show_doc(apply_leaf)


show_doc(cond_init)


show_doc(in_channels)


# ## Functions to get information of a model

show_doc(children)


show_doc(first_layer)


show_doc(last_layer)


show_doc(num_children)


show_doc(range_children)


show_doc(trainable_params)


# ## Functions to deal with BatchNorm layers

show_doc(bn2float)


show_doc(set_bn_eval)


show_doc(split_bn_bias)


# ## Other functions

show_doc(calc_loss)


show_doc(data_collate)


show_doc(model_type)


show_doc(np_address)


show_doc(split_model, doc_string=False)


# Splits the `model` according to the layer in `splits`. If `splits` are layers, the model is split at those (not included) sequentially. If `want_idxs` is True, the corresponding indexes are returned. If `splits` are lists of layers, the model is split according to those.

show_doc(split_model_idx)


show_doc(trange_of)


# Return a tensor from a range that has the same length as `x`.

# ## Undocumented Methods - Methods moved below this line will intentionally be hidden

show_doc(tensor__array__)


# ## New Methods - Please document or move to the undocumented section

show_doc(init_default)


show_doc(log_uniform)


show_doc(grab_idx)


show_doc(uniform_int)


show_doc(to_cpu)


show_doc(logit)


show_doc(FloatItem)


show_doc(logit_)


show_doc(rand_bool)


show_doc(one_param)


show_doc(uniform)
