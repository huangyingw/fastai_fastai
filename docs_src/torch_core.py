# coding: utf-8
# # Torch Core
# This module contains all the basic functions we need in other modules of the fastai library (split with [`core`](/core.html#core) that contains the ones not requiring pytorch). Its documentation can easily be skipped at a first read, unless you want to know what a given fuction does.
from fastai.gen_doc.nbdoc import *
from fastai.torch_core import *
# ## Global constants
# `AdamW = partial(optim.Adam, betas=(0.9,0.99))` <div style="text-align: right"><a href="https://github.com/fastai/fastai/blob/master/fastai/torch_core.py#L43">[source]</a></div>
# `bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)` <div style="text-align: right"><a href="https://github.com/fastai/fastai/blob/master/fastai/torch_core.py#L41">[source]</a></div>
# `defaults.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')` <div style="text-align: right"><a href="https://github.com/fastai/fastai/blob/master/fastai/torch_core.py#L62">[source]</a></div>
# If you are trying to make fastai run on the CPU, simply change the default device: `defaults.device = 'cpu'`.
#
# Alternatively, if not using wildcard imports: `fastai.torch_core.defaults.device = 'cpu'`.
# ## Functions that operate conversions
show_doc(flatten_model, full_name='flatten')
# Flattens all the layers of `m`.
show_doc(model2half)
show_doc(np2model_tensor)
show_doc(requires_grad)
show_doc(tensor)
show_doc(to_cpu)
show_doc(to_data)
show_doc(to_detach)
show_doc(to_device)
show_doc(to_half)
show_doc(to_np)
show_doc(try_int)
# ## Functions to deal with model initialization
show_doc(apply_init)
show_doc(apply_leaf)
show_doc(cond_init)
show_doc(in_channels)
show_doc(init_default)
# ## Functions to get information of a model
show_doc(children)
show_doc(first_layer)
show_doc(last_layer)
show_doc(num_children)
show_doc(one_param)
show_doc(range_children)
show_doc(trainable_params)
# ## Functions to deal with BatchNorm layers
show_doc(bn2float)
show_doc(set_bn_eval)
show_doc(split_bn_bias)
# ## Functions to get random tensors
show_doc(log_uniform)
log_uniform(0.5, 2, (8,))
show_doc(rand_bool)
rand_bool(0.5, 8)
show_doc(uniform)
uniform(0, 1, (8,))
show_doc(uniform_int)
uniform_int(0, 2, (8,))
# ## Other functions
show_doc(FloatItem, title_level=3)
show_doc(calc_loss)
show_doc(data_collate)
show_doc(grab_idx)
show_doc(logit)
show_doc(logit_)
show_doc(model_type)
show_doc(np_address)
show_doc(split_model)
# If `splits` are layers, the model is split at those (not included) sequentially. If `want_idxs` is True, the corresponding indexes are returned. If `splits` are lists of layers, the model is split according to those.
show_doc(split_model_idx)
show_doc(trange_of)
# ## Undocumented Methods - Methods moved below this line will intentionally be hidden
show_doc(tensor__array__)
# ## New Methods - Please document or move to the undocumented section
