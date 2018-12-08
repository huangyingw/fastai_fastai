
# coding: utf-8

# # Mixed precision training

# This module allows the forward and backward passes of your neural net to be done in fp16 (also known as *half precision*). This is particularly important if you have an NVIDIA GPU with [tensor cores](https://www.nvidia.com/en-us/data-center/tensorcore/), since it can speed up your training by 200% or more.

from fastai.gen_doc.nbdoc import *
from fastai.callbacks.fp16 import *
from fastai import *
from fastai.vision import *


# ## Overview

# To train your model in mixed precision you just have to call [`Learner.to_fp16`](/train.html#to_fp16), which converts the model and modifies the existing [`Learner`](/basic_train.html#Learner) to add [`MixedPrecision`](/callbacks.fp16.html#MixedPrecision).

show_doc(Learner.to_fp16)


# For example:

path = untar_data(URLs.MNIST_SAMPLE)
data = ImageDataBunch.from_folder(path)
model = simple_cnn((3, 16, 16, 2))
learn = Learner(data, model, metrics=[accuracy]).to_fp16()
learn.fit_one_cycle(1)


# Details about mixed precision training are available in [NVIDIA's documentation](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html). We will just summarize the basics here.
#
# The only parameter you may want to tweak is `loss_scale`. This is used to scale the loss up, so that it doesn't underflow fp16, leading to loss of accuracy (this is reversed for the final gradient calculation after converting back to fp32). Generally the default `512` works well, however. You can also enable or disable the flattening of the master parameter tensor with `flat_master=True`, however in our testing the different is negligible.
#
# Internally, the callback ensures that all model parameters (except batchnorm layers, which require fp32) are converted to fp16, and an fp32 copy is also saved. The fp32 copy (the `master` parameters) is what is used for actually updating with the optimizer; the fp16 parameters are used for calculating gradients. This helps avoid underflow with small learning rates.
#
# All of this is implemented by the following Callback.

show_doc(MixedPrecision)


# You don't have to call the following functions yourself - they're called by the callback framework automatically. They're just documented here so you can see exactly what the callback is doing.

show_doc(MixedPrecision.on_backward_begin)


show_doc(MixedPrecision.on_backward_end)


show_doc(MixedPrecision.on_loss_begin)


show_doc(MixedPrecision.on_step_end)


show_doc(MixedPrecision.on_train_begin)


show_doc(MixedPrecision.on_train_end)
