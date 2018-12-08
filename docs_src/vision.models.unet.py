
# coding: utf-8

# # Dynamic U-Net

# This module builds a dynamic [U-Net](https://arxiv.org/abs/1505.04597) from any backbone pretrained on ImageNet, automatically inferring the intermediate sizes.

from fastai.gen_doc.nbdoc import *
from fastai.vision.models.unet import *


# ![U-Net architecure](imgs/u-net-architecture.png)
#
# This is the original U-Net. The difference here is that the left part is a pretrained model.

show_doc(DynamicUnet, doc_string=False)


# Builds a U-Net from a given `encoder` (that can be a pretrained model) and with a final output of `n_classes`. During the initialization, it uses [`Hooks`](/callbacks.hooks.html#Hooks) to determine the intermediate features sizes by passing a dummy input throught the model.

show_doc(UnetBlock, doc_string=False)


# Builds a U-Net block that receives the output of the last block to be upsampled (size `up_in_c`) and the activations features from an intermediate layer of the `encoder` (size `x_in_c`, this is the lateral connection). The `hook` is set to this intermediate layer to store the output needed for this block.

# ## Undocumented Methods - Methods moved below this line will intentionally be hidden

# ## New Methods - Please document or move to the undocumented section

show_doc(UnetBlock.forward)
