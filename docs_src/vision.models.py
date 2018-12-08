
# coding: utf-8

# # Computer Vision models zoo

from fastai.gen_doc.nbdoc import *
from fastai.vision.models.darknet import Darknet
from fastai.vision.models.wrn import wrn_22, WideResNet


# On top of the models offered by [torchivision](https://pytorch.org/docs/stable/torchvision/models.html), the fastai library has implementations for the following models:
#
# - Darknet architecture, which is the base of [Yolo v3](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
# - Unet architecture based on a pretrained model. The original unet is described [here](https://arxiv.org/abs/1505.04597), the model implementation is detailed in [`models.unet`](/vision.models.unet.html#vision.models.unet)
# - Wide resnets architectures, as introduced in [this article](https://arxiv.org/abs/1605.07146).

show_doc(Darknet, doc_string=False)


# Create a Darknet with blocks of sizes given in `num_blocks`, ending with `num_classes` and using `nf` initial features. Darknet53 uses `num_blocks = [1,2,8,8,4]`.

show_doc(WideResNet, doc_string=False)


# Create a wide resnet with blocks `num_groups` groups, each containing blocks of size `N`. `k` is the width of the resnet, `start_nf` the initial number of features. Dropout of `drop_p` is applied at the end of each block.

show_doc(wrn_22)


# Creates a wide resnet for CIFAR-10 with `num_groups=3`, `N=3`, `k=6` and `drop_p=0.`.
