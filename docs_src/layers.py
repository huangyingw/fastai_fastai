# coding: utf-8
# # Model Layers
# This module contains many layer classes that we might be interested in using in our models. These layers complement the default [Pytorch layers](https://pytorch.org/docs/stable/nn.html) which we can also use as predefined layers.
from fastai.vision import *
from fastai.gen_doc.nbdoc import *
# ## Custom fastai modules
show_doc(AdaptiveConcatPool2d, title_level=3)
from fastai.gen_doc.nbdoc import *
from fastai.layers import *
# The output will be `2*sz`, or just 2 if `sz` is None.
# The [`AdaptiveConcatPool2d`](/layers.html#AdaptiveConcatPool2d) object uses adaptive average pooling and adaptive max pooling and concatenates them both. We use this because it provides the model with the information of both methods and improves performance. This technique is called `adaptive` because it allows us to decide on what output dimensions we want, instead of choosing the input's dimensions to fit a desired output size.
#
# Let's try training with Adaptive Average Pooling first, then with Adaptive Max Pooling and finally with the concatenation of them both to see how they fare in performance.
#
# We will first define a [`simple_cnn`](/layers.html#simple_cnn) using [Adapative Max Pooling](https://pytorch.org/docs/stable/nn.html#torch.nn.AdaptiveMaxPool2d) by changing the source code a bit.
path = untar_data(URLs.MNIST_SAMPLE)
data = ImageDataBunch.from_folder(path)
def simple_cnn_max(actns: Collection[int], kernel_szs: Collection[int]=None,
               strides: Collection[int]=None) -> nn.Sequential:
    "CNN with `conv2d_relu` layers defined by `actns`, `kernel_szs` and `strides`"
    nl = len(actns) - 1
    kernel_szs = ifnone(kernel_szs, [3] * nl)
    strides = ifnone(strides, [2] * nl)
    layers = [conv_layer(actns[i], actns[i + 1], kernel_szs[i], stride=strides[i])
        for i in range(len(strides))]
    layers.append(nn.Sequential(nn.AdaptiveMaxPool2d(1), Flatten()))
    return nn.Sequential(*layers)
model = simple_cnn_max((3, 16, 16, 2))
learner = Learner(data, model, metrics=[accuracy])
learner.fit(1)
# Now let's try with [Adapative Average Pooling](https://pytorch.org/docs/stable/nn.html#torch.nn.AdaptiveAvgPool2d) now.
def simple_cnn_avg(actns: Collection[int], kernel_szs: Collection[int]=None,
               strides: Collection[int]=None) -> nn.Sequential:
    "CNN with `conv2d_relu` layers defined by `actns`, `kernel_szs` and `strides`"
    nl = len(actns) - 1
    kernel_szs = ifnone(kernel_szs, [3] * nl)
    strides = ifnone(strides, [2] * nl)
    layers = [conv_layer(actns[i], actns[i + 1], kernel_szs[i], stride=strides[i])
        for i in range(len(strides))]
    layers.append(nn.Sequential(nn.AdaptiveAvgPool2d(1), Flatten()))
    return nn.Sequential(*layers)
model = simple_cnn_avg((3, 16, 16, 2))
learner = Learner(data, model, metrics=[accuracy])
learner.fit(1)
# Finally we will try with the concatenation of them both [`AdaptiveConcatPool2d`](/layers.html#AdaptiveConcatPool2d). We will see that, in fact, it increases our accuracy and decreases our loss considerably!
def simple_cnn(actns: Collection[int], kernel_szs: Collection[int]=None,
               strides: Collection[int]=None) -> nn.Sequential:
    "CNN with `conv2d_relu` layers defined by `actns`, `kernel_szs` and `strides`"
    nl = len(actns) - 1
    kernel_szs = ifnone(kernel_szs, [3] * nl)
    strides = ifnone(strides, [2] * nl)
    layers = [conv_layer(actns[i], actns[i + 1], kernel_szs[i], stride=strides[i])
        for i in range(len(strides))]
    layers.append(nn.Sequential(AdaptiveConcatPool2d(1), Flatten()))
    return nn.Sequential(*layers)
model = simple_cnn((3, 16, 16, 2))
learner = Learner(data, model, metrics=[accuracy])
learner.fit(1)
show_doc(Lambda, title_level=3)
# This is very useful to use functions as layers in our networks inside a [Sequential](https://pytorch.org/docs/stable/nn.html#torch.nn.Sequential) object. So, for example, say we want to apply a [log_softmax loss](https://pytorch.org/docs/stable/nn.html#torch.nn.functional.log_softmax) and we need to change the shape of our output batches to be able to use this loss. We can add a layer that applies the necessary change in shape by calling:
#
# `Lambda(lambda x: x.view(x.size(0),-1))`
# Let's see an example of how the shape of our output can change when we add this layer.
model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
)
model.cuda()
for xb, yb in data.train_dl:
    out = (model(*[xb]))
    print(out.size())
    break
model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    Lambda(lambda x: x.view(x.size(0), -1))
)
model.cuda()
for xb, yb in data.train_dl:
    out = (model(*[xb]))
    print(out.size())
    break
show_doc(Flatten)
# The function we build above is actually implemented in our library as [`Flatten`](/layers.html#Flatten). We can see that it returns the same size when we run it.
model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    Flatten(),
)
model.cuda()
for xb, yb in data.train_dl:
    out = (model(*[xb]))
    print(out.size())
    break
show_doc(PoolFlatten)
# We can combine these two final layers ([AdaptiveAvgPool2d](https://pytorch.org/docs/stable/nn.html#torch.nn.AdaptiveAvgPool2d) and [`Flatten`](/layers.html#Flatten)) by using [`PoolFlatten`](/layers.html#PoolFlatten).
model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1), nn.ReLU(),
    PoolFlatten()
)
model.cuda()
for xb, yb in data.train_dl:
    out = (model(*[xb]))
    print(out.size())
    break
# Another use we give to the Lambda function is to resize batches with [`ResizeBatch`](/layers.html#ResizeBatch) when we have a layer that expects a different input than what comes from the previous one.
show_doc(ResizeBatch)
a = torch.tensor([[1., -1.], [1., -1.]])
print(a)
out = ResizeBatch(4)
print(out(a))
show_doc(Debugger, title_level=3)
# The debugger module allows us to peek inside a network while its training and see in detail what is going on. We can see inputs, ouputs and sizes at any point in the network.
#
# For instance, if you run the following:
#
# ``` python
# model = nn.Sequential(
#     nn.Conv2d(3,  16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
#     Debugger(),
#     nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
#     nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1), nn.ReLU(),
# )
#
# model.cuda()
#
# learner = Learner(data, model, metrics=[accuracy])
# learner.fit(5)
# ```
# ... you'll see something like this:
#
# ```
# /home/ubuntu/fastai/fastai/layers.py(74)forward()
#      72     def forward(self,x:Tensor) -> Tensor:
#      73         set_trace()
# ---> 74         return x
#      75
#      76 class StdUpsample(nn.Module):
#
# ipdb>
# ```
show_doc(PixelShuffle_ICNR, title_level=3)
show_doc(MergeLayer, title_level=3)
show_doc(PartialLayer, title_level=3)
show_doc(SigmoidRange, title_level=3)
show_doc(SequentialEx, title_level=3)
show_doc(SelfAttention, title_level=3)
# ## Loss functions
show_doc(FlattenedLoss, title_level=3)
# Create an instance of `func` with `args` and `kwargs`. When passing an output and target, it
# - puts `axis` first in output and target with a transpose
# - casts the target to `float` is `floatify=True`
# - squeezes the `output` to two dimensions if `is_2d`, otherwise one dimension, squeezes the target to one dimension
# - applied the instance of `func`.
show_doc(BCEFlat)
show_doc(BCEWithLogitsFlat)
show_doc(CrossEntropyFlat)
show_doc(MSELossFlat)
show_doc(NoopLoss)
show_doc(WassersteinLoss)
# ## Helper functions to create modules
show_doc(bn_drop_lin, doc_string=False)
# The [`bn_drop_lin`](/layers.html#bn_drop_lin) function returns a sequence of [batch normalization](https://arxiv.org/abs/1502.03167), [dropout](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) and a linear layer. This custom layer is usually used at the end of a model.
#
# `n_in` represents the number of size of the input `n_out` the size of the output, `bn` whether we want batch norm or not, `p` is how much dropout and `actn` is an optional parameter to add an activation function at the end.
show_doc(conv2d)
show_doc(conv2d_trans)
show_doc(conv_layer, doc_string=False)
# The [`conv_layer`](/layers.html#conv_layer) function returns a sequence of [nn.Conv2D](https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d), [BatchNorm](https://arxiv.org/abs/1502.03167) and a ReLU or [leaky RELU](https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf) activation function.
#
# `n_in` represents the number of size of the input `n_out` the size of the output, `ks` kernel size, `stride` the stride with which we want to apply the convolutions. `bias` will decide if they have bias or not (if None, defaults to True unless using batchnorm). `norm_type` selects type of normalization (or `None`). If `leaky` is None, the activation is a standard `ReLU`, otherwise it's a `LearkyReLU` of slope `leaky`. Finally if `transpose=True`, the convolution is replaced by a `ConvTranspose2D`.
show_doc(embedding, doc_string=False)
# Create an [embedding layer](https://arxiv.org/abs/1711.09160) with input size `ni` and output size `nf`.
show_doc(relu)
show_doc(res_block)
show_doc(sigmoid_range)
show_doc(simple_cnn)
# ## Initialization of modules
show_doc(batchnorm_2d)
show_doc(icnr)
show_doc(trunc_normal_)
show_doc(icnr)
show_doc(NormType)
# ## Undocumented Methods - Methods moved below this line will intentionally be hidden
show_doc(Debugger.forward)
show_doc(Lambda.forward)
show_doc(AdaptiveConcatPool2d.forward)
show_doc(NoopLoss.forward)
show_doc(PixelShuffle_ICNR.forward)
show_doc(WassersteinLoss.forward)
show_doc(MergeLayer.forward)
show_doc(SigmoidRange.forward)
show_doc(MergeLayer.forward)
show_doc(SelfAttention.forward)
show_doc(SequentialEx.forward)
show_doc(SequentialEx.append)
show_doc(SequentialEx.extend)
show_doc(SequentialEx.insert)
show_doc(PartialLayer.forward)
# ## New Methods - Please document or move to the undocumented section
