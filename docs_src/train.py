# coding: utf-8
# # Additional training functions
# [`train`](/train.html#train) provides a number of extension methods that are added to [`Learner`](/basic_train.html#Learner) (see below for a list and details), along with three simple callbacks:
#
# - [`ShowGraph`](/train.html#ShowGraph)
# - [`GradientClipping`](/train.html#GradientClipping)
# - [`BnFreeze`](/train.html#BnFreeze)
from fastai.gen_doc.nbdoc import *
from fastai.train import *
from fastai.vision import *
# ## [`Learner`](/basic_train.html#Learner) extension methods
# These methods are automatically added to all [`Learner`](/basic_train.html#Learner) objects created after importing this module. They provide convenient access to a number of callbacks, without requiring them to be manually created.
show_doc(fit_one_cycle)
show_doc(one_cycle_scheduler)
# See [`OneCycleScheduler`](/callbacks.one_cycle.html#OneCycleScheduler) for details.
show_doc(lr_find)
# See [`LRFinder`](/callbacks.lr_finder.html#LRFinder) for details.
show_doc(to_fp16)
# See [`MixedPrecision`](/callbacks.fp16.html#MixedPrecision) for details.
show_doc(to_fp32)
show_doc(mixup)
# See [`MixUpCallback`](/callbacks.mixup.html#MixUpCallback) for more details.
# ## Additional callbacks
# We'll show examples below using our MNIST sample. As usual the `on_something` methods are directly called by the fastai library, no need to call them yourself.
path = untar_data(URLs.MNIST_SAMPLE)
data = ImageDataBunch.from_folder(path)
show_doc(ShowGraph, title_level=3)
# ```python
# learn = create_cnn(data, models.resnet18, metrics=accuracy, callback_fns=ShowGraph)
# learn.fit(3)
# ```
# ![Training graph](imgs/train_graph.gif)
show_doc(ShowGraph.on_epoch_end)
show_doc(GradientClipping)
learn = create_cnn(data, models.resnet18, metrics=accuracy,
    callback_fns=partial(GradientClipping, clip=0.1))
learn.fit(1)
show_doc(GradientClipping.on_backward_end)
show_doc(BnFreeze)
# For batchnorm layers where `requires_grad==False`, you generally don't want to update their moving average statistics, in order to avoid the model's statistics getting out of sync with its pre-trained weights. You can add this callback to automate this freezing of statistics (internally, it calls `eval` on these layers).
learn = create_cnn(data, models.resnet18, metrics=accuracy, callback_fns=BnFreeze)
learn.fit(1)
show_doc(BnFreeze.on_epoch_begin)
# ## Undocumented Methods - Methods moved below this line will intentionally be hidden
