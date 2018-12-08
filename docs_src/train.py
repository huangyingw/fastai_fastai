
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
from fastai import *


# ## [`Learner`](/basic_train.html#Learner) extension methods

# These methods are automatically added to all [`Learner`](/basic_train.html#Learner) objects created after importing this module. They provide convenient access to a number of callbacks, without requiring them to be manually created.

show_doc(fit_one_cycle)


# Fit a model with 1cycle training. See [`OneCycleScheduler`](/callbacks.one_cycle.html#OneCycleScheduler) for details.

show_doc(lr_find)


# See [`LRFinder`](/callbacks.lr_finder.html#LRFinder) for details.

show_doc(to_fp16)


# See [`MixedPrecision`](/callbacks.fp16.html#MixedPrecision) for details.

show_doc(mixup)


# See [`MixUpCallback`](/callbacks.mixup.html#MixUpCallback) for more details.

# A last extension method comes from the module tta.

show_doc(Learner.TTA, full_name='TTA')


# Applies Test Time Augmentation to `learn` on the dataset `ds_type`. We take the average of our regular predictions (with a weight `beta`) with the average of predictions obtained thourh augmented versions of the training set (with a weight `1-beta`). The transforms decided for the training set are applied with a few changes `scale` controls the scale for zoom (which isn't random), the cropping isn't random but we make sure to get the four corners of the image. Flipping isn't random but applied once on each of those corner images (so that makes 8 augmented versions total).

# We'll show examples below using our MNIST sample.

path = untar_data(URLs.MNIST_SAMPLE)
data = ImageDataBunch.from_folder(path)


show_doc(ShowGraph)


# ```python
# learn = create_cnn(data, models.resnet18, metrics=accuracy, callback_fns=ShowGraph)
# learn.fit(3)
# ```

# ![Training graph](imgs/train_graph.gif)

show_doc(ShowGraph.on_epoch_end, doc_string=False)


# If we have `last_metrics`, plot them in `self.pbar`. Set the size of the graph with `n_epochs`.

show_doc(GradientClipping)


# Clips gradient at a maximum absolute value of `clip` during training. For instance:

learn = create_cnn(data, models.resnet18, metrics=accuracy,
    callback_fns=partial(GradientClipping, clip=0.1))
learn.fit(1)


show_doc(GradientClipping.on_backward_end, doc_string=False)


# Clip the gradients after they are computed but before the optimizer step.

show_doc(BnFreeze)


# For batchnorm layers where `requires_grad==False`, you generally don't want to update their moving average statistics, in order to avoid the model's statistics getting out of sync with its pre-trained weights. You can add this callback to automate this freezing of statistics (internally, it calls `eval` on these layers).

learn = create_cnn(data, models.resnet18, metrics=accuracy, callback_fns=BnFreeze)
learn.fit(1)


show_doc(BnFreeze.on_epoch_begin, doc_string=False)


# Set back the batchnorm layers on `eval` mode after the model has been set to [`train`](/train.html#train).

# ## Undocumented Methods - Methods moved below this line will intentionally be hidden

show_doc(one_cycle_scheduler)
