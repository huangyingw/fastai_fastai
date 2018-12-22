# coding: utf-8
# # List of callbacks
from fastai.gen_doc.nbdoc import *
from fastai.callbacks import *
from fastai.basic_train import *
from fastai.train import *
from fastai import callbacks
# fastai's training loop is highly extensible, with a rich *callback* system. See the [`callback`](/callback.html#callback) docs if you're interested in writing your own callback. See below for a list of callbacks that are provided with fastai, grouped by the module they're defined in.
#
# Every callback that is passed to [`Learner`](/basic_train.html#Learner) with the `callback_fns` parameter will be automatically stored as an attribute. The attribute name is snake-cased, so for instance [`ActivationStats`](/callbacks.hooks.html#ActivationStats) will appear as `learn.activation_stats` (assuming your object is named `learn`).
# ## [`Callback`](/callback.html#Callback)
#
# This sub-package contains more sophisticated callbacks that each are in their own module. They are (click the link for more details):
#
# ### [`CSVLogger`](/callbacks.csv_logger.html#CSVLogger)
#
# Log the results of training in a csv file.
#
# ### [`OneCycleScheduler`](/callbacks.one_cycle.html#OneCycleScheduler)
#
# Train with Leslie Smith's [1cycle annealing](https://sgugger.github.io/the-1cycle-policy.html) method.
#
# ### [`MixedPrecision`](/callbacks.fp16.html#MixedPrecision)
#
# Use fp16 to [take advantage of tensor cores](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html) on recent NVIDIA GPUs for a 200% or more speedup.
#
# ### [`GeneralScheduler`](/callbacks.general_sched.html#GeneralScheduler)
#
# Create your own multi-stage annealing schemes with a convenient API.
#
# ### [`MixUpCallback`](/callbacks.mixup.html#MixUpCallback)
#
# Data augmentation using the method from [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)
#
# ### [`LRFinder`](/callbacks.lr_finder.html#LRFinder)
#
# Use Leslie Smith's [learning rate finder](https://www.jeremyjordan.me/nn-learning-rate/) to find a good learning rate for training your model.
#
# ### [`HookCallback`](/callbacks.hooks.html#HookCallback)
#
# Convenient wrapper for registering and automatically deregistering [PyTorch hooks](https://pytorch.org/tutorials/beginner/former_torchies/nn_tutorial.html#forward-and-backward-function-hooks). Also contains pre-defined hook callback: [`ActivationStats`](/callbacks.hooks.html#ActivationStats).
#
# ### [`RNNTrainer`](/callbacks.rnn.html#RNNTrainer)
#
# Callback taking care of all the tweaks to train an RNN.
#
# ### [`TerminateOnNaNCallback`](/callbacks.tracker.html#TerminateOnNaNCallback)
#
# Stop training if the loss reaches NaN.
#
# ### [`EarlyStoppingCallback`](/callbacks.tracker.html#EarlyStoppingCallback)
#
# Stop training if a given metric/validation loss doesn't improve.
#
# ### [`SaveModelCallback`](/callbacks.tracker.html#SaveModelCallback)
#
# Save the model at every epoch, or the best model for a given metric/validation loss.
#
# ### [`ReduceLROnPlateauCallback`](/callbacks.tracker.html#ReduceLROnPlateauCallback)
#
# Reduce the learning rate each time a given metric/validation loss doesn't improve by a certain factor.
# ## [`train`](/train.html#train) and [`basic_train`](/basic_train.html#basic_train)
#
# ### [`Recorder`](/basic_train.html#Recorder)
#
# Track per-batch and per-epoch smoothed losses and metrics.
#
# ### [`ShowGraph`](/train.html#ShowGraph)
#
# Dynamically display a learning chart during training.
#
# ### [`BnFreeze`](/train.html#BnFreeze)
#
# Freeze batchnorm layer moving average statistics for non-trainable layers.
#
# ### [`GradientClipping`](/train.html#GradientClipping)
#
# Clips gradient during training.
#
