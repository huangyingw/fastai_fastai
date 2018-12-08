
# coding: utf-8

# # Training metrics

# *Metrics* for training fastai models are simply functions that take `input` and `target` tensors, and return some metric of interest for training. You can write your own metrics by defining a function of that type, and passing it to [`Learner`](/basic_train.html#Learner) in the [code]metrics[/code] parameter, or use one of the following pre-defined functions.

from fastai.gen_doc.nbdoc import *
from fastai import *


# ## Predefined metrics:

show_doc(accuracy)


show_doc(accuracy_thresh, doc_string=False)


# Compute accuracy when `y_pred` and `y_true` for multi-label models, based on comparing predictions to `thresh`, `sigmoid` will be applied to `y_pred` if the corresponding flag is True.

show_doc(dice)


show_doc(error_rate)


show_doc(fbeta)


# See the [F1 score wikipedia page](https://en.wikipedia.org/wiki/F1_score) for details.

show_doc(exp_rmspe)


show_doc(Fbeta_binary, title_level=3)


# ## Creating your own metric

# Creating a new metric can be as simple as creating a new function. If you metric is an average over the total number of elements in your dataset, just write the function that will compute it on a batch (taking `pred` and `targ` as arguments). It will then be automatically averaged over the batches (taking their different sizes into acount).
#
# Sometimes metrics aren't simple averages however. If we take the example of precision for instance, we have to divide the number of true positives by the number of predictions we made for that class. This isn't an average over the number of elements we have in the dataset, we only consider those where we made a positive prediction for a specific thing. Computing the precision for each batch, then averaging them will yield to a result that may be close to the real value, but won't be it exactly (and it really depends on how you deal with special case of 0 positive predicitions).
#
# This why in fastai, every metric is implemented as a callback. If you pass a regular function, the library transforms it to a proper callback called `AverageCallback`. The callback metrics are only called during the validation phase, and only for the following events:
# - <code>on_epoch_begin</code> (for initialization)
# - <code>on_batch_begin</code> (if we need to have a look at the input/target and maybe modify them)
# - <code>on_batch_end</code> (to analyze the last results and update our computation)
# - <code>on_epoch_end</code>(to wrap up the final result that should be stored in `.metric`)
#
# As an example, is here the exact implementation of the [`AverageMetric`](/callback.html#AverageMetric) callback that transforms a function like [`accuracy`](/metrics.html#accuracy) into a metric callback.

class AverageMetric(Callback):
    def __init__(self, func):
        self.func, self.name = func, func.__name__

    def on_epoch_begin(self, **kwargs):
        self.val, self.count = 0., 0

    def on_batch_end(self, last_output, last_target, train, **kwargs):
        self.count += last_target.size(0)
        self.val += last_target.size(0) * self.func(last_output, last_target).detach().item()

    def on_epoch_end(self, **kwargs):
        self.metric = self.val / self.count


# And here is another example that properly computes the precision for a given class.

class Precision(Callback):
    
    def on_epoch_begin(self, **kwargs):
        self.correct, self.total = 0, 0
    
    def on_batch_end(self, last_output, last_target, **kwargs):
        preds = last_output.argmax(1)
        self.correct += ((preds == 0) * (last_target == 0)).float().sum()
        self.total += (preds == 0).float().sum()
    
    def on_epoch_end(self, **kwargs):
        self.metric = self.correct / self.total


# ## Undocumented Methods - Methods moved below this line will intentionally be hidden

show_doc(Fbeta_binary.on_batch_end)


show_doc(Fbeta_binary.on_epoch_begin)


show_doc(Fbeta_binary.on_epoch_end)


# ## New Methods - Please document or move to the undocumented section
