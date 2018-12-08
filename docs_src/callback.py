
# coding: utf-8

# # Classes for callback implementors

from fastai.gen_doc.nbdoc import *
from fastai.callback import *
from fastai import *


# fastai provides a powerful *callback* system, which is documented on the [`callbacks`](/callbacks.html#callbacks) page; look on that page if you're just looking for how to use existing callbacks. If you want to create your own, you'll need to use the classes discussed below.
#
# A key motivation for the callback system is that additional functionality can be entirely implemented in a single callback, so that it's easily read. By using this trick, we will have different methods categorized in different callbacks where we will find clearly stated all the interventions the method makes in training. For instance in the [`LRFinder`](/callbacks.lr_finder.html#LRFinder) callback, on top of running the fit function with exponentially growing LRs, it needs to handle some preparation and clean-up, and all this code can be in the same callback so we know exactly what it is doing and where to look if we need to change something.
#
# In addition, it allows our [`fit`](/basic_train.html#fit) function to be very clean and simple, yet still easily extended. So far in implementing a number of recent papers, we haven't yet come across any situation where we had to modify our training loop source code - we've been able to use callbacks every time.

show_doc(Callback)


# To create a new type of callback, you'll need to inherit from this class, and implement one or more methods as required for your purposes. Perhaps the easiest way to get started is to look at the source code for some of the pre-defined fastai callbacks. You might be surprised at how simple they are! For instance, here is the **entire** source code for [`GradientClipping`](/train.html#GradientClipping):
#
# ```python
# @dataclass
# class GradientClipping(LearnerCallback):
#     clip:float
#     def on_backward_end(self, **kwargs):
#         if self.clip:
#             nn.utils.clip_grad_norm_(self.learn.model.parameters(), self.clip)
# ```

# You generally want your custom callback constructor to take a [`Learner`](/basic_train.html#Learner) parameter, e.g.:
#
# ```python
# @dataclass
# class MyCallback(Callback):
#     learn:Learner
# ```
#
# Note that this allows the callback user to just pass your callback name to `callback_fns` when constructing their [`Learner`](/basic_train.html#Learner), since that always passes `self` when constructing callbacks from `callback_fns`. In addition, by passing the learner, this callback will have access to everything: e.g all the inputs/outputs as they are calculated, the losses, and also the data loaders, the optimizer, etc. At any time:
# - Changing self.learn.data.train_dl or self.data.valid_dl will change them inside the fit function (we just need to pass the [`DataBunch`](/basic_data.html#DataBunch) object to the fit function and not data.train_dl/data.valid_dl)
# - Changing self.learn.opt.opt (We have an [`OptimWrapper`](/callback.html#OptimWrapper) on top of the actual optimizer) will change it inside the fit function.
# - Changing self.learn.data or self.learn.opt directly WILL NOT change the data or the optimizer inside the fit function.

# In any of the callbacks you can unpack in the kwargs:
# - `n_epochs`, contains the number of epochs the training will take in total
# - `epoch`, contains the number of the current
# - `iteration`, contains the number of iterations done since the beginning of training
# - `num_batch`, contains the number of the batch we're at in the dataloader
# - `last_input`, contains the last input that got through the model (eventually updated by a callback)
# - `last_target`, contains the last target that gor through the model (eventually updated by a callback)
# - `last_output`, contains the last output spitted by the model (eventually updated by a callback)
# - `last_loss`, contains the last loss computed (eventually updated by a callback)
# - `smooth_loss`, contains the smoothed version of the loss
# - `last_metrics`, contains the last validation loss and emtrics computed
# - `pbar`, the progress bar

# ### Methods your subclass can implement

# All of these methods are optional; your subclass can handle as many or as few as you require.

show_doc(Callback.on_train_begin)


# Here we can initiliaze anything we need.
# The optimizer has now been initialized. We can change any hyper-parameters by typing, for instance:
#
# ```
# self.opt.lr = new_lr
# self.opt.mom = new_mom
# self.opt.wd = new_wd
# self.opt.beta = new_beta
# ```

show_doc(Callback.on_epoch_begin)


# This is not technically required since we have `on_train_begin` for epoch 0 and `on_epoch_end` for all the other epochs,
# yet it makes writing code that needs to be done at the beginning of every epoch easy and more readable.

show_doc(Callback.on_batch_begin)


# Here is the perfect place to prepare everything before the model is called.
# Example: change the values of the hyperparameters (if we don't do it on_batch_end instead)
#
# If we return something, that will be the new value for `xb`,`yb`.

show_doc(Callback.on_loss_begin)


# Here is the place to run some code that needs to be executed after the output has been computed but before the
# loss computation.
# Example: putting the output back in FP32 when training in mixed precision.
#
# If we return something, that will be the new value for the output.

show_doc(Callback.on_backward_begin)


# Here is the place to run some code that needs to be executed after the loss has been computed but before the gradient computation.
# Example: `reg_fn` in RNNs.
#
# If we return something, that will be the new value for loss. Since the recorder is always called first,
# it will have the raw loss.

show_doc(Callback.on_backward_end)


# Here is the place to run some code that needs to be executed after the gradients have been computed but
# before the optimizer is called.

show_doc(Callback.on_step_end)


# Here is the place to run some code that needs to be executed after the optimizer step but before the gradients
# are zeroed

show_doc(Callback.on_batch_end)


# Here is the place to run some code that needs to be executed after a batch is fully done.
# Example: change the values of the hyperparameters (if we don't do it on_batch_begin instead)
#
# If we return true, the current epoch is interrupted (example: lr_finder stops the training when the loss explodes)

show_doc(Callback.on_epoch_end)


# Here is the place to run some code that needs to be executed at the end of an epoch.
# Example: Save the model if we have a new best validation loss/metric.
#
# If we return true, the training stops (example: early stopping)

show_doc(Callback.on_train_end)


# Here is the place to tidy everything. It's always executed even if there was an error during the training loop,
# and has an extra kwarg named exception to check if there was an exception or not.
# Examples: save log_files, load best model found during training

# ## Annealing functions

# The following functions provide different annealing schedules. You probably won't need to call them directly, but would instead use them as part of a callback. Here's what each one looks like:

annealings = "NO LINEAR COS EXP POLY".split()
fns = [annealing_no, annealing_linear, annealing_cos, annealing_exp, annealing_poly(0.8)]
for fn, t in zip(fns, annealings):
    plt.plot(np.arange(0, 100), [fn(2, 1e-2, o)
        for o in np.linspace(0.01, 1, 100)], label=t)
plt.legend();


show_doc(annealing_cos)


show_doc(annealing_exp)


show_doc(annealing_linear)


show_doc(annealing_no)


show_doc(annealing_poly)


show_doc(CallbackHandler)


# You probably won't need to use this class yourself. It's used by fastai to combine all the callbacks together and call any relevant callback functions for each training stage. The methods below simply call the equivalent method in each callback function in [`self.callbacks`](/callbacks.html#callbacks).

show_doc(CallbackHandler.on_backward_begin)


show_doc(CallbackHandler.on_backward_end)


show_doc(CallbackHandler.on_batch_begin)


show_doc(CallbackHandler.on_batch_end)


show_doc(CallbackHandler.on_epoch_begin)


show_doc(CallbackHandler.on_epoch_end)


show_doc(CallbackHandler.on_loss_begin)


show_doc(CallbackHandler.on_step_end)


show_doc(CallbackHandler.on_train_begin)


show_doc(CallbackHandler.on_train_end)


show_doc(OptimWrapper)


# This is a convenience class that provides a consistent API for getting and setting optimizer hyperparameters. For instance, for [`optim.Adam`](https://pytorch.org/docs/stable/optim.html#torch.optim.Adam) the momentum parameter is actually `betas[0]`, whereas for [`optim.SGD`](https://pytorch.org/docs/stable/optim.html#torch.optim.SGD) it's simply `momentum`. As another example, the details of handling weight decay depend on whether you are using `true_wd` or the traditional L2 regularization approach.
#
# This class also handles setting different WD and LR for each layer group, for discriminative layer training.

show_doc(OptimWrapper.create)


show_doc(OptimWrapper.new)


show_doc(OptimWrapper.read_defaults)


show_doc(OptimWrapper.read_val)


show_doc(OptimWrapper.set_val)


show_doc(OptimWrapper.step)


show_doc(OptimWrapper.zero_grad)


show_doc(SmoothenValue)


# Used for smoothing loss in [`Recorder`](/basic_train.html#Recorder).

show_doc(SmoothenValue.add_value)


show_doc(Stepper)


# Used for creating annealing schedules, mainly for [`OneCycleScheduler`](/callbacks.one_cycle.html#OneCycleScheduler).

show_doc(Stepper.step)


show_doc(AverageMetric)


# See the documentation on [`metrics`](/metrics.html#metrics) for more information.

# ## Undocumented Methods - Methods moved below this line will intentionally be hidden

show_doc(AverageMetric.on_epoch_begin)


show_doc(AverageMetric.on_batch_end)


show_doc(AverageMetric.on_epoch_end)


# ## New Methods - Please document or move to the undocumented section
