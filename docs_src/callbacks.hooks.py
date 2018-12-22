# coding: utf-8
# # Hook callbacks
# This provides both a standalone class and a callback for registering and automatically deregistering [PyTorch hooks](https://pytorch.org/tutorials/beginner/former_torchies/nn_tutorial.html#forward-and-backward-function-hooks), along with some pre-defined hooks. Hooks can be attached to any [`nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module), for either the forward or the backward pass.
#
# We'll start by looking at the pre-defined hook [`ActivationStats`](/callbacks.hooks.html#ActivationStats), then we'll see how to create our own.
from fastai.gen_doc.nbdoc import *
from fastai.callbacks.hooks import *
from fastai.train import *
from fastai.vision import *
show_doc(ActivationStats)
# [`ActivationStats`](/callbacks.hooks.html#ActivationStats) saves the layer activations in `self.stats` for all `modules` passed to it. By default it will save activations for *all* modules. For instance:
path = untar_data(URLs.MNIST_SAMPLE)
data = ImageDataBunch.from_folder(path)
#learn = create_cnn(data, models.resnet18, callback_fns=ActivationStats)
learn = Learner(data, simple_cnn((3, 16, 16, 2)), callback_fns=ActivationStats)
learn.fit(1)
# The saved `stats` is a `FloatTensor` of shape `(2,num_modules,num_batches)`. The first axis is `(mean,stdev)`.
len(learn.data.train_dl), len(learn.activation_stats.modules)
learn.activation_stats.stats.shape
# So this shows the standard deviation (`axis0==1`) of 2th last layer (`axis1==-2`) for each batch (`axis2`):
plt.plot(learn.activation_stats.stats[1][-2].numpy());
# ### Internal implementation
show_doc(ActivationStats.hook)
# ### Callback methods
# You don't call these yourself - they're called by fastai's [`Callback`](/callback.html#Callback) system automatically to enable the class's functionality.
show_doc(ActivationStats.on_train_begin)
show_doc(ActivationStats.on_batch_end)
show_doc(ActivationStats.on_train_end)
show_doc(Hook)
# Registers and manually deregisters a [PyTorch hook](https://pytorch.org/tutorials/beginner/former_torchies/nn_tutorial.html#forward-and-backward-function-hooks). Your `hook_func` will be called automatically when forward/backward (depending on `is_forward`) for your module `m` is run, and the result of that function is placed in `self.stored`.
show_doc(Hook.remove)
# Deregister the hook, if not called already.
show_doc(Hooks)
# Acts as a `Collection` (i.e. `len(hooks)` and `hooks[i]`) and an `Iterator` (i.e. `for hook in hooks`) of a group of hooks, one for each module in `ms`, with the ability to remove all as a group. Use `stored` to get all hook results. `hook_func` and `is_forward` behavior is the same as [`Hook`](/callbacks.hooks.html#Hook). See the source code for [`HookCallback`](/callbacks.hooks.html#HookCallback) for a simple example.
show_doc(Hooks.remove)
# Deregister all hooks created by this class, if not previously called.
# ## Convenience functions for hooks
show_doc(hook_output)
# Function that creates a [`Hook`](/callbacks.hooks.html#Hook) for `module` that simply stores the output of the layer.
show_doc(hook_outputs)
# Function that creates a [`Hook`](/callbacks.hooks.html#Hook) for all passed `modules` that simply stores the output of the layers. For example, the (slightly simplified) source code of [`model_sizes`](/callbacks.hooks.html#model_sizes) is:
#
# ```python
# def model_sizes(m, size):
#     x = m(torch.zeros(1, in_channels(m), *size))
#     return [o.stored.shape for o in hook_outputs(m)]
# ```
show_doc(model_sizes)
show_doc(model_summary)
show_doc(num_features_model)
# It can be useful to get the size of each layer of a model (e.g. for printing a summary, or for generating cross-connections for a [`DynamicUnet`](/vision.models.unet.html#DynamicUnet)), however they depend on the size of the input. This function calculates the layer sizes by passing in a minimal tensor of `size`.
show_doc(dummy_batch)
show_doc(dummy_eval)
show_doc(HookCallback)
# For all `modules`, uses a callback to automatically register a method `self.hook` (that you must define in an inherited class) as a hook. This method must have the signature:
#
# ```python
# def hook(self, m:Model, input:Tensors, output:Tensors)
# ```
#
# If `do_remove` then the hook is automatically deregistered at the end of training. See [`ActivationStats`](/callbacks.hooks.html#ActivationStats) for a simple example of inheriting from this class.
# ### Callback methods
# You don't call these yourself - they're called by fastai's [`Callback`](/callback.html#Callback) system automatically to enable the class's functionality.
show_doc(HookCallback.on_train_begin)
show_doc(HookCallback.on_train_end)
# ## Undocumented Methods - Methods moved below this line will intentionally be hidden
show_doc(HookCallback.remove)
show_doc(Hook.hook_fn)
# ## New Methods - Please document or move to the undocumented section
