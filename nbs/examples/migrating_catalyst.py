# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     split_at_heading: true
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# all_slow
# -

from fastai.vision.all import *

# # Tutorial - Migrating from Catalyst
#
# > Incrementally adding fastai goodness to your Catalyst training

# ## Catalyst code

# We're going to use the MNIST training code from Catalyst's README (as at August 2020), converted to a module.

from migrating_catalyst import *

# To use it in fastai, we first convert the Catalyst dict into a `DataLoaders` object:

data = DataLoaders(loaders['train'], loaders['valid']).cuda()


# ### Using callbacks

# In the Catalyst code, a training loop is defined manually, which is where the input tensor is flattened. In fastai, there's no need to define your own training loop - you can insert your own code into any part of the training process by using a callback, which can even modify data, gradients, the loss function, or anything else in the training loop:

@before_batch_cb
def cb(self, xb, yb): return (xb[0].view(xb[0].size(0), -1),), yb


# The Catalyst example also modifies the training loop to add metrics, but you can pass these directly to your `Learner` in fastai:

metrics = [accuracy, top_k_accuracy]
learn = Learner(data, model, loss_func=F.cross_entropy, opt_func=Adam,
                metrics=metrics, cbs=cb)

# You can now fit your model. fastai supports many schedulers. We recommend using 1cycle:

learn.fit_one_cycle(1, 0.02)

# As you can see, migrating from Catalyst allowed us to replace 17 lines of code (in `CustomRunner`) with just 3 lines, and doesn't require you to change any of your existing data pipelines, optimizers, loss functions, models, etc. Once you've made this change, you can then benefit from fastai's rich set of callbacks, transforms, visualizations, and so forth.
#
# Note that fastai is very different from Catalyst, in that it is much more than just a training loop (although we're only using the training loop in this example) - it is a complete framework including GPU-accelerated transformations, end-to-end inference, integrated applications for vision, text, tabular, and collaborative filtering, and so forth. You can use any part of the framework on its own, or combine them together, as described in the [fastai paper](https://arxiv.org/abs/2002.04688).

# ### Changing the model

# Instead of using callbacks, in this case you can also simply change the model. Here we pull the `view()` out of the training loop, and into the model, using fastai's `Flatten` layer:

model = nn.Sequential(
    Flatten(),
    torch.nn.Linear(28 * 28, 10))

# We can now create a `Learner` and train without using any callbacks:

learn = Learner(data, model, loss_func=F.cross_entropy, opt_func=Adam, metrics=metrics)
learn.fit_one_cycle(1, 0.02)
