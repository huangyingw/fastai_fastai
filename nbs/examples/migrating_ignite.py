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

# # Tutorial - Migrating from Ignite
#
# > Incrementally adding fastai goodness to your Ignite training

# We're going to use the MNIST training code from Ignite's examples directory (as at August 2020), converted to a module.

# +
from migrating_ignite import *

from fastai.vision.all import *
# -

# To use it in fastai, we first pull the DataLoaders from the module into a `DataLoaders` object:

data = DataLoaders(*get_data_loaders(64, 128)).cuda()

# We can now create a `Learner` and fit:

opt_func = partial(SGD, momentum=0.5)
learn = Learner(data, Net(), loss_func=nn.NLLLoss(), opt_func=opt_func, metrics=accuracy)
learn.fit_one_cycle(1, 0.01)

# As you can see, migrating from Ignite allowed us to replace 52 lines of code (in `run()`) with just 3 lines, and doesn't require you to change any of your existing data pipelines, optimizers, loss functions, models, etc. Once you've made this change, you can then benefit from fastai's rich set of callbacks, transforms, visualizations, and so forth.
#
# Note that fastai is very different from Ignite, in that it is much more than just a training loop (although we're only using the training loop in this example) - it is a complete framework including GPU-accelerated transformations, end-to-end inference, integrated applications for vision, text, tabular, and collaborative filtering, and so forth. You can use any part of the framework on its own, or combine them together, as described in the [fastai paper](https://arxiv.org/abs/2002.04688).
