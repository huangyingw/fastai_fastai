# coding: utf-8
# # Learning Rate Finder
get_ipython().run_line_magic('matplotlib', 'inline')
from fastai.gen_doc.nbdoc import *
from fastai.vision import *
from fastai.callbacks import *
# Learning rate finder plots lr vs loss relationship for a [`Learner`](/basic_train.html#Learner). The idea is to reduce the amount of guesswork on picking a good starting learning rate.
#
# **Overview:**
# 1. First run lr_find `learn.lr_find()`
# 2. Plot the learning rate vs loss `learn.recorder.plot()`
# 3. Pick a learning rate before it diverges then start training
#
# **Technical Details:** (first [described]('https://arxiv.org/abs/1506.01186') by Leslie Smith)
# >Train [`Learner`](/basic_train.html#Learner) over a few iterations. Start with a very low `start_lr` and change it at each mini-batch until it reaches a very high `end_lr`. [`Recorder`](/basic_train.html#Recorder) will record the loss at each iteration. Plot those losses against the learning rate to find the optimal value before it diverges.
# ## Choosing a good learning rate
# For a more intuitive explanation, please check out [Sylvain Gugger's post](https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html)
path = untar_data(URLs.MNIST_SAMPLE)
data = ImageDataBunch.from_folder(path)
def simple_learner(): return Learner(data, simple_cnn((3, 16, 16, 2)), metrics=[accuracy])
learn = simple_learner()
# First we run this command to launch the search:
show_doc(Learner.lr_find)
learn.lr_find(stop_div=False, num_it=200)
# Then we plot the loss versus the learning rates. We're interested in finding a good order of magnitude of learning rate, so we plot with a log scale.
learn.recorder.plot()
# Then, we choose a value that is approximately in the middle of the sharpest downward slope. In this case, training with 3e-2 looks like it should work well:
simple_learner().fit(2, 1e-2)
# Don't just pick the minimum value from the plot!
learn = simple_learner()
simple_learner().fit(2, 1e-0)
# Picking a value before the downward slope results in slow training:
learn = simple_learner()
simple_learner().fit(2, 1e-3)
show_doc(LRFinder)
# ### Callback methods
# You don't call these yourself - they're called by fastai's [`Callback`](/callback.html#Callback) system automatically to enable the class's functionality.
show_doc(LRFinder.on_train_begin)
show_doc(LRFinder.on_batch_end)
show_doc(LRFinder.on_epoch_end)
show_doc(LRFinder.on_train_end)
