# coding: utf-8
# # Tracking Callbacks
from fastai.gen_doc.nbdoc import *
from fastai.vision import *
from fastai.callbacks import *
# This module regroups the callbacks that track one of the metrics computed at the end of each epoch to take some decision about training. To show examples of use, we'll use our sample of MNIST and a simple cnn model.
path = untar_data(URLs.MNIST_SAMPLE)
data = ImageDataBunch.from_folder(path)
show_doc(TerminateOnNaNCallback)
# Sometimes, training diverges and the loss goes to nan. In that case, there's no point continuing, so this callback stops the training.
model = simple_cnn((3, 16, 16, 2))
learn = Learner(data, model, metrics=[accuracy])
learn.fit_one_cycle(1, 1e4)
# Using it prevents that situation to happen.
model = simple_cnn((3, 16, 16, 2))
learn = Learner(data, model, metrics=[accuracy], callbacks=[TerminateOnNaNCallback()])
learn.fit(2, 1e4)
# ### Callback methods
# You don't call these yourself - they're called by fastai's [`Callback`](/callback.html#Callback) system automatically to enable the class's functionality.
show_doc(TerminateOnNaNCallback.on_batch_end)
show_doc(TerminateOnNaNCallback.on_epoch_end)
show_doc(EarlyStoppingCallback)
# This callback tracks the quantity in `monitor` during the training of `learn`. `mode` can be forced to 'min' or 'max' but will automatically try to determine if the quantity should be the lowest possible (validation loss) or the highest possible (accuracy). Will stop training after `patience` epochs if the quantity hasn't improved by `min_delta`.
model = simple_cnn((3, 16, 16, 2))
learn = Learner(data, model, metrics=[accuracy],
                callback_fns=[partial(EarlyStoppingCallback, monitor='accuracy', min_delta=0.01, patience=3)])
learn.fit(50, 1e-42)
# ### Callback methods
# You don't call these yourself - they're called by fastai's [`Callback`](/callback.html#Callback) system automatically to enable the class's functionality.
show_doc(EarlyStoppingCallback.on_train_begin)
show_doc(EarlyStoppingCallback.on_epoch_end)
show_doc(SaveModelCallback)
# This callback tracks the quantity in `monitor` during the training of `learn`. `mode` can be forced to 'min' or 'max' but will automatically try to determine if the quantity should be the lowest possible (validation loss) or the highest possible (accuracy). Will save the model in `name` whenever determined by `every` ('improvement' or 'epoch'). Loads the best model at the end of training is `every='improvement'`.
# ### Callback methods
# You don't call these yourself - they're called by fastai's [`Callback`](/callback.html#Callback) system automatically to enable the class's functionality.
show_doc(SaveModelCallback.on_epoch_end)
show_doc(SaveModelCallback.on_train_end)
show_doc(ReduceLROnPlateauCallback)
# This callback tracks the quantity in `monitor` during the training of `learn`. `mode` can be forced to 'min' or 'max' but will automatically try to determine if the quantity should be the lowest possible (validation loss) or the highest possible (accuracy). Will reduce the learning rate by `factor` after `patience` epochs if the quantity hasn't improved by `min_delta`.
# ### Callback methods
# You don't call these yourself - they're called by fastai's [`Callback`](/callback.html#Callback) system automatically to enable the class's functionality.
show_doc(ReduceLROnPlateauCallback.on_train_begin)
show_doc(ReduceLROnPlateauCallback.on_epoch_end)
show_doc(TrackerCallback)
show_doc(TrackerCallback.get_monitor_value)
# ### Callback methods
# You don't call these yourself - they're called by fastai's [`Callback`](/callback.html#Callback) system automatically to enable the class's functionality.
show_doc(TrackerCallback.on_train_begin)
# ## Undocumented Methods - Methods moved below this line will intentionally be hidden
# ## New Methods - Please document or move to the undocumented section
