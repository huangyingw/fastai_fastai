# coding: utf-8
# # Basic training functionality
from fastai.basic_train import *
from fastai.gen_doc.nbdoc import *
from fastai.vision import *
from fastai.distributed import *
# [`basic_train`](/basic_train.html#basic_train) wraps together the data (in a [`DataBunch`](/basic_data.html#DataBunch) object) with a pytorch model to define a [`Learner`](/basic_train.html#Learner) object. This is where the basic training loop is defined for the [`fit`](/basic_train.html#fit) function. The [`Learner`](/basic_train.html#Learner) object is the entry point of most of the [`Callback`](/callback.html#Callback) functions that will customize this training loop in different ways (and made available through the [`train`](/train.html#train) module), notably:
#
#  - [`Learner.lr_find`](/train.html#lr_find) will launch an LR range test that will help you select a good learning rate
#  - [`Learner.fit_one_cycle`](/train.html#fit_one_cycle) will launch a training using the 1cycle policy, to help you train your model fast.
#  - [`Learner.to_fp16`](/train.html#to_fp16) will convert your model in half precision and help you launch a training in mixed precision.
show_doc(Learner, title_level=2)
# The main purpose of [`Learner`](/basic_train.html#Learner) is to train `model` using [`Learner.fit`](/basic_train.html#Learner.fit). After every epoch, all *metrics* will be printed, and will also be available to callbacks.
#
# The default weight decay will be `wd`, which will be handled using the method from [Fixing Weight Decay Regularization in Adam](https://arxiv.org/abs/1711.05101) if `true_wd` is set (otherwise it's L2 regularization). If `bn_wd` is False then weight decay will be removed from batchnorm layers, as recommended in [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677). You can ensure that batchnorm layer learnable params are trained even for frozen layer groups, by enabling `train_bn`.
#
# To use [discriminative layer training](#Discriminative-layer-training) pass an [`nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) for each layer group to be optimized with different settings.
#
# Any model files created will be saved in `path`/`model_dir`.
#
# You can pass a list of [`callbacks`](/callbacks.html#callbacks) that you have already created, or (more commonly) simply pass a list of callback functions to `callback_fns` and each function will be called (passing `self`) on object initialization, with the results stored as callback objects. For a walk-through, see the [training overview](/training.html) page. You may also want to use an `application` to fit your model, e.g. using the [`create_cnn`](/vision.learner.html#create_cnn) method:
path = untar_data(URLs.MNIST_SAMPLE)
data = ImageDataBunch.from_folder(path)
learn = create_cnn(data, models.resnet18, metrics=accuracy)
learn.fit(1)
# ### Model fitting methods
show_doc(Learner.fit)
# Uses [discriminative layer training](#Discriminative-layer-training) if multiple learning rates or weight decay values are passed. To control training behaviour, use the [`callback`](/callback.html#callback) system or one or more of the pre-defined [`callbacks`](/callbacks.html#callbacks).
show_doc(Learner.fit_one_cycle)
# Uses the [`OneCycleScheduler`](/callbacks.one_cycle.html#OneCycleScheduler) callback.
show_doc(Learner.lr_find)
# Runs the learning rate finder defined in [`LRFinder`](/callbacks.lr_finder.html#LRFinder), as discussed in [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186).
# ### See results
show_doc(Learner.get_preds)
show_doc(Learner.validate)
show_doc(Learner.show_results)
show_doc(Learner.predict)
show_doc(Learner.pred_batch)
show_doc(Learner.interpret, full_name='interpret')
jekyll_note('This function only works in the vision application.')
# ### Model summary
show_doc(Learner.summary)
# ### Test time augmentation
show_doc(Learner.TTA, full_name='TTA')
# Applies Test Time Augmentation to `learn` on the dataset `ds_type`. We take the average of our regular predictions (with a weight `beta`) with the average of predictions obtained thourh augmented versions of the training set (with a weight `1-beta`). The transforms decided for the training set are applied with a few changes `scale` controls the scale for zoom (which isn't random), the cropping isn't random but we make sure to get the four corners of the image. Flipping isn't random but applied once on each of those corner images (so that makes 8 augmented versions total).
# ### Gradient clipping
show_doc(Learner.clip_grad)
# ### Mixed precision training
show_doc(Learner.to_fp16)
# Uses the [`MixedPrecision`](/callbacks.fp16.html#MixedPrecision) callback to train in mixed precision (i.e. forward and backward passes using fp16, with weight updates using fp32), using all [NVIDIA recommendations](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html) for ensuring speed and accuracy.
show_doc(Learner.to_fp32)
# ### Discriminative layer training
# When fitting a model you can pass a list of learning rates (and/or weight decay amounts), which will apply a different rate to each *layer group* (i.e. the parameters of each module in `self.layer_groups`). See the [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146) paper for details and experimental results in NLP (we also frequently use them successfully in computer vision, but have not published a paper on this topic yet). When working with a [`Learner`](/basic_train.html#Learner) on which you've called `split`, you can set hyperparameters in four ways:
#
# 1. `param = [val1, val2 ..., valn]` (n = number of layer groups)
# 2. `param = val`
# 3. `param = slice(start,end)`
# 4. `param = slice(end)`
#
# If we chose to set it in way 1, we must specify a number of values exactly equal to the number of layer groups. If we chose to set it in way 2, the chosen value will be repeated for all layer groups. See [`Learner.lr_range`](/basic_train.html#Learner.lr_range) for an explanation of the `slice` syntax).
#
# Here's an example of how to use discriminative learning rates (note that you don't actually need to manually call [`Learner.split`](/basic_train.html#Learner.split) in this case, since fastai uses this exact function as the default split for `resnet18`; this is just to show how to customize it):
# creates 3 layer groups
learn.split(lambda m: (m[0][6], m[1]))
# only randomly initialized head now trainable
learn.freeze()
learn.fit_one_cycle(1)
# all layers now trainable
learn.unfreeze()
# optionally, separate LR and WD for each group
learn.fit_one_cycle(1, max_lr=(1e-4, 1e-3, 1e-2), wd=(1e-4, 1e-4, 1e-1))
show_doc(Learner.lr_range)
# Rather than manually setting an LR for every group, it's often easier to use [`Learner.lr_range`](/basic_train.html#Learner.lr_range). This is a convenience method that returns one learning rate for each layer group. If you pass `slice(start,end)` then the first group's learning rate is `start`, the last is `end`, and the remaining are evenly geometrically spaced.
#
# If you pass just `slice(end)` then the last group's learning rate is `end`, and all the other groups are `end/10`. For instance (for our learner that has 3 layer groups):
learn.lr_range(slice(1e-5, 1e-3)), learn.lr_range(slice(1e-3))
show_doc(Learner.unfreeze)
# Sets every layer group to *trainable* (i.e. `requires_grad=True`).
show_doc(Learner.freeze)
# Sets every layer group except the last to *untrainable* (i.e. `requires_grad=False`).
show_doc(Learner.freeze_to)
show_doc(Learner.split)
# A convenience method that sets `layer_groups` based on the result of [`split_model`](/torch_core.html#split_model). If `split_on` is a function, it calls that function and passes the result to [`split_model`](/torch_core.html#split_model) (see above for example).
# ### Saving and loading models
# Simply call [`Learner.save`](/basic_train.html#Learner.save) and [`Learner.load`](/basic_train.html#Learner.load) to save and load models. Only the parameters are saved, not the actual architecture (so you'll need to create your model in the same way before loading weights back in). Models are saved to the `path`/`model_dir` directory.
show_doc(Learner.load)
show_doc(Learner.save)
# ### Segmentation model
show_doc(unet_learner)
# Build a [`Learner`](/basic_train.html#Learner) for segmentation tasks from [`data`](/vision.data.html#vision.data), using `arch` that may be `pretrained` if that flag is `True`. `split_on` will overwrite the default way the layers are split for differential learning rates. The underlying model is a `DynamicUnet` with `blur`, `blur_final`, `self_attention`, `sigmoid` and `last_cross`. `norm_type` is passed to [`conv_layer`](/layers.html#conv_layer), the `kwargs` are passed to the [`Learner`](/basic_train.html#Learner) constructor.
# ### Other methods
show_doc(Learner.init)
# Initializes all weights (except batchnorm) using function `init`, which will often be from PyTorch's [`nn.init`](https://pytorch.org/docs/stable/nn.html#torch-nn-init) module.
show_doc(Learner.mixup)
# Uses [`MixUpCallback`](/callbacks.mixup.html#MixUpCallback).
show_doc(Learner.backward)
show_doc(Learner.create_opt)
# You generally won't need to call this yourself - it's used to create the [`optim`](https://pytorch.org/docs/stable/optim.html#module-torch.optim) optimizer before fitting the model.
show_doc(Learner.dl)
show_doc(Recorder, title_level=2)
# A [`Learner`](/basic_train.html#Learner) creates a [`Recorder`](/basic_train.html#Recorder) object automatically - you do not need to explicitly pass to `callback_fns` - because other callbacks rely on it being available. It stores the smoothed loss, hyperparameter values, and metrics each batch, and provides plotting methods for each. Note that [`Learner`](/basic_train.html#Learner) automatically sets an attribute with the snake-cased name of each callback, so you can access this through `Learner.recorder`, as shown below.
# ### Plotting methods
show_doc(Recorder.plot)
# This is mainly used with the learning rate finder, since it shows a scatterplot of loss vs learning rate.
learn = create_cnn(data, models.resnet18, metrics=accuracy)
learn.lr_find()
learn.recorder.plot()
show_doc(Recorder.plot_losses)
# Note that validation losses are only calculated once per epoch, whereas training losses are calculated after every batch.
learn.fit_one_cycle(2)
learn.recorder.plot_losses()
show_doc(Recorder.plot_lr)
learn.recorder.plot_lr(show_moms=True)
show_doc(Recorder.plot_metrics)
# Note that metrics are only collected at the end of each epoch, so you'll need to train at least two epochs to have anything to show here.
learn.recorder.plot_metrics()
# ### Callback methods
# You don't call these yourself - they're called by fastai's [`Callback`](/callback.html#Callback) system automatically to enable the class's functionality.
show_doc(Recorder.on_backward_begin)
show_doc(Recorder.on_batch_begin)
show_doc(Recorder.on_epoch_end)
show_doc(Recorder.on_train_begin)
# ### Inner functions
# The following functions are used along the way by the [`Recorder`](/basic_train.html#Recorder) or can be called by other callbacks.
show_doc(Recorder.add_metrics)
show_doc(Recorder.add_metric_names)
show_doc(Recorder.format_stats)
# ## Module functions
# Generally you'll want to use a [`Learner`](/basic_train.html#Learner) to train your model, since they provide a lot of functionality and make things easier. However, for ultimate flexibility, you can call the same underlying functions that [`Learner`](/basic_train.html#Learner) calls behind the scenes:
show_doc(fit)
# Note that you have to create the `Optimizer` yourself if you call this function, whereas [`Learn.fit`](/basic_train.html#fit) creates it for you automatically.
show_doc(train_epoch)
# You won't generally need to call this yourself - it's what [`fit`](/basic_train.html#fit) calls for each epoch.
show_doc(validate)
# This is what [`fit`](/basic_train.html#fit) calls after each epoch. You can call it if you want to run inference on a [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) manually.
show_doc(get_preds)
show_doc(loss_batch)
# You won't generally need to call this yourself - it's what [`fit`](/basic_train.html#fit) and [`validate`](/basic_train.html#validate) call for each batch. It only does a backward pass if you set `opt`.
# ## Other classes
show_doc(LearnerCallback, title_level=3)
show_doc(RecordOnCPU, title_level=3)
# ## Undocumented Methods - Methods moved below this line will intentionally be hidden
show_doc(Learner.tta_only)
show_doc(Learner.TTA)
show_doc(RecordOnCPU.on_batch_begin)
# ## New Methods - Please document or move to the undocumented section
show_doc(Learner.distributed)
