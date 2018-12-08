
# coding: utf-8

# # Computer Vision Learner

# [`vision.learner`](/vision.learner.html#vision.learner) is the module that defines the [`create_cnn`](/vision.learner.html#create_cnn) method, to easily get a model suitable for transfer learning.

from fastai.gen_doc.nbdoc import *
from fastai.vision import *
from fastai import *


# ## Transfer learning

# Transfer learning is a technique where you use a model trained on a very large dataset (usually [ImageNet](http://image-net.org/) in computer vision) and then adapt it to your own dataset. The idea is that it has learned to recognize many features on all of this data, and that you will benefit from this knowledge, especially if your dataset is small, compared to starting from a randomly initiliazed model. It has been proved in [this article](https://arxiv.org/abs/1805.08974) on a wide range of tasks that transfer learning nearly always give better results.
#
# In practice, you need to change the last part of your model to be adapted to your own number of classes. Most convolutional models end with a few linear layers (a part will call head). The last convolutional layer will have analyzed features in the image that went through the model, and the job of the head is to convert those in predictions for each of our classes. In transfer learning we will keep all the convolutional layers (called the body or the backbone of the model) with their weights pretrained on ImageNet but will define a new head initiliazed randomly.
#
# Then we will train the model we obtain in two phases: first we freeze the body weights and only train the head (to convert those analyzed features into predictions for our own data), then we unfreeze the layers of the backbone (gradually if necessary) and fine-tune the whole model (possily using differential learning rates).
#
# The [`create_cnn`](/vision.learner.html#create_cnn) factory method helps you to automatically get a pretrained model from a given architecture with a custom head that is suitable for your data.

show_doc(create_cnn, doc_string=False)


# This method creates a [`Learner`](/basic_train.html#Learner) object from the [`data`](/vision.data.html#vision.data) object and model inferred from it with the backbone given in `arch`. Specifically, it will cut the model defined by `arch` (randomly initialized if `pretrained` is False) at the last convolutional layer by default (or as defined in `cut`, see below) and add:
# - an [`AdaptiveConcatPool2d`](/layers.html#AdaptiveConcatPool2d) layer,
# - a [`Flatten`](/layers.html#Flatten) layer,
# - blocks of \[[`nn.BatchNorm1d`](https://pytorch.org/docs/stable/nn.html#torch.nn.BatchNorm1d), [`nn.Dropout`](https://pytorch.org/docs/stable/nn.html#torch.nn.Dropout), [`nn.Linear`](https://pytorch.org/docs/stable/nn.html#torch.nn.Linear), [`nn.ReLU`](https://pytorch.org/docs/stable/nn.html#torch.nn.ReLU)\] layers.
#
# The blocks are defined by the `lin_ftrs` and `ps` arguments. Specifically, the first block will have a number of inputs inferred from the backbone `arch` and the last one will have a number of outputs equal to `data.c` (which contains the number of classes of the data) and the intermediate blocks have a number of inputs/outputs determined by `lin_frts` (of course a block has a number of inputs equal to the number of outputs of the previous block). The default is to have an intermediate hidden size of 512 (which makes two blocks `model_activation` -> 512 -> `n_classes`). If you pass a float then the final dropout layer will have the value `ps`, and the remaining will be `ps/2`. If you pass a list then the values are used for dropout probabilities directly.
#
# Note that the very last block doesn't have a [`nn.ReLU`](https://pytorch.org/docs/stable/nn.html#torch.nn.ReLU) activation, to allow you to use any final activation you want (generally included in the loss function in pytorch). Also, the backbone will be frozen if you choose `pretrained=True` (so only the head will train if you call [`fit`](/basic_train.html#fit)) so that you can immediately start phase one of training as described above.
#
# Alternatively, you can define your own `custom_head` to put on top of the backbone. If you want to specify where to split `arch` you should so in the argument `cut` which can either be the index of a specific layer (the result will not include that layer) or a function that, when passed the model, will return the backbone you want.
#
# The final model obtained by stacking the backbone and the head (custom or defined as we saw) is then separated in groups for gradual unfreezeing or differential learning rates. You can specify of to split the backbone in groups with the optional argument `split_on` (should be a function that returns those groups when given the backbone).
#
# The `kwargs` will be passed on to [`Learner`](/basic_train.html#Learner), so you can put here anything that [`Learner`](/basic_train.html#Learner) will accept ([`metrics`](/metrics.html#metrics), `loss_func`, `opt_func`...)

path = untar_data(URLs.MNIST_SAMPLE)
data = ImageDataBunch.from_folder(path)


learner = create_cnn(data, models.resnet18, metrics=[accuracy])
learner.fit_one_cycle(1, 1e-3)


learner.save('one_epoch')


# ### Get predictions

# Once you've actually trained your model, you may want to use it on a single image. This is done by using the following method.

show_doc(Learner.predict)


img = learner.data.train_ds[0][0]
learner.predict(img)


# Here the predict class for our iamge is '3', which corresponds to a label of 0. The probabilities the model found for each class are 99.65% and 0.35% respectively, so its confidence is pretty high.
#
# Note that if you want to load your trained model and use it on inference mode with the previous function, you can create a `cnn_learner` from empty data. First export the relevant bits of your data object by typing:

data.export()


# And then you can load an empty data object that has the same internal state like this:

empty_data = ImageDataBunch.load_empty(path, tfms=get_transforms()).normalize(imagenet_stats)
learn = create_cnn(empty_data, models.resnet18)
learn = learn.load('one_epoch')


# ### Customize your model

# You can customize [`create_cnn`](/vision.learner.html#create_cnn) for your own model's default `cut` and `split_on` functions by adding them to the dictionary `model_meta`. The key should be your model and the value should be a dictionary with the keys `cut` and `split_on` (see the source code for examples). The constructor will call [`create_body`](/vision.learner.html#create_body) and [`create_head`](/vision.learner.html#create_head) for you based on `cut`; you can also call them yourself, which is particularly useful for testing.

show_doc(create_body)


show_doc(create_head, doc_string=False)


# Model head that takes `nf` features, runs through `lin_ftrs`, and ends with `nc` classes. `ps` is the probability of the dropouts, as documented above in [`create_cnn`](/vision.learner.html#create_cnn).

show_doc(ClassificationInterpretation, title_level=3)


# This provides a confusion matrix and visualization of the most incorrect images. Pass in your [`data`](/vision.data.html#vision.data), calculated `preds`, actual `y`, and your `losses`, and then use the methods below to view the model interpretation results. For instance:

learn = create_cnn(data, models.resnet18)
learn.fit(1)
preds, y, losses = learn.get_preds(with_loss=True)
interp = ClassificationInterpretation(data, preds, y, losses)


# The following factory method gives a more convenient way to create an instance of this class:

show_doc(ClassificationInterpretation.from_learner)


# You can also use a shortcut `learn.interpret()` to do the same.

show_doc(Learner.interpret, full_name='interpret')


# Note that this shortcut is a [`Learner`](/basic_train.html#Learner) object/class method that can be called as: `learn.interpret()`.

show_doc(ClassificationInterpretation.plot_top_losses)


# The `k` items are arranged as a square, so it will look best if `k` is a square number (4, 9, 16, etc). The title of each image shows: prediction, actual, loss, probability of actual class.

interp.plot_top_losses(9, figsize=(7, 7))


show_doc(ClassificationInterpretation.top_losses)


# Returns tuple of *(losses,indices)*.

interp.top_losses(9)


show_doc(ClassificationInterpretation.plot_confusion_matrix)


# If [`normalize`](/vision.data.html#normalize), plots the percentages with `norm_dec` digits. `slice_size` can be used to avoid out of memory error if your set is too big. `kwargs` are passed to `plt.figure`.

interp.plot_confusion_matrix()


show_doc(ClassificationInterpretation.confusion_matrix)


interp.confusion_matrix()


show_doc(ClassificationInterpretation.most_confused)


# #### Working with large datasets

# When working with large datasets, memory problems can arise when computing the confusion matrix. For example, an error can look like this:
#
#     RuntimeError: $ Torch: not enough memory: you tried to allocate 64GB. Buy new RAM! at /opt/conda/conda-bld/pytorch-nightly_1540719301766/work/aten/src/TH/THGeneral.cpp:204
#
# In this case it is possible to force [`ClassificationInterpretation`](/vision.learner.html#ClassificationInterpretation) to compute the confusion matrix for data slices and then aggregate the result by specifying slice_size parameter.

interp.confusion_matrix(slice_size=10)


interp.plot_confusion_matrix(slice_size=10)


interp.most_confused(slice_size=10)


# ## GANs

show_doc(gan_learner)


# If `noise_size` is set, the GAN will generate fakes from a noise of this size, otherwise it'll use the inputs in data. If `wgan` is set to `True`, overrides the loss functions for a WGAN. `loss_funcD` and `loss_funcG` are used for discriminator and the generator. `kwargs` are passed to the [`Learner`](/basic_train.html#Learner) init.

show_doc(GANLearner, title_level=3, doc_string=False)


# Subclass of [`Learner`](/basic_train.html#Learner) to deal with `predict` and `show_results` for GANs.

# ## Undocumented Methods - Methods moved below this line will intentionally be hidden

show_doc(GANLearner)


show_doc(GANLearner.show_results)


show_doc(GANLearner.add_gan_trainer)


show_doc(GANLearner.predict)


# ## New Methods - Please document or move to the undocumented section
