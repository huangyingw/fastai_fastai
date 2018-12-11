
# coding: utf-8

# # GANs

get_ipython().run_line_magic('matplotlib', 'inline')
from fastai.gen_doc.nbdoc import *
from fastai import *
from fastai.vision import *
from fastai.vision.gan import *


# GAN stands for [Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf) and were invented by Ian Goodfellow. The concept is that we will train two models at the same time: a generator and a critic. The generator will try to make new images similar to the ones in our dataset, and the critic's job will try to classify real images from the fake ones the generator does. The generator returns images, the discriminator a feature map (it can be a single number depending on the input size). Usually the discriminator will be trained to retun 0. everywhere for fake images and 1. everywhere for real ones.
#
# This module contains all the necessary function to create a GAN.

# We train them against each other in the sense that at each step (more or less), we:
# 1. Freeze the generator and train the discriminator for one step by:
#   - getting one batch of true images (let's call that `real`)
#   - generating one batch of fake images (let's call that `fake`)
#   - have the discriminator evaluate each batch and compute a loss function from that; the important part is that it rewards positively the detection of real images and penalizes the fake ones
#   - update the weights of the discriminator with the gradients of this loss
#
#
# 2. Freeze the discriminator and train the generator for one step by:
#   - generating one batch of fake images
#   - evaluate the discriminator on it
#   - return a loss that rewards posisitivly the discriminator thinking those are real images; the important part is that it rewards positively the detection of real images and penalizes the fake ones
#   - update the weights of the generator with the gradients of this loss

show_doc(GANLearner)


# This is the general constructor to create a GAN, you might want to use one of the factory methods that are easier to use. Create a GAN from [`data`](/vision.data.html#vision.data), a `generator` and a `critic`. The [`data`](/vision.data.html#vision.data) should have the inputs the `generator` will expect and the images wanted as targets.
#
# `gen_loss_func` is the loss function that will be applied to the `generator`. It takes three argument `fake_pred`, `target`, `output` and should return a rank 0 tensor. `output` is the result of the `generator` applied to the input (the xs of the batch), `target` is the ys of the batch and `fake_pred` is the result of the `discriminator` being given `output`. `output`and `target` can be used to add a specific loss to the GAN loss (pixel loss, feature loss) and for a good training of the gan, the loss should encourage `fake_pred` to be as close to 1 as possible (the `generator` is trained to fool the `critic`).
#
# `crit_loss_func` is the loss function that will be applied to the `critic`. It takes two arguments `real_pred` and `fake_pred`. `real_pred` is the result of the `critic` on the target images (the ys of the batch) and `fake_pred` is the result of the `critic` applied on a batch of fake, generated byt the `generator` from the xs of the batch.
#
# `switcher` is a [`Callback`](/callback.html#Callback) that should tell the GAN when to switch from critic to generator and vice versa. By default it does 5 iterations of the critic for 1 iteration of the generator. The model begins the training with the `generator` if `gen_first=True`. If `switch_eval=True`, the model that isn't trained is switched on eval mode (left in training mode otherwise, which means some statistics like the running mean in batchnorm layers are updated, or the dropouts are applied).
#
# `clip` should be set to a certain value if one wants to clip the weights (see the [Wassertein GAN](https://arxiv.org/pdf/1701.07875.pdf) for instance).
#
# If `show_img=True`, one image generated by the GAN is shown at the end of each epoch.

# ### Factory methods

show_doc(GANLearner.from_learners)


# Directly creates a [`GANLearner`](/vision.gan.html#GANLearner) from two [`Learner`](/basic_train.html#Learner): one for the `generator` and one for the `critic`. The `switcher` and all `kwargs` will be passed to the initialization of [`GANLearner`](/vision.gan.html#GANLearner) along with the following loss functions:
#
# - `loss_func_crit` is the mean of `learn_crit.loss_func` applied to `real_pred` and a target of ones with `learn_crit.loss_func` applied to `fake_pred` and a target of zeros
# - `loss_func_gen` is the mean of `learn_crit.loss_func` applied to `fake_pred` and a target of ones (to full the discriminator) with `learn_gen.loss_func` applied to `output` and `target`. The weights of each of those contributions can be passed in `weights_gen` (default is 1. and 1.)

show_doc(GANLearner.wgan)


# The Wasserstein GAN is detailed in [this article]. `switcher` and the `kwargs` will be passed to the [`GANLearner`](/vision.gan.html#GANLearner) init, `clip`is the weight clipping.

# ## Switchers

# In any GAN training, you will need to tell the [`Learner`](/basic_train.html#Learner) when to switch from generator to critic and vice versa. The two following [`Callback`](/callback.html#Callback) are examples to help you with that.

show_doc(FixedGANSwitcher, title_level=3)


show_doc(FixedGANSwitcher.on_train_begin)


show_doc(FixedGANSwitcher.on_batch_end)


show_doc(AdaptiveGANSwitcher, title_level=3)


show_doc(AdaptiveGANSwitcher.on_batch_end)


# ## Specific models

show_doc(basic_critic)


# This model contains a first 4 by 4 convolutional layer of stride 2 from `n_channels` to `n_features` followed by `n_extra_layers` 3 by 3 convolutional layer of stride 1. Then we put as many 4 by 4 convolutional layer of stride 2 with a number of features multiplied by 2 at each stage so that the `in_size` becomes 1. `kwargs` can be used to customize the convolutional layers and are passed to [`conv_layer`](/layers.html#conv_layer).

show_doc(basic_generator)


# This model contains a first 4 by 4 transposed convolutional layer of stride 1 from `noise_size` to the last numbers of features of the corresponding critic. Then we put as many 4 by 4 transposed convolutional layer of stride 2 with a number of features divided by 2 at each stage so that the image ends up being of height and widht `in_size//2`. At the end, we add`n_extra_layers` 3 by 3 convolutional layer of stride 1. The last layer is a transpose convolution of size 4 by 4 and stride 2 followed by `tanh`. `kwargs` can be used to customize the convolutional layers and are passed to [`conv_layer`](/layers.html#conv_layer).

show_doc(GANTrainer)


# [`LearnerCallback`](/basic_train.html#LearnerCallback) that will be responsible to handle the two different optimizers (one for the generator and one for the critic), and do all the work behind the scenes so that the generator (or the critic) are in training mode with parameters requirement gradients each time we switch.
#
# `switch_eval=True` means that the [`GANTrainer`](/vision.gan.html#GANTrainer) will put the model that isn't training into eval mode (if it's `False` its running statistics like in batchnorm layers will be updated and dropout will be applied). `clip` is the clipping applied to the weights (if not `None`). `beta` is the coefficient for the moving averages as the [`GANTrainer`](/vision.gan.html#GANTrainer)tracks separately the generator loss and the critic loss. `gen_first=True` means the training begins with the generator (with the critic if it's `False`). If `show_img=True` we show a generated image at the end of each epoch.

show_doc(GANTrainer.switch)


# If `gen_mode` is left as `None`, just put the model in the other mode (critic if it was in generator mode and vice versa).

show_doc(GANTrainer.on_train_begin)


show_doc(GANTrainer.on_epoch_begin)


show_doc(GANTrainer.on_batch_begin)


show_doc(GANTrainer.on_backward_begin)


show_doc(GANTrainer.on_epoch_end)


show_doc(GANTrainer.on_train_end)


show_doc(GANModule)


# If `gen_mode` is left as `None`, just put the model in the other mode (critic if it was in generator mode and vice versa).

show_doc(GANModule.switch)


show_doc(GANLoss)


# ## Undocumented Methods - Methods moved below this line will intentionally be hidden

show_doc(GANLoss.critic)


show_doc(GANModule.forward)


show_doc(GANLoss.generator)


# ## New Methods - Please document or move to the undocumented section
