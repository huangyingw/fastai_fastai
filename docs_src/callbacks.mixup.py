
# coding: utf-8

# # Mixup data augmentation

from fastai.gen_doc.nbdoc import *
from fastai.callbacks.mixup import *
from fastai.vision import *
from fastai import *


# ## What is Mixup?

# This module contains the implementation of a data augmentation technique called [Mixup](https://arxiv.org/abs/1710.09412). It is extremely efficient at regularizing models in computer vision (we used it to get our time to train CIFAR10 to 94% on one GPU to 6 minutes).
#
# As the name kind of suggests, the authors of the mixup article propose to train the model on a mix of the pictures of the training set. Let’s say we’re on CIFAR10 for instance, then instead of feeding the model the raw images, we take two (which could be in the same class or not) and do a linear combination of them: in terms of tensor it’s
#
# `new_image = t * image1 + (1-t) * image2`
#
# where t is a float between 0 and 1. Then the target we assign to that image is the same combination of the original targets:
#
# `new_target = t * target1 + (1-t) * target2`
#
# assuming your targets are one-hot encoded (which isn’t the case in pytorch usually). And that’s as simple as this.
#
# ![mixup](imgs/mixup.png)
#
# Dog or cat? The right answer here is 70% dog and 30% cat!
#
# As the picture above shows, it’s a bit hard for a human eye to comprehend the pictures obtained (although we do see the shapes of a dog and a cat) but somehow, it makes a lot of sense to the model which trains more efficiently. The final loss (training or validation) will be higher than when training without mixup even if the accuracy is far better, which means that a model trained like this will make predictions that are a bit less confident.

# ## Basic Training

# To test this method, we will first build a [`simple_cnn`](/layers.html#simple_cnn) and train it like we did with [`basic_train`](/basic_train.html#basic_train) so we can compare its results with a network trained with Mixup.

path = untar_data(URLs.MNIST_SAMPLE)
data = ImageDataBunch.from_folder(path)
model = simple_cnn((3, 16, 16, 2))
learn = Learner(data, model, metrics=[accuracy])


learn.fit(8)


# ## Mixup implementation in the library

# In the original article, the authors suggested four things:
#
#     1. Create two separate dataloaders and draw a batch from each at every iteration to mix them up
#     2. Draw a t value following a beta distribution with a parameter alpha (0.4 is suggested in their article)
#     3. Mix up the two batches with the same value t.
#     4. Use one-hot encoded targets
#
# The implementation of this module is based on these suggestions but was modified when experiments suggested modifications with positive impact in performance.

# The authors suggest to use the beta distribution with the same parameters alpha. Why do they suggest this? Well it looks like this:
#
# ![betadist](imgs/betadist-mixup.png)
#
# so it means there is a very high probability of picking values close to 0 or 1 (in which case the image is almost from 1 category) and then a somewhat constant probability of picking something in the middle (0.33 as likely as 0.5 for instance).
#
# While this works very well, it’s not the fastest way we can do this and this is the first suggestion we will adjust. The main point that slows down this process is wanting two different batches at every iteration (which means loading twice the amount of images and applying to them the other data augmentation function). To avoid this slow down, ou be a little smarter and mixup a batch with a shuffled version of itself (this way the images mixed up are still different).
#
# Using the same parameter t for the whole batch is another suggestion we will modify. In our experiments, we noticed that the model can train faster if we draw a different `t` for every image in the batch (both options get to the same result in terms of accuracy, it’s just that one arrives there more slowly).
# The last trick we have to apply with this is that there can be some duplicates with this strategy: let’s say we decide to mix `image0` with `image1` then `image1` with `image0`, and that we draw `t=0.1` for the first, and `t=0.9` for the second. Then
#
# `image0 * 0.1 + shuffle0 * (1-0.1) = image0 * 0.1 + image1 * 0.9`
#
# and
#
# `image1 * 0.9 + shuffle1 * (1-0.9) = image1 * 0.9 + image0 * 0.1`
#
# will be the sames. Of course we have to be a bit unlucky but in practice, we saw there was a drop in accuracy by using this without removing those duplicates. To avoid them, the tricks is to replace the vector of parameters `t` we drew by:
#
# `t = max(t, 1-t)`
#
# The beta distribution with the two parameters equal is symmetric in any case, and this way we insure that the biggest coefficient is always near the first image (the non-shuffled batch).

# ## Adding Mixup to the Mix

# Now we will add [`MixUpCallback`](/callbacks.mixup.html#MixUpCallback) to our Learner so that it modifies our input and target accordingly. The [`mixup`](/train.html#mixup) function does that for us behind the scene, with a few other tweaks detailed below.

model = simple_cnn((3, 16, 16, 2))
learner = Learner(data, model, metrics=[accuracy]).mixup()
learner.fit(8)


# Training the net with Mixup improves the best accuracy. Note that the validation loss is higher than without MixUp, because the model makes less confident predictions: without mixup, most precisions are very close to 0. or 1. (in terms of probability) whereas the model with MixUp will give predictions that are more nuanced. Be sure to know what is the thing you want to optimize (lower loss or better accuracy) before using it.

show_doc(MixUpCallback, doc_string=False)


# Create a [`Callback`](/callback.html#Callback) for mixup on `learn` with a parameter `alpha` for the beta distribution. `stack_x` and `stack_y` determines if we stack our inputs/targets with the vector lambda drawn or do the linear combination (in general, we stack the inputs or ouputs when they correspond to categories or classes and do the linear combination otherwise).

show_doc(MixUpCallback.on_batch_begin, doc_string=False)


# Draws a vector of lambda following a beta distribution with `self.alpha` and operates the mixup on `last_input` and `last_target` according to `self.stack_x` and `self.stack_y`.

# ## Dealing with the loss

# We often have to modify the loss so that it is compatible with Mixup: pytorch was very careful to avoid one-hot encoding targets when it could, so it seems a bit of a drag to undo this. Fortunately for us, if the loss is a classic [cross-entropy](https://pytorch.org/docs/stable/nn.html#torch.nn.functional.cross_entropy), we have
#
# `loss(output, new_target) = t * loss(output, target1) + (1-t) * loss(output, target2)`
#
# so we won’t one-hot encode anything and just compute those two losses then do the linear combination.
#
# The following class is used to adapt the loss to mixup. Note that the [`mixup`](/train.html#mixup) function will use it to change the `Learner.loss_func` if necessary.

show_doc(MixUpLoss, doc_string=False, title_level=3)


# Create a loss function from `crit` that is compatible with MixUp.

# ## Undocumented Methods - Methods moved below this line will intentionally be hidden

show_doc(MixUpLoss.forward)
