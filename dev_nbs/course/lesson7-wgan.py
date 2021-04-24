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

# %matplotlib inline

from fastai.vision.all import *
from fastai.vision.gan import *

# ## LSun bedroom data

# For this lesson, we'll be using the bedrooms from the [LSUN dataset](http://lsun.cs.princeton.edu/2017/). The full dataset is a bit too large so we'll use a sample from [kaggle](https://www.kaggle.com/jhoward/lsun_bedroom).

path = untar_data(URLs.LSUN_BEDROOMS)

# We then grab all the images in the folder with the data block API. We don't create a validation set here for reasons we'll explain later. It consists of random noise of size 100 by default (can be changed if you replace `generate_noise` by `partial(generate_noise, size=...)`) as inputs and the images of bedrooms as targets.

dblock = DataBlock(blocks=(TransformBlock, ImageBlock),
                   get_x=generate_noise,
                   get_items=get_image_files,
                   splitter=IndexSplitter([]))


def get_dls(bs, size):
    dblock = DataBlock(blocks=(TransformBlock, ImageBlock),
                       get_x=generate_noise,
                       get_items=get_image_files,
                       splitter=IndexSplitter([]),
                       item_tfms=Resize(size, method=ResizeMethod.Crop),
                       batch_tfms=Normalize.from_stats(torch.tensor([0.5, 0.5, 0.5]), torch.tensor([0.5, 0.5, 0.5])))
    return dblock.dataloaders(path, path=path, bs=bs)


# We'll begin with a small size since GANs take a lot of time to train.

dls = get_dls(128, 64)

dls.show_batch(max_n=16)

# ## Models

# GAN stands for [Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf) and were invented by Ian Goodfellow. The concept is that we will train two models at the same time: a generator and a critic. The generator will try to make new images similar to the ones in our dataset, and the critic will try to classify real images from the ones the generator does. The generator returns images, the critic a single number (usually 0. for fake images and 1. for real ones).
#
# We train them against each other in the sense that at each step (more or less), we:
# 1. Freeze the generator and train the critic for one step by:
#   - getting one batch of true images (let's call that `real`)
#   - generating one batch of fake images (let's call that `fake`)
#   - have the critic evaluate each batch and compute a loss function from that; the important part is that it rewards positively the detection of real images and penalizes the fake ones
#   - update the weights of the critic with the gradients of this loss
#
#
# 2. Freeze the critic and train the generator for one step by:
#   - generating one batch of fake images
#   - evaluate the critic on it
#   - return a loss that rewards posisitivly the critic thinking those are real images; the important part is that it rewards positively the detection of real images and penalizes the fake ones
#   - update the weights of the generator with the gradients of this loss
#
# Here, we'll use the [Wassertein GAN](https://arxiv.org/pdf/1701.07875.pdf).

# We create a generator and a critic that we pass to `gan_learner`. The noise_size is the size of the random vector from which our generator creates images.

generator = basic_generator(64, n_channels=3, n_extra_layers=1)
critic = basic_critic(64, n_channels=3, n_extra_layers=1, act_cls=partial(nn.LeakyReLU, negative_slope=0.2))

learn = GANLearner.wgan(dls, generator, critic, opt_func=partial(Adam, mom=0.))

learn.recorder.train_metrics = True
learn.recorder.valid_metrics = False

learn.fit(30, 2e-4, wd=0)

# learn.gan_trainer.switch(gen_mode=True)
learn.show_results(max_n=16, figsize=(8, 8), ds_idx=0)
