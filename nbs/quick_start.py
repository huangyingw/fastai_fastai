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

# hide
# skip
from fastai.tabular.all import *
from fastai.collab import *
from fastai.text.all import *
from fastai.vision.all import *
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# +
# all_slow
# -

# hide

# # fastai applications - quick start

# fastai's applications all use the same basic steps and code:
#
# - Create appropriate `DataLoaders`
# - Create a `Learner`
# - Call a *fit* method
# - Make predictions or view results.
#
# In this quick start, we'll show these steps for a wide range of difference applications and datasets. As you'll see, the code in each case is extremely similar, despite the very different models and data being used.

# ## Computer vision classification

# The code below does the following things:
#
# 1. A dataset called the [Oxford-IIIT Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/) that contains 7,349 images of cats and dogs from 37 different breeds will be downloaded from the fast.ai datasets collection to the GPU server you are using, and will then be extracted.
# 2. A *pretrained model* that has already been trained on 1.3 million images, using a competition-winning model will be downloaded from the internet.
# 3. The pretrained model will be *fine-tuned* using the latest advances in transfer learning, to create a model that is specially customized for recognizing dogs and cats.
#
# The first two steps only need to be run once. If you run it again, it will use the dataset and model that have already been downloaded, rather than downloading them again.

# +
path = untar_data(URLs.PETS) / 'images'


def is_cat(x): return x[0].isupper()


dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224))

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)
# -

# You can do inference with your model with the `predict` method:

img = PILImage.create('images/cat.jpg')
img

is_cat, _, probs = learn.predict(img)
print(f"Is this a cat?: {is_cat}.")
print(f"Probability it's a cat: {probs[1].item():.6f}")

# ### Computer vision segmentation

# Here is how we can train a segmentation model with fastai, using a subset of the [*Camvid* dataset](http://www0.cs.ucl.ac.uk/staff/G.Brostow/papers/Brostow_2009-PRL.pdf):

# +
path = untar_data(URLs.CAMVID_TINY)
dls = SegmentationDataLoaders.from_label_func(
    path, bs=8, fnames=get_image_files(path / "images"),
    label_func=lambda o: path / 'labels' / f'{o.stem}_P{o.suffix}',
    codes=np.loadtxt(path / 'codes.txt', dtype=str)
)

learn = unet_learner(dls, resnet34)
learn.fine_tune(8)
# -

# We can visualize how well it achieved its task, by asking the model to color-code each pixel of an image.

learn.show_results(max_n=6, figsize=(7, 8))

# ## Natural language processing

# Here is all of the code necessary to train a model that can classify the sentiment of a movie review better than anything that existed in the world just five years ago:

dls = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid='test')
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
learn.fine_tune(2, 1e-2)

# Predictions are done with `predict`, as for computer vision:

learn.predict("I really liked that movie!")

# ## Tabular

# Building models from plain *tabular* data is done using the same basic steps as the previous models. Here is the code necessary to train a model that will predict whether a person is a high-income earner, based on their socioeconomic background:

# +
path = untar_data(URLs.ADULT_SAMPLE)

dls = TabularDataLoaders.from_csv(path / 'adult.csv', path=path, y_names="salary",
                                  cat_names=['workclass', 'education', 'marital-status', 'occupation',
                                             'relationship', 'race'],
                                  cont_names=['age', 'fnlwgt', 'education-num'],
                                  procs=[Categorify, FillMissing, Normalize])

learn = tabular_learner(dls, metrics=accuracy)
learn.fit_one_cycle(2)
# -

# ## Recommendation systems

# Recommendation systems are very important, particularly in e-commerce. Companies like Amazon and Netflix try hard to recommend products or movies that users might like. Here's how to train a model that will predict movies people might like, based on their previous viewing habits, using the [MovieLens dataset](https://doi.org/10.1145/2827872):

path = untar_data(URLs.ML_SAMPLE)
dls = CollabDataLoaders.from_csv(path / 'ratings.csv')
learn = collab_learner(dls, y_range=(0.5, 5.5))
learn.fine_tune(6)

# We can use the same `show_results` call we saw earlier to view a few examples of user and movie IDs, actual ratings, and predictions:

learn.show_results()
