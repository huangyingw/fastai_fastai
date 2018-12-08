
# coding: utf-8

# # Collaborative filtering

from fastai.gen_doc.nbdoc import *


# This package contains all the necessary functions to quickly train a model for a collaborative filtering task. Let's start by importing all we'll need.

from fastai import *
from fastai.collab import *


# ## Overview

# Collaborative filtering is when you're tasked to predict how much a user is going to like a certain item. The fastai library contains a [`CollabFilteringDataset`](/collab.html#CollabFilteringDataset) class that will help you create datasets suitable for training, and a function `get_colab_learner` to build a simple model directly from a ratings table. Let's first see how we can get started before devling in the documentation.
#
# For our example, we'll use a small subset of the [MovieLens](https://grouplens.org/datasets/movielens/) dataset. In there, we have to predict the rating a user gave a given movie (from 0 to 5). It comes in the form of a csv file where each line is the rating of a movie by a given person.

path = untar_data(URLs.ML_SAMPLE)
ratings = pd.read_csv(path / 'ratings.csv')
ratings.head()


# We'll first turn the `userId` and `movieId` columns in category codes, so that we can replace them with their codes when it's time to feed them to an `Embedding` layer. This step would be even more important if our csv had names of users, or names of items in it. To do it, we wimply have to call a [`CollabDataBunch`](/collab.html#CollabDataBunch) factory method.

data = CollabDataBunch.from_df(ratings)


# Now that this step is done, we can directly create a [`Learner`](/basic_train.html#Learner) object:

learn = collab_learner(data, n_factors=50, y_range=(0., 5.))


# And then immediately begin training

learn.fit_one_cycle(5, 5e-3, wd=0.1)


show_doc(CollabDataBunch, doc_string=False)


# This is the basic class to buil a [`DataBunch`](/basic_data.html#DataBunch) suitable for colaborative filtering.

show_doc(CollabDataBunch.from_df, doc_string=False)


# Takes a `ratings` dataframe and splits it randomly for train and test following `pct_val` (unless it's None). `user_name`, `item_name` and `rating_name` give the names of the corresponding columns (defaults to the first, the second and the third column). Optionally a `test` dataframe can be passed an a `seed` for the separation between training and validation set. The `kwargs` will be passed to [`DataBunch.create`](/basic_data.html#DataBunch.create).

# ## Model and [`Learner`](/basic_train.html#Learner)

show_doc(EmbeddingDotBias, doc_string=False, title_level=3)


# Creates a simple model with `Embedding` weights and biases for `n_users` and `n_items`, with `n_factors` latent factors. Takes the dot product of the embeddings and adds the bias, then if `y_range` is specified, feed the result to a sigmoid rescaled to go from `y_range[0]` to `y_range[1]`.

show_doc(collab_learner, doc_string=False)


# Creates a [`Learner`](/basic_train.html#Learner) object built from the data in `ratings`, `pct_val`, `user_name`, `item_name`, `rating_name` to [`CollabFilteringDataset`](/collab.html#CollabFilteringDataset). Optionally, creates another [`CollabFilteringDataset`](/collab.html#CollabFilteringDataset) for `test`. `kwargs` are fed to [`DataBunch.create`](/basic_data.html#DataBunch.create) with these datasets. The model is given by [`EmbeddingDotBias`](/collab.html#EmbeddingDotBias) with `n_factors` if `use_nn` is `False`, and is a neural net with `emb_szs` otherwise. In both cases the numbers of users and items will be inferred from the data, `y_range` is the range of the output (optional) and you can pass [`metrics`](/metrics.html#metrics).

# ## Links with the Data Block API

show_doc(CollabLine, doc_string=False, title_level=3)


# Subclass of [`TabularLine`](/tabular.data.html#TabularLine) for collaborative filtering.

show_doc(CollabList, title_level=3, doc_string=False)


# Subclass of [`TabularList`](/tabular.data.html#TabularList) for collaborative filtering.

# ## Undocumented Methods - Methods moved below this line will intentionally be hidden

show_doc(EmbeddingDotBias.forward)


# ## New Methods - Please document or move to the undocumented section
