# coding: utf-8
# # Collaborative filtering
from fastai.gen_doc.nbdoc import *
# This package contains all the necessary functions to quickly train a model for a collaborative filtering task. Let's start by importing all we'll need.
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
show_doc(CollabDataBunch)
# The init function shouldn't be called directly (as it's the one of a basic [`DataBunch`](/basic_data.html#DataBunch)), instead, you'll want to use the following factory method.
show_doc(CollabDataBunch.from_df)
# Take a `ratings` dataframe and splits it randomly for train and test following `pct_val` (unless it's None). `user_name`, `item_name` and `rating_name` give the names of the corresponding columns (defaults to the first, the second and the third column). Optionally a `test` dataframe can be passed an a `seed` for the separation between training and validation set. The `kwargs` will be passed to [`DataBunch.create`](/basic_data.html#DataBunch.create).
# ## Model and [`Learner`](/basic_train.html#Learner)
show_doc(CollabLearner, title_level=3)
# This is a subclass of [`Learner`](/basic_train.html#Learner) that just introduces helper functions to analyze results, the initialization is the same as a regular [`Learner`](/basic_train.html#Learner).
show_doc(CollabLearner.bias)
show_doc(CollabLearner.get_idx)
show_doc(CollabLearner.weight)
show_doc(EmbeddingDotBias, title_level=3)
# Creates a simple model with `Embedding` weights and biases for `n_users` and `n_items`, with `n_factors` latent factors. Takes the dot product of the embeddings and adds the bias, then if `y_range` is specified, feed the result to a sigmoid rescaled to go from `y_range[0]` to `y_range[1]`.
show_doc(EmbeddingNN, title_level=3)
# `emb_szs` will overwrite the default and `kwargs` are passed to [`TabularModel`](/tabular.models.html#TabularModel).
show_doc(collab_learner)
# More specifically, binds [`data`](/data.html#data) with a model that is either an [`EmbeddingDotBias`](/collab.html#EmbeddingDotBias) with `n_factors` if `use_nn=False` or a [`EmbeddingNN`](/collab.html#EmbeddingNN) with `emb_szs` otherwise. In both cases the numbers of users and items will be inferred from the data, `y_range` can be specifided in the `kwargs` and you can pass [`metrics`](/metrics.html#metrics) or `wd` to the [`Learner`](/basic_train.html#Learner) constructor.
# ## Links with the Data Block API
show_doc(CollabLine, doc_string=False, title_level=3)
# Subclass of [`TabularLine`](/tabular.data.html#TabularLine) for collaborative filtering.
show_doc(CollabList, title_level=3, doc_string=False)
# Subclass of [`TabularList`](/tabular.data.html#TabularList) for collaborative filtering.
# ## Undocumented Methods - Methods moved below this line will intentionally be hidden
show_doc(EmbeddingDotBias.forward)
show_doc(CollabList.reconstruct)
show_doc(EmbeddingNN.forward)
# ## New Methods - Please document or move to the undocumented section
