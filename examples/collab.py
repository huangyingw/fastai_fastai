
# coding: utf-8

from fastai import *          # Quick access to most common functionality
from fastai.collab import *   # Quick access to collab filtering functionality


# ## Collaborative filtering example

# `collab` models use data in a `DataFrame` of user, items, and ratings.

path = untar_data(URLs.ML_SAMPLE)
path


ratings = pd.read_csv(path / 'ratings.csv')
series2cat(ratings, 'userId', 'movieId')
ratings.head()


data = CollabDataBunch.from_df(ratings, seed=42)


y_range = [0, 5.5]


# That's all we need to create and train a model:

learn = collab_learner(data, n_factors=50, y_range=y_range)
learn.fit_one_cycle(4, 5e-3)
