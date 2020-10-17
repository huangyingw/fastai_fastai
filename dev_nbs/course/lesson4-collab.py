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

from fastai.collab import *
from fastai.tabular.all import *

# ## Collaborative filtering example

# `collab` models use data in a `DataFrame` of user, items, and ratings.

user, item, title = 'userId', 'movieId', 'title'

path = untar_data(URLs.ML_SAMPLE)
path

ratings = pd.read_csv(path / 'ratings.csv')
ratings.head()

# That's all we need to create and train a model:

dls = CollabDataLoaders.from_df(ratings, bs=64, seed=42)

y_range = [0, 5.5]

learn = collab_learner(dls, n_factors=50, y_range=y_range)

learn.fit_one_cycle(3, 5e-3)

# ## Movielens 100k

# Let's try with the full Movielens 100k data dataset, available from http://files.grouplens.org/datasets/movielens/ml-100k.zip

path = Config().data / 'ml-100k'

ratings = pd.read_csv(path / 'u.data', delimiter='\t', header=None,
                      names=[user, item, 'rating', 'timestamp'])
ratings.head()

movies = pd.read_csv(path / 'u.item', delimiter='|', encoding='latin-1', header=None,
                     names=[item, 'title', 'date', 'N', 'url', *[f'g{i}' for i in range(19)]])
movies.head()

len(ratings)

rating_movie = ratings.merge(movies[[item, title]])
rating_movie.head()

dls = CollabDataLoaders.from_df(rating_movie, seed=42, valid_pct=0.1, bs=64, item_name=title, path=path)

dls.show_batch()

y_range = [0, 5.5]

learn = collab_learner(dls, n_factors=40, y_range=y_range)

learn.lr_find()

learn.fit_one_cycle(5, 5e-3, wd=1e-1)

learn.save('dotprod')

# Here's [some benchmarks](https://www.librec.net/release/v1.3/example.html) on the same dataset for the popular Librec system for collaborative filtering. They show best results based on RMSE of 0.91, which corresponds to an MSE of `0.91**2 = 0.83`.

# ## Interpretation

# ### Setup

learn.load('dotprod')

learn.model

g = rating_movie.groupby('title')['rating'].count()
top_movies = g.sort_values(ascending=False).index.values[:1000]
top_movies[:10]

# ### Movie bias

movie_bias = learn.model.bias(top_movies, is_item=True)
movie_bias.shape

mean_ratings = rating_movie.groupby('title')['rating'].mean()
movie_ratings = [(b, i, mean_ratings.loc[i]) for i, b in zip(top_movies, movie_bias)]

def item0(o): return o[0]

sorted(movie_ratings, key=item0)[:15]

sorted(movie_ratings, key=lambda o: o[0], reverse=True)[:15]

# ### Movie weights

movie_w = learn.model.weight(top_movies, is_item=True)
movie_w.shape

movie_pca = movie_w.pca(3)
movie_pca.shape

fac0, fac1, fac2 = movie_pca.t()
movie_comp = [(f, i) for f, i in zip(fac0, top_movies)]

sorted(movie_comp, key=itemgetter(0), reverse=True)[:10]

sorted(movie_comp, key=itemgetter(0))[:10]

movie_comp = [(f, i) for f, i in zip(fac1, top_movies)]

sorted(movie_comp, key=itemgetter(0), reverse=True)[:10]

sorted(movie_comp, key=itemgetter(0))[:10]

idxs = np.random.choice(len(top_movies), 50, replace=False)
idxs = list(range(50))
X = fac0[idxs]
Y = fac2[idxs]
plt.figure(figsize=(15, 15))
plt.scatter(X, Y)
for i, x, y in zip(top_movies[idxs], X, Y):
    plt.text(x, y, i, color=np.random.rand(3) * 0.7, fontsize=11)
plt.show()
