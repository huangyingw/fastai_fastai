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
from fastai.collab import *
from fastai.tabular.all import *
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# # Collaborative filtering tutorial
#
# > Using the fastai library for collaborative filtering.


# +
# all_slow
# -

# This tutorial highlights on how to quickly build a `Learner` and train a model on collaborative filtering tasks.

# ## Training a model

# For this tutorial, we will use the [Movielens 100k data dataset](https://grouplens.org/datasets/movielens/100k/). We can download it easily and decompress it with the following function:

path = untar_data(URLs.ML_100k)

# The main table is in `u.data`. Since it's not a proper csv, we have to specify a few things while opening it: the tab delimiter, the columns we want to keep and their names.

ratings = pd.read_csv(path / 'u.data', delimiter='\t', header=None,
                      usecols=(0, 1, 2), names=['user', 'movie', 'rating'])
ratings.head()

# Movie ids are not ideal to look at things, so we load the corresponding movie id to the title that is in the table `u.item`:

movies = pd.read_csv(path / 'u.item', delimiter='|', encoding='latin-1',
                     usecols=(0, 1), names=('movie', 'title'), header=None)
movies.head()

# Next we merge it to our ratings table:

ratings = ratings.merge(movies)
ratings.head()

# We can then build a `DataLoaders` object from this table. By default, it takes the first column for user, the second column for the item (here our movies) and the third column for the ratings. We need to change the value of `item_name` in our case, to use the titles instead of the ids:

dls = CollabDataLoaders.from_df(ratings, item_name='title', bs=64)

# In all applications, when the data has been assembled in a `DataLoaders`, you can have a look at it with the `show_batch` method:

dls.show_batch()

# fastai can create and train a collaborative filtering model by using `collab_learner`:

learn = collab_learner(dls, n_factors=50, y_range=(0, 5.5))

# It uses a simple dot product model with 50 latent factors. To train it using the 1cycle policy, we just run this command:

learn.fit_one_cycle(5, 5e-3, wd=0.1)

# Here's [some benchmarks](https://www.librec.net/release/v1.3/example.html) on the same dataset for the popular Librec system for collaborative filtering. They show best results based on RMSE of 0.91 (scroll down to the 100k dataset), which corresponds to an MSE of `0.91**2 = 0.83`. So in less than a minute, we got pretty good results!

# ## Interpretation

# Let's analyze the results of our previous model. We will keep the 1000 most rated movies for this:

g = ratings.groupby('title')['rating'].count()
top_movies = g.sort_values(ascending=False).index.values[:1000]
top_movies[:10]

# ### Movie bias

# Our model has learned one bias per movie, a unique number independent of users that can be interpreted as the intrinsic "value" of the movie. We can grab the bias of each movie in our `top_movies` list with the following command:

movie_bias = learn.model.bias(top_movies, is_item=True)
movie_bias.shape

# Let's compare those biases with the average ratings:

mean_ratings = ratings.groupby('title')['rating'].mean()
movie_ratings = [(b, i, mean_ratings.loc[i]) for i, b in zip(top_movies, movie_bias)]

# Now let's have a look at the movies with the worst bias:


def item0(o): return o[0]


sorted(movie_ratings, key=item0)[:15]

# Or the ones with the best bias:

sorted(movie_ratings, key=lambda o: o[0], reverse=True)[:15]

# There is certainly a strong correlation!

# ### Movie weights

# Now let's try to analyze the latent factors our model has learned. We can grab the weights for each movie in `top_movies` the same way as we did for the bias before.

movie_w = learn.model.weight(top_movies, is_item=True)
movie_w.shape

# Let's try a PCA to reduce the dimensions and see if we can see what the model learned:

movie_pca = movie_w.pca(3)
movie_pca.shape

fac0, fac1, fac2 = movie_pca.t()
movie_comp = [(f, i) for f, i in zip(fac0, top_movies)]

# Here are the highest score on the first dimension:

sorted(movie_comp, key=itemgetter(0), reverse=True)[:10]

# And the worst:

sorted(movie_comp, key=itemgetter(0))[:10]

# Same thing for our second dimension:

movie_comp = [(f, i) for f, i in zip(fac1, top_movies)]

sorted(movie_comp, key=itemgetter(0), reverse=True)[:10]

sorted(movie_comp, key=itemgetter(0))[:10]

# And we can even plot the movies according to their scores on those dimensions:

idxs = np.random.choice(len(top_movies), 50, replace=False)
idxs = list(range(50))
X = fac0[idxs]
Y = fac2[idxs]
plt.figure(figsize=(15, 15))
plt.scatter(X, Y)
for i, x, y in zip(top_movies[idxs], X, Y):
    plt.text(x, y, i, color=np.random.rand(3) * 0.7, fontsize=11)
plt.show()
