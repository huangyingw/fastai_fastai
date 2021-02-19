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
from nbdev.export import *
from nbdev.showdoc import *
from fastai.tabular.all import *
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# +
# default_exp collab
# default_class_lvl 3
# -

# export

# hide


# # Collaborative filtering
#
# > Tools to quickly get the data and train models suitable for collaborative filtering

# This module contains all the high-level functions you need in a collaborative filtering application to assemble your data, get a model and train it with a `Learner`. We will go other those in order but you can also check the [collaborative filtering tutorial](http://docs.fast.ai/tutorial.collab).

# ## Gather the data

# export
class TabularCollab(TabularPandas):
    "Instance of `TabularPandas` suitable for collaborative filtering (with no continuous variable)"
    with_cont = False


# This is just to use the internal of the tabular application, don't worry about it.

# +
# export
class CollabDataLoaders(DataLoaders):
    "Base `DataLoaders` for collaborative filtering."
    @delegates(DataLoaders.from_dblock)
    @classmethod
    def from_df(cls, ratings, valid_pct=0.2, user_name=None, item_name=None, rating_name=None, seed=None, path='.', **kwargs):
        "Create a `DataLoaders` suitable for collaborative filtering from `ratings`."
        user_name = ifnone(user_name, ratings.columns[0])
        item_name = ifnone(item_name, ratings.columns[1])
        rating_name = ifnone(rating_name, ratings.columns[2])
        cat_names = [user_name, item_name]
        splits = RandomSplitter(valid_pct=valid_pct, seed=seed)(range_of(ratings))
        to = TabularCollab(ratings, [Categorify], cat_names, y_names=[rating_name], y_block=TransformBlock(), splits=splits)
        return to.dataloaders(path=path, **kwargs)

    @classmethod
    def from_csv(cls, csv, **kwargs):
        "Create a `DataLoaders` suitable for collaborative filtering from `csv`."
        return cls.from_df(pd.read_csv(csv), **kwargs)


CollabDataLoaders.from_csv = delegates(to=CollabDataLoaders.from_df)(CollabDataLoaders.from_csv)
# -

# This class should not be used directly, one of the factory methods should be preferred instead. All those factory methods accept as arguments:
#
# - `valid_pct`: the random percentage of the dataset to set aside for validation (with an optional `seed`)
# - `user_name`: the name of the column containing the user (defaults to the first column)
# - `item_name`: the name of the column containing the item (defaults to the second column)
# - `rating_name`: the name of the column containing the rating (defaults to the third column)
# - `path`: the folder where to work
# - `bs`: the batch size
# - `val_bs`: the batch size for the validation `DataLoader` (defaults to `bs`)
# - `shuffle_train`: if we shuffle the training `DataLoader` or not
# - `device`: the PyTorch device to use (defaults to `default_device()`)

show_doc(CollabDataLoaders.from_df)

# Let's see how this works on an example:

path = untar_data(URLs.ML_SAMPLE)
ratings = pd.read_csv(path / 'ratings.csv')
ratings.head()

dls = CollabDataLoaders.from_df(ratings, bs=64)
dls.show_batch()

show_doc(CollabDataLoaders.from_csv)

dls = CollabDataLoaders.from_csv(path / 'ratings.csv', bs=64)


# ## Models

# fastai provides two kinds of models for collaborative filtering: a dot-product model and a neural net.

# export
class EmbeddingDotBias(Module):
    "Base dot model for collaborative filtering."

    def __init__(self, n_factors, n_users, n_items, y_range=None):
        self.y_range = y_range
        (self.u_weight, self.i_weight, self.u_bias, self.i_bias) = [Embedding(*o) for o in [
            (n_users, n_factors), (n_items, n_factors), (n_users, 1), (n_items, 1)
        ]]

    def forward(self, x):
        users, items = x[:, 0], x[:, 1]
        dot = self.u_weight(users) * self.i_weight(items)
        res = dot.sum(1) + self.u_bias(users).squeeze() + self.i_bias(items).squeeze()
        if self.y_range is None:
            return res
        return torch.sigmoid(res) * (self.y_range[1] - self.y_range[0]) + self.y_range[0]

    @classmethod
    def from_classes(cls, n_factors, classes, user=None, item=None, y_range=None):
        "Build a model with `n_factors` by inferring `n_users` and  `n_items` from `classes`"
        if user is None:
            user = list(classes.keys())[0]
        if item is None:
            item = list(classes.keys())[1]
        res = cls(n_factors, len(classes[user]), len(classes[item]), y_range=y_range)
        res.classes, res.user, res.item = classes, user, item
        return res

    def _get_idx(self, arr, is_item=True):
        "Fetch item or user (based on `is_item`) for all in `arr`"
        assert hasattr(self, 'classes'), "Build your model with `EmbeddingDotBias.from_classes` to use this functionality."
        classes = self.classes[self.item] if is_item else self.classes[self.user]
        c2i = {v: k for k, v in enumerate(classes)}
        try:
            return tensor([c2i[o] for o in arr])
        except Exception as e:
            print(f"""You're trying to access {'an item' if is_item else 'a user'} that isn't in the training data.
                  If it was in your original data, it may have been split such that it's only in the validation set now.""")

    def bias(self, arr, is_item=True):
        "Bias for item or user (based on `is_item`) for all in `arr`"
        idx = self._get_idx(arr, is_item)
        layer = (self.i_bias if is_item else self.u_bias).eval().cpu()
        return to_detach(layer(idx).squeeze(), gather=False)

    def weight(self, arr, is_item=True):
        "Weight for item or user (based on `is_item`) for all in `arr`"
        idx = self._get_idx(arr, is_item)
        layer = (self.i_weight if is_item else self.u_weight).eval().cpu()
        return to_detach(layer(idx), gather=False)


# The model is built with `n_factors` (the length of the internal vectors), `n_users` and `n_items`. For a given user and item, it grabs the corresponding weights and bias and returns
# ``` python
# torch.dot(user_w, item_w) + user_b + item_b
# ```
# Optionally, if `y_range` is passed, it applies a `SigmoidRange` to that result.

x, y = dls.one_batch()
model = EmbeddingDotBias(50, len(dls.classes['userId']), len(dls.classes['movieId']), y_range=(0, 5)
                         ).to(x.device)
out = model(x)
assert (0 <= out).all() and (out <= 5).all()

show_doc(EmbeddingDotBias.from_classes)

# `y_range` is passed to the main init. `user` and `item` are the names of the keys for users and items in `classes` (default to the first and second key respectively). `classes` is expected to be a dictionary key to list of categories like the result of `dls.classes` in a `CollabDataLoaders`:

dls.classes

# Let's see how it can be used in practice:

model = EmbeddingDotBias.from_classes(50, dls.classes, y_range=(0, 5)
                                      ).to(x.device)
out = model(x)
assert (0 <= out).all() and (out <= 5).all()

# Two convenience methods are added to easily access the weights and bias when a model is created with `EmbeddingDotBias.from_classes`:

show_doc(EmbeddingDotBias.weight)

# The elements of `arr` are expected to be class names (which is why the model needs to be created with `EmbeddingDotBias.from_classes`)

mov = dls.classes['movieId'][42]
w = model.weight([mov])
test_eq(w, model.i_weight(tensor([42])))

show_doc(EmbeddingDotBias.bias)

# The elements of `arr` are expected to be class names (which is why the model needs to be created with `EmbeddingDotBias.from_classes`)

mov = dls.classes['movieId'][42]
b = model.bias([mov])
test_eq(b, model.i_bias(tensor([42])))


# export
class EmbeddingNN(TabularModel):
    "Subclass `TabularModel` to create a NN suitable for collaborative filtering."
    @delegates(TabularModel.__init__)
    def __init__(self, emb_szs, layers, **kwargs):
        super().__init__(emb_szs=emb_szs, n_cont=0, out_sz=1, layers=layers, **kwargs)


show_doc(EmbeddingNN)

# `emb_szs` should be a list of two tuples, one for the users, one for the items, each tuple containing the number of users/items and the corresponding embedding size (the function `get_emb_sz` can give a good default). All the other arguments are passed to `TabularModel`.

emb_szs = get_emb_sz(dls.train_ds, {})
model = EmbeddingNN(emb_szs, [50], y_range=(0, 5)
                    ).to(x.device)
out = model(x)
assert (0 <= out).all() and (out <= 5).all()


# ## Create a `Learner`

# The following function lets us quickly create a `Learner` for collaborative filtering from the data.

# export
@delegates(Learner.__init__)
def collab_learner(dls, n_factors=50, use_nn=False, emb_szs=None, layers=None, config=None, y_range=None, loss_func=None, **kwargs):
    "Create a Learner for collaborative filtering on `dls`."
    emb_szs = get_emb_sz(dls, ifnone(emb_szs, {}))
    if loss_func is None:
        loss_func = MSELossFlat()
    if config is None:
        config = tabular_config()
    if y_range is not None:
        config['y_range'] = y_range
    if layers is None:
        layers = [n_factors]
    if use_nn:
        model = EmbeddingNN(emb_szs=emb_szs, layers=layers, **config)
    else:
        model = EmbeddingDotBias.from_classes(n_factors, dls.classes, y_range=y_range)
    return Learner(dls, model, loss_func=loss_func, **kwargs)


# If `use_nn=False`, the model used is an `EmbeddingDotBias` with `n_factors` and `y_range`. Otherwise, it's a `EmbeddingNN` for which you can pass `emb_szs` (will be inferred from the `dls` with `get_emb_sz` if you don't provide any), `layers` (defaults to `[n_factors]`) `y_range`, and a `config` that you can create with `tabular_config` to customize your model.
#
# `loss_func` will default to `MSELossFlat` and all the other arguments are passed to `Learner`.

learn = collab_learner(dls, y_range=(0, 5))

learn.fit_one_cycle(1)

# ## Export -

# hide
notebook2script()
