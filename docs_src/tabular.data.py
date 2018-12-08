
# coding: utf-8

# # Tabular data handling

# This module defines the main class to handle tabular data in the fastai library: [`TabularDataset`](/tabular.data.html#TabularDataset). As always, there is also a helper function to quickly get your data.
#
# To allow you to easily create a [`Learner`](/basic_train.html#Learner) for your data, it provides [`tabular_learner`](/tabular.data.html#tabular_learner).

from fastai.gen_doc.nbdoc import *
from fastai.tabular import *
from fastai import *


show_doc(TabularDataBunch, doc_string=False)


# The best way to quickly get your data in a [`DataBunch`](/basic_data.html#DataBunch) suitable for tabular data is to organize it in two (or three) dataframes. One for training, one for validation, and if you have it, one for testing. Here we are interested in a subsample of the [adult dataset](https://archive.ics.uci.edu/ml/datasets/adult).

path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path / 'adult.csv')
valid_idx = range(len(df) - 2000, len(df))
df.head()


cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
dep_var = '>=50k'


show_doc(TabularDataBunch.from_df, doc_string=False)


# Creates a [`DataBunch`](/basic_data.html#DataBunch) in `path` from `train_df`, `valid_df` and optionally `test_df`. The dependent variable is `dep_var`, while the categorical and continuous variables are in the `cat_names` columns and `cont_names` columns respectively. If `cont_names` is None then we assume all variables that aren't dependent or categorical are continuous. The [`TabularTransform`](/tabular.transform.html#TabularTransform) in `tfms` are applied to the dataframes as preprocessing, then the categories are replaced by their codes+1 (leaving 0 for `nan`) and the continuous variables are normalized. You can pass the `stats` to use for that step. If `log_output` is True, the dependant variable is replaced by its log.
#
# Note that the transforms should be passed as `Callable`: the actual initialization with `cat_names` and `cont_names` is done inside.

procs = [FillMissing, Categorify, Normalize]
data = TabularDataBunch.from_df(path, df, dep_var, valid_idx=valid_idx, procs=procs, cat_names=cat_names)


#  You can then easily create a [`Learner`](/basic_train.html#Learner) for this data with [`tabular_learner`](/tabular.data.html#tabular_learner).

show_doc(tabular_learner)


# `emb_szs` is a `dict` mapping categorical column names to embedding sizes; you only need to pass sizes for columns where you want to override the default behaviour of the model.

show_doc(TabularList)


# Basic class to create a list of inputs in `items` for tabular data. `cat_names` and `cont_names` are the names of the categorical and the continuous variables respectively. `processor` will be applied to the inputs or one will be created from the transforms in `procs`.

show_doc(TabularList.from_df)


show_doc(TabularList.get_emb_szs)


show_doc(TabularList.show_xys)


show_doc(TabularList.show_xyzs)


show_doc(TabularLine, doc_string=False)


# An object that will contain the encoded `cats`, the continuous variables `conts`, the `classes` and the `names` of the columns. This is the basic input for a dataset dealing with tabular data.

show_doc(TabularProcessor)


# Create a [`PreProcessor`](/data_block.html#PreProcessor) from `procs`.

# ## Undocumented Methods - Methods moved below this line will intentionally be hidden

show_doc(TabularProcessor.process_one)


show_doc(TabularList.new)


show_doc(TabularList.get)


show_doc(TabularProcessor.process)


show_doc(TabularList.reconstruct)


# ## New Methods - Please document or move to the undocumented section
