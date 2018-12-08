
# coding: utf-8

# # Tabular data preprocessing

from fastai.gen_doc.nbdoc import *
from fastai.tabular import *
from fastai import *


# ## Overview

# This package contains the basic class to define a transformation for preprocessing dataframes of tabular data, as well as basic [`TabularTransform`](/tabular.transform.html#TabularTransform). Preprocessing includes things like
# - replacing non-numerical variables by categories, then their ids,
# - filling missing values,
# - normalizing continuous variables.
#
# In all those steps we have to be careful to use the correspondance we decide on our training set (which id we give to each category, what is the value we put for missing data, or how the mean/std we use to normalize) on our validation or test set. To deal with this, we use a speciall class called [`TabularTransform`](/tabular.transform.html#TabularTransform).
#
# The data used in this document page is a subset of the [adult dataset](https://archive.ics.uci.edu/ml/datasets/adult). It gives a certain amount of data on individuals to train a model to predict wether their salary is greater than \$50k or not.

path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path / 'adult.csv')
train_df, valid_df = df.iloc[:800].copy(), df.iloc[800:1000].copy()
train_df.head()


# We see it contains numerical variables (like `age` or `education-num`) as well as categorical ones (like `workclass` or `relationship`). The original dataset is clean, but we removed a few values to give examples of dealing with missing variables.

cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
cont_names = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']


# ## Transforms for tabular data

show_doc(TabularProc, doc_string=False)


# Base class for creating transforms for dataframes with categorical variables `cat_names` and continuous variables `cont_names`. Note that any column not in one of those lists won't be touched.

show_doc(TabularProc.__call__)


# This simply calls `apply_test` if `test` or `apply_train` otherwise. Those functions apply the changes in place.

show_doc(TabularProc.apply_train, doc_string=False)


# Must be implemented by an inherited class with the desired transformation logic.

show_doc(TabularProc.apply_test, doc_string=False)


# If not implemented by an inherited class, defaults to calling `apply_train`.

# The following [`TabularTransform`](/tabular.transform.html#TabularTransform) are implemented in the fastai library. Note that the replacement from categories to codes as well as the normalization of continuous variables are automatically done in a [`TabularDataset`](/tabular.data.html#TabularDataset).

show_doc(Categorify, doc_string=False)


# Changes the categorical variables in `cat_names` in categories. Variables in `cont_names` aren't affected.

show_doc(Categorify.apply_train, doc_string=False)


# Transforms the variable in the `cat_names` columns in categories. The category codes are the unique values in these columns.

show_doc(Categorify.apply_test, doc_string=False)


# Transforms the variable in the `cat_names` columns in categories. The category codes are the ones used for the training set, new categories are replaced by NaN.

tfm = Categorify(cat_names, cont_names)
tfm(train_df)
tfm(valid_df, test=True)


# Since we haven't changed the categories by their codes, nothing visible has changed in the dataframe yet, but we can check that the variables are now categorical and view their corresponding codes.

train_df['workclass'].cat.categories


# The test set will be given the same category codes as the training set.

valid_df['workclass'].cat.categories


show_doc(FillMissing, doc_string=False)


# Transform that fills the missing values in `cont_names`. `cat_names` variables are left untouched (their missing value will be raplced by code 0 in the [`TabularDataset`](/tabular.data.html#TabularDataset)). [`fill_strategy`](#FillStrategy) is adopted to replace those nans and if `add_col` is True, whenever a column `c` has missing values, a column named `c_nan` is added and flags the line where the value was missing.

show_doc(FillMissing.apply_train, doc_string=False)


# Fills the missing values in the `cont_names` columns.

show_doc(FillMissing.apply_test, doc_string=False)


# Fills the missing values in the `cont_names` columns with the ones picked during train.

train_df[cont_names].head()


tfm = FillMissing(cat_names, cont_names)
tfm(train_df)
tfm(valid_df, test=True)
train_df[cont_names].head()


# Values issing in the `education-num` column are replaced by 10, which is the median of the column in `train_df`. Categorical variables are not changed, since `nan` is simply used as another category.

valid_df[cont_names].head()


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


show_doc(FillStrategy, alt_doc_string='Enum flag represents determines how `FillMissing` should handle missing/nan values', arg_comments={
    'MEDIAN': 'nans are replaced by the median value of the column',
    'COMMON': 'nans are replaced by the most common value of the column',
    'CONSTANT': 'nans are replaced by `fill_val`'
})


show_doc(Normalize, doc_string=False)


show_doc(Normalize.apply_train, doc_string=False)


# Computes the means and stds on the continuous variables of `df` then normalizes those columns.

show_doc(Normalize.apply_test, doc_string=False)


# Use the means and stds stored to normalize the continuous columns of `df`.

# ## Undocumented Methods - Methods moved below this line will intentionally be hidden

# ## New Methods - Please document or move to the undocumented section
