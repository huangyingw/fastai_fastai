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
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# # Tabular training
#
# > How to use the tabular application in fastai

# To illustrate the tabular application, we will use the example of the [Adult dataset](https://archive.ics.uci.edu/ml/datasets/Adult) where we have to predict if a person is earning more or less than $50k per year using some general data.


# We can download a sample of this dataset with the usual `untar_data` command:

path = untar_data(URLs.ADULT_SAMPLE)
path.ls()

# Then we can have a look at how the data is structured:

df = pd.read_csv(path / 'adult.csv')
df.head()

# Some of the columns are continuous (like age) and we will treat them as float numbers we can feed our model directly. Others are categorical (like workclass or education) and we will convert them to a unique index that we will feed to embedding layers. We can specify our categorical and continuous column names, as well as the name of the dependent variable in `TabularDataLoaders` factory methods:

dls = TabularDataLoaders.from_csv(path / 'adult.csv', path=path, y_names="salary",
                                  cat_names=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race'],
                                  cont_names=['age', 'fnlwgt', 'education-num'],
                                  procs=[Categorify, FillMissing, Normalize])

# The last part is the list of pre-processors we apply to our data:
#
# - `Categorify` is going to take every categorical variable and make a map from integer to unique categories, then replace the values by the corresponding index.
# - `FillMissing` will fill the missing values in the continuous variables by the median of existing values (you can choose a specific value if you prefer)
# - `Normalize` will normalize the continuous variables (substract the mean and divide by the std)
#
#

# To further expose what's going on below the surface, let's rewrite this utilizing `fastai`'s `TabularPandas` class. We will need to make one adjustment, which is defining how we want to split our data. By default the factory method above used a random 80/20 split, so we will do the same:

splits = RandomSplitter(valid_pct=0.2)(range_of(df))

to = TabularPandas(df, procs=[Categorify, FillMissing, Normalize],
                   cat_names=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race'],
                   cont_names=['age', 'fnlwgt', 'education-num'],
                   y_names='salary',
                   splits=splits)

# Once we build our `TabularPandas` object, our data is completely preprocessed as seen below:

to.xs.iloc[:2]

# Now we can build our `DataLoaders` again:

dls = to.dataloaders(bs=64)

# > Later we will explore why using `TabularPandas` to preprocess will be valuable.

# The `show_batch` method works like for every other application:

dls.show_batch()

# We can define a model using the `tabular_learner` method. When we define our model, `fastai` will try to infer the loss function based on our `y_names` earlier.
#
# **Note**: Sometimes with tabular data, your `y`'s may be encoded (such as 0 and 1). In such a case you should explicitly pass `y_block = CategoryBlock` in your constructor so `fastai` won't presume you are doing regression.

learn = tabular_learner(dls, metrics=accuracy)

# And we can train that model with the `fit_one_cycle` method (the `fine_tune` method won't be useful here since we don't have a pretrained model).

learn.fit_one_cycle(1)

# We can then have a look at some predictions:

learn.show_results()

# Or use the predict method on a row:

row, clas, probs = learn.predict(df.iloc[0])

row.show()

clas, probs

# To get prediction on a new dataframe, you can use the `test_dl` method of the `DataLoaders`. That dataframe does not need to have the dependent variable in its column.

test_df = df.copy()
test_df.drop(['salary'], axis=1, inplace=True)
dl = learn.dls.test_dl(test_df)

# Then `Learner.get_preds` will give you the predictions:

learn.get_preds(dl=dl)

# > Note: Since machine learning models can't magically understand categories it was never trained on, the data should reflect this. If there are different missing values in your test data you should address this before training

# ## `fastai` with Other Libraries
#
# As mentioned earlier, `TabularPandas` is a powerful and easy preprocessing tool for tabular data. Integration with libraries such as Random Forests and XGBoost requires only one extra step, that the `.dataloaders` call did for us. Let's look at our `to` again. It's values are stored in a `DataFrame` like object, where we can extract the `cats`, `conts,` `xs` and `ys` if we want to:

to.xs[:3]

# Now that everything is encoded, you can then send this off to XGBoost or Random Forests by extracting the train and validation sets and their values:

X_train, y_train = to.train.xs, to.train.ys.values.ravel()
X_test, y_test = to.valid.xs, to.valid.ys.values.ravel()

# And now we can directly send this in!
