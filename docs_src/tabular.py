# coding: utf-8
# # Tabular data
from fastai.gen_doc.nbdoc import *
from fastai.tabular.models import *
# [`tabular`](/tabular.html#tabular) contains all the necessary classes to deal with tabular data, across two modules:
# - [`tabular.transform`](/tabular.transform.html#tabular.transform): defines the [`TabularTransform`](/tabular.transform.html#TabularTransform) class to help with preprocessing;
# - [`tabular.data`](/tabular.data.html#tabular.data): defines the [`TabularDataset`](/tabular.data.html#TabularDataset) that handles that data, as well as the methods to quickly get a [`TabularDataBunch`](/tabular.data.html#TabularDataBunch).
#
# To create a model, you'll need to use [`models.tabular`](/tabular.html#tabular). See below for an end-to-end example using all these modules.
# ## Preprocessing tabular data
# First, let's import everything we need for the tabular application.
from fastai.tabular import *
# Tabular data usually comes in the form of a delimited file (such as .csv) containing variables of different kinds: text/category, numbers, and perhaps some missing values. The example we'll work with in this section is a sample of the [adult dataset](https://archive.ics.uci.edu/ml/datasets/adult) which has some census information on individuals. We'll use it to train a model to predict whether salary is greater than \$50k or not.
path = untar_data(URLs.ADULT_SAMPLE)
path
df = pd.read_csv(path / 'adult.csv')
df.head()
# Here all the information that will form our input is in the 14 first columns, and the dependent variable is the last column. We will split our input between two types of variables: categorical and continuous.
# - Categorical variables will be replaced by a category - a unique id that identifies them - before they are passed through an embedding layer.
# - Continuous variables will be normalized and then directly fed to the model.
#
# Another thing we need to handle are the missing values: our model isn't going to like receiving NaNs so we should remove them in a smart way. All of this preprocessing is done by [`TabularTransform`](/tabular.transform.html#TabularTransform) objects and [`TabularDataset`](/tabular.data.html#TabularDataset).
#
# We can define a bunch of Transforms that will be applied to our variables. Here we transform all categorical variables into categories. We also replace missing values for continuous variables by the median column value and normalize those.
procs = [FillMissing, Categorify, Normalize]
# To split our data into training and validation sets, we use valid indexes
valid_idx = range(len(df) - 2000, len(df))
# Then let's manually split our variables into categorical and continuous variables (we can ignore the dependant variable at this stage). fastai will assume all variables that aren't dependent or categorical are continuous, unless we explicitly pass a list to the `cont_names` parameter when constructing our [`DataBunch`](/basic_data.html#DataBunch).
dep_var = '>=50k'
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
# Now we're ready to pass this information to [`TabularDataBunch.from_df`](/tabular.data.html#TabularDataBunch.from_df) to create the [`DataBunch`](/basic_data.html#DataBunch) that we'll use for training.
data = TabularDataBunch.from_df(path, df, dep_var, valid_idx=valid_idx, procs=procs, cat_names=cat_names)
print(data.train_ds.cont_names)  # `cont_names` defaults to: set(df)-set(cat_names)-{dep_var}
# We can grab a mini-batch of data and take a look (note that [`to_np`](/torch_core.html#to_np) here converts from pytorch tensor to numpy):
(cat_x, cont_x), y = next(iter(data.train_dl))
for o in (cat_x, cont_x, y): print(to_np(o[:5]))
# After being processed in [`TabularDataset`](/tabular.data.html#TabularDataset), the categorical variables are replaced by ids and the continuous variables are normalized. The codes corresponding to categorical variables are all put together, as are all the continuous variables.
# ## Defining a model
# Once we have our data ready in a [`DataBunch`](/basic_data.html#DataBunch), we just need to create a model to then define a [`Learner`](/basic_train.html#Learner) and start training. The fastai library has a flexible and powerful [`TabularModel`](/tabular.models.html#TabularModel) in [`models.tabular`](/tabular.html#tabular). To use that function, we just need to specify the embedding sizes for each of our categorical variables.
learn = tabular_learner(data, layers=[200, 100], emb_szs={'native-country': 10}, metrics=accuracy)
learn.fit_one_cycle(1, 1e-2)
# As usual, we can use the [`Learner.predict`](/basic_train.html#Learner.predict) method to get predictions. In this case, we need to pass the row of a dataframe that has the same names of categorical and continuous variables as our training or validation dataframe.
learn.predict(df.iloc[0])
