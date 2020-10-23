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
from nbdev.export import notebook2script
from nbdev.showdoc import *
from fastai.tabular.core import *
from fastai.data.all import *
from fastai.torch_basics import *
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# +
# default_exp tabular.data
# -

# export

# hide


# # Tabular data
#
# > Helper functions to get data in a `DataLoaders` in the tabular application and higher class `TabularDataLoaders`

# The main class to get your data ready for model training is `TabularDataLoaders` and its factory methods. Checkout the [tabular tutorial](http://docs.fast.ai/tutorial.tabular) for examples of use.

# ## TabularDataLoaders -

# +
# export
class TabularDataLoaders(DataLoaders):
    "Basic wrapper around several `DataLoader`s with factory methods for tabular data"
    @classmethod
    @delegates(Tabular.dataloaders, but=["dl_type", "dl_kwargs"])
    def from_df(cls, df, path='.', procs=None, cat_names=None, cont_names=None, y_names=None, y_block=None,
                valid_idx=None, **kwargs):
        "Create from `df` in `path` using `procs`"
        if cat_names is None:
            cat_names = []
        if cont_names is None:
            cont_names = list(set(df) - set(L(cat_names)) - set(L(y_names)))
        splits = RandomSplitter()(df) if valid_idx is None else IndexSplitter(valid_idx)(df)
        to = TabularPandas(df, procs, cat_names, cont_names, y_names, splits=splits, y_block=y_block)
        return to.dataloaders(path=path, **kwargs)

    @classmethod
    def from_csv(cls, csv, skipinitialspace=True, **kwargs):
        "Create from `csv` file in `path` using `procs`"
        return cls.from_df(pd.read_csv(csv, skipinitialspace=skipinitialspace), **kwargs)

    @delegates(TabDataLoader.__init__)
    def test_dl(self, test_items, rm_type_tfms=None, process=True, **kwargs):
        to = self.train_ds.new(test_items)
        if process:
            to.process()
        return self.valid.new(to, **kwargs)


Tabular._dbunch_type = TabularDataLoaders
TabularDataLoaders.from_csv = delegates(to=TabularDataLoaders.from_df)(TabularDataLoaders.from_csv)
# -

# This class should not be used directly, one of the factory methods should be preferred instead. All those factory methods accept as arguments:
#
# - `cat_names`: the names of the categorical variables
# - `cont_names`: the names of the continuous variables
# - `y_names`: the names of the dependent variables
# - `y_block`: the `TransformBlock` to use for the target
# - `valid_idx`: the indices to use for the validation set (defaults to a random split otherwise)
# - `bs`: the batch size
# - `val_bs`: the batch size for the validation `DataLoader` (defaults to `bs`)
# - `shuffle_train`: if we shuffle the training `DataLoader` or not
# - `n`: overrides the numbers of elements in the dataset
# - `device`: the PyTorch device to use (defaults to `default_device()`)

show_doc(TabularDataLoaders.from_df)

# Let's have a look on an example with the adult dataset:

path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path / 'adult.csv', skipinitialspace=True)
df.head()

cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
cont_names = ['age', 'fnlwgt', 'education-num']
procs = [Categorify, FillMissing, Normalize]

# + tags=[]
dls = TabularDataLoaders.from_df(df, path, procs=procs, cat_names=cat_names, cont_names=cont_names,
                                 y_names="salary", valid_idx=list(range(800, 1000)), bs=64)
# -

dls.show_batch()

show_doc(TabularDataLoaders.from_csv)

cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
cont_names = ['age', 'fnlwgt', 'education-num']
procs = [Categorify, FillMissing, Normalize]
dls = TabularDataLoaders.from_csv(path / 'adult.csv', path=path, procs=procs, cat_names=cat_names, cont_names=cont_names,
                                  y_names="salary", valid_idx=list(range(800, 1000)), bs=64)

# External structured data files can contain unexpected spaces, e.g. after a comma. We can see that in the first row of adult.csv `"49, Private,101320, ..."`. Often trimming is needed. Pandas has a convenient parameter `skipinitialspace` that is exposed by `TabularDataLoaders.from_csv()`. Otherwise category labels use for inference later such as `workclass`:`Private` will be categorized wrongly to *0* or `"#na#"` if training label was read as `" Private"`. Let's test this feature.

# + tags=[]
test_data = {
    'age': [49],
    'workclass': ['Private'],
    'fnlwgt': [101320],
    'education': ['Assoc-acdm'],
    'education-num': [12.0],
    'marital-status': ['Married-civ-spouse'],
    'occupation': [''],
    'relationship': ['Wife'],
    'race': ['White'],
}
input = pd.DataFrame(test_data)
tdl = dls.test_dl(input)

test_ne(0, tdl.dataset.iloc[0]['workclass'])
# -

# ## Export -

# + tags=[]
# hide
notebook2script()
# -
