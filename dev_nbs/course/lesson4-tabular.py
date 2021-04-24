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

# # Tabular models

from fastai.tabular.all import *

# Tabular data should be in a Pandas `DataFrame`.

path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path / 'adult.csv')

dep_var = 'salary'
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
cont_names = ['age', 'fnlwgt', 'education-num']
procs = [Categorify, FillMissing, Normalize]

# +
#test = TabularList.from_df(df.iloc[800:1000].copy(), path=path, cat_names=cat_names, cont_names=cont_names)
# -

splits = IndexSplitter(list(range(800, 1000)))(range_of(df))

# +
#splits = (L(splits[0], use_list=True), L(splits[1], use_list=True))
# -

to = TabularPandas(df, procs, cat_names, cont_names, y_names="salary", splits=splits)

dls = to.dataloaders(bs=64)

dls.show_batch()

learn = tabular_learner(dls, layers=[200, 100], metrics=accuracy)

learn.fit(1, 1e-2)

# ## Inference -> To do

row = df.iloc[0]

learn.predict(row)
