# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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
from torch.utils.dlpack import from_dlpack
from nbdev.showdoc import *
from fastai.tabular.core import *
from fastai.data.all import *
from fastai.torch_basics import *
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# export
try:
    import cudf
    import nvcategory
except:
    print("This requires rapids, see https://rapids.ai/ for installation details")

# hide


# +
# default_exp tabular.rapids
# -

# # Tabular with rapids
#
# > Basic functions to preprocess tabular data before assembling it in a `DataLoaders` on the GPU.

# export
@patch
def __array__(self: cudf.DataFrame): return self.pandas().__array__()


# export
class TabularGPU(Tabular):
    def transform(self, cols, f):
        for c in cols:
            self[c] = f(self[c])

    def __getattr__(self, k):
        if isinstance(self.items, cudf.DataFrame) and k in self.items.columns:
            return self.items[k]
        return super().__getattr__(k)


# ## TabularProcessors

# export
def _to_str(c): return c if c.dtype == "object" else c.astype("str")
def _remove_none(c):
    if None in c:
        c.remove(None)
    return c


# +
# export
@Categorify
def setups(self, to: TabularGPU):
    self.lbls = {n: nvcategory.from_strings(_to_str(to.iloc[:, n]).data).keys() for n in to.all_cat_names}
    self.classes = {n: CategoryMap(_remove_none(c.to_host()), add_na=(n in to.cat_names)) for n, c in self.lbls.items()}

@patch
def _apply_cats_gpu(self: Categorify, c):
    return cudf.Series(nvcategory.from_strings(_to_str(c).data).set_keys(self.lbls[c.name]).values()).add(add)

@Categorify
def encodes(self, to: TabularGPU):
    def _apply_cats_gpu(add, c):
        return cudf.Series(nvcategory.from_strings(_to_str(c).data).set_keys(self.lbls[c.name]).values()).add(add)
    to.transform(to.cat_names, partial(_apply_cats_gpu, 1))
    to.transform(L(to.cat_y), partial(_apply_cats_gpu, 0))


# -

df = cudf.from_pandas(pd.DataFrame({'a': [0, 1, 2, 0, 2]}))
to = TabularGPU(df, Categorify, 'a')
cat = to.procs.categorify
test_eq(list(cat['a']), ['#na#', '0', '1', '2'])
test_eq(to.a.to_array(), np.array([1, 2, 3, 1, 3]))
df1 = cudf.from_pandas(pd.DataFrame({'a': [1, 0, 3, -1, 2]}))
to1 = to.new(df1)
cat(to1)
# Values that weren't in the training df are sent to 0 (na)
test_eq(to1.a.to_array(), np.array([2, 1, 0, 0, 3]))

# Test decode
to2 = TabularPandas(to1.items.to_pandas(), None, 'a')
to2 = cat.decode(to2)
test_eq(to2.a, np.array(['1', '0', '#na#', '#na#', '2']))

df = cudf.from_pandas(pd.DataFrame({'a': [0, 1, 2, 3, 2]}))
to = TabularGPU(df, Categorify, 'a', splits=[[0, 1, 2], [3, 4]])
cat = to.procs.categorify
test_eq(list(cat['a']), ['#na#', '0', '1', '2'])
test_eq(to.a.to_array(), np.array([1, 2, 3, 0, 3]))


# +
# TODO Categorical (fails for now)
#df = cudf.from_pandas(pd.DataFrame({'a':pd.Categorical(['M','H','L','M'], categories=['H','M','L'], ordered=True)}))
#to = TabularGPU(df, Categorify, 'a')
#cat = to.procs.categorify
#test_eq(cat['a'].to_host(), ['H','M','L'])
#test_eq(df["a"].to_array(), [2,1,3,2])

# +
# export
@Normalize
def setups(self, to: TabularGPU):
    self.means = {n: to.iloc[:, n].mean() for n in to.cont_names}
    self.stds = {n: to.iloc[:, n].std(ddof=0) + 1e-7 for n in to.cont_names}

@Normalize
def encodes(self, to: TabularGPU):
    to.transform(to.cont_names, lambda c: (c - self.means[c.name]) / self.stds[c.name])


# +
df = cudf.from_pandas(pd.DataFrame({'a': [0, 1, 2, 3, 4]}))
to = TabularGPU(df, Normalize, cont_names='a')
norm = to.procs.normalize
x = np.array([0, 1, 2, 3, 4])
m, s = x.mean(), x.std()
test_eq(norm.means['a'], m)
test_close(norm.stds['a'], s)
test_close(to.a.to_array(), (x - m) / s)
df1 = cudf.from_pandas(pd.DataFrame({'a': [5, 6, 7]}))
to1 = to.new(df1)
norm(to1)
test_close(to1.a.to_array(), (np.array([5, 6, 7]) - m) / s)

to2 = TabularPandas(to1.items.to_pandas(), None, cont_names='a')
to2 = norm.decode(to2)
test_close(to2.a, [5, 6, 7])

# +
df = cudf.from_pandas(pd.DataFrame({'a': [0, 1, 2, 3, 4]}))
to = TabularGPU(df, Normalize, cont_names='a', splits=[[0, 1, 2], [3, 4]])
norm = to.procs.normalize

x = np.array([0, 1, 2])
m, s = x.mean(), x.std()
test_eq(norm.means, {'a': m})
test_close(norm.stds['a'], s)
test_close(to.a.to_array(), (np.array([0, 1, 2, 3, 4]) - m) / s)


# -

# export
@patch
def median(self: cudf.Series):
    "Get the median of `self`"
    col = self.dropna().reset_index(drop=True).sort_values()
    return col[len(col) // 2] if len(col) % 2 != 0 else (col[len(col) // 2] + col[len(col) // 2 - 1]) / 2


col = cudf.Series([0, 1, np.nan, 1, 2, 3, 4])
test_eq(col.median(), 1.5)
col = cudf.Series([np.nan, 1, np.nan, 1, 2, 3, 4])
test_eq(col.median(), 2)


# export
@patch
def idxmax(self: cudf.Series):
    "Return the index of the first occurrence of the max in `self`"
    return self.argsort(ascending=False).index[0]


# +
# export
@FillMissing
def setups(self, to: TabularGPU):
    self.na_dict = {}
    for n in to.cont_names:
        col = to.iloc[:, n]
        if col.isnull().any():
            self.na_dict[n] = self.fill_strategy(col, self.fill_vals[n])

@FillMissing
def encodes(self, to: TabularGPU):
    for n in to.cont_names:
        if n in self.na_dict:
            if self.add_col:
                to.items[n + '_na'] = to[n].isnull()
                if n + '_na' not in to.cat_names:
                    to.cat_names.append(n + '_na')
            to[n] = to[n].fillna(self.na_dict[n])
        elif df[n].isnull().any():
            raise Exception(f"nan values in `{n}` but not in setup training set")


# +
fill1, fill2, fill3 = (FillMissing(fill_strategy=s)
                       for s in [FillStrategy.median, FillStrategy.constant, FillStrategy.mode])
df = cudf.from_pandas(pd.DataFrame({'a': [0, 1, np.nan, 1, 2, 3, 4]}))
df1 = df.copy()
df2 = df.copy()
tos = TabularGPU(df, fill1, cont_names='a'), TabularGPU(df1, fill2, cont_names='a'), TabularGPU(df2, fill3, cont_names='a')

test_eq(fill1.na_dict, {'a': 1.5})
test_eq(fill2.na_dict, {'a': 0})
test_eq(fill3.na_dict, {'a': 1.0})

for t in tos:
    test_eq(t.cat_names, ['a_na'])

for to_, v in zip(tos, [1.5, 0., 1.]):
    test_eq(to_.a.to_array(), np.array([0, 1, v, 1, 2, 3, 4]))
    test_eq(to_.a_na.to_array(), np.array([0, 0, 1, 0, 0, 0, 0]))
# -

dfa = cudf.from_pandas(pd.DataFrame({'a': [np.nan, 0, np.nan]}))
tos = [t.new(o) for t, o in zip(tos, (dfa, dfa.copy(), dfa.copy()))]
for t in tos:
    t.process()
for to_, v in zip(tos, [1.5, 0., 1.]):
    test_eq(to_.a.to_array(), np.array([v, 0, v]))
    test_eq(to_.a_na.to_array(), np.array([1, 0, 1]))

# ## Tabular Pipelines -

# +
procs = [Normalize, Categorify, FillMissing, noop]
df = cudf.from_pandas(pd.DataFrame({'a': [0, 1, 2, 1, 1, 2, 0], 'b': [0, 1, np.nan, 1, 2, 3, 4]}))
to = TabularGPU(df, procs, cat_names='a', cont_names='b')

# Test setup and apply on df_trn
test_eq(to.a.to_array(), [1, 2, 3, 2, 2, 3, 1])
test_eq(to.b_na.to_array(), [1, 1, 2, 1, 1, 1, 1])
x = np.array([0, 1, 1.5, 1, 2, 3, 4])
m, s = x.mean(), x.std()
test_close(to.b.to_array(), (x - m) / s)
test_eq(to.procs.classes, {'a': ['#na#', '0', '1', '2'], 'b_na': ['#na#', 'False', 'True']})

# +
# Test apply on y_names
procs = [Normalize, Categorify, FillMissing, noop]
df = cudf.from_pandas(pd.DataFrame({'a': [0, 1, 2, 1, 1, 2, 0], 'b': [0, 1, np.nan, 1, 2, 3, 4], 'c': ['b', 'a', 'b', 'a', 'a', 'b', 'a']}))
to = TabularGPU(df, procs, cat_names='a', cont_names='b', y_names='c')

test_eq(to.cat_names, ['a', 'b_na'])
test_eq(to.a.to_array(), [1, 2, 3, 2, 2, 3, 1])
test_eq(to.b_na.to_array(), [1, 1, 2, 1, 1, 1, 1])
test_eq(to.c.to_array(), [1, 0, 1, 0, 0, 1, 0])
x = np.array([0, 1, 1.5, 1, 2, 3, 4])
m, s = x.mean(), x.std()
test_close(to.b.to_array(), (x - m) / s)
test_eq(to.procs.classes, {'a': ['#na#', '0', '1', '2'], 'b_na': ['#na#', 'False', 'True'], 'c': ['a', 'b']})

# +
procs = [Normalize, Categorify, FillMissing, noop]
df = cudf.from_pandas(pd.DataFrame({'a': [0, 1, 2, 1, 1, 2, 0], 'b': [0, 1, np.nan, 1, 2, 3, 4], 'c': ['b', 'a', 'b', 'a', 'a', 'b', 'a']}))
to = TabularGPU(df, procs, cat_names='a', cont_names='b', y_names='c')

test_eq(to.cat_names, ['a', 'b_na'])
test_eq(to.a.to_array(), [1, 2, 3, 2, 2, 3, 1])
test_eq(to.a.dtype, int)
test_eq(to.b_na.to_array(), [1, 1, 2, 1, 1, 1, 1])
test_eq(to.c.to_array(), [1, 0, 1, 0, 0, 1, 0])

# +
procs = [Normalize, Categorify, FillMissing, noop]
df = cudf.from_pandas(pd.DataFrame({'a': [0, 1, 2, 1, 1, 2, 0], 'b': [0, np.nan, 1, 1, 2, 3, 4], 'c': ['b', 'a', 'b', 'a', 'a', 'b', 'a']}))
to = TabularGPU(df, procs, cat_names='a', cont_names='b', y_names='c', splits=[[0, 1, 4, 6], [2, 3, 5]])

test_eq(to.cat_names, ['a', 'b_na'])
test_eq(to.a.to_array(), [1, 2, 2, 1, 0, 2, 0])
test_eq(to.a.dtype, int)
test_eq(to.b_na.to_array(), [1, 2, 1, 1, 1, 1, 1])
test_eq(to.c.to_array(), [1, 0, 0, 0, 1, 0, 1])

# +
# export

@ReadTabBatch
def encodes(self, to: TabularGPU):
    return from_dlpack(to.cats.to_dlpack()).long(), from_dlpack(to.conts.to_dlpack()).float(), from_dlpack(to.targ.to_dlpack()).long()


# -

# ## Integration example

path = untar_data(URLs.ADULT_SAMPLE)
df = cudf.from_pandas(pd.read_csv(path / 'adult.csv'))
df_trn, df_tst = df.iloc[:10000].copy(), df.iloc[10000:].copy()
df_trn.head()

# +
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
cont_names = ['age', 'fnlwgt', 'education-num']
procs = [Categorify, FillMissing, Normalize]

splits = RandomSplitter()(range_of(df_trn))
# -

# %time to = TabularGPU(df_trn, procs, splits=splits, cat_names=cat_names, cont_names=cont_names, y_names="salary")

splits = [list(range(len(splits[0]))), list(range(len(splits[0]), 10000))]
dsets = Datasets(to, splits=splits, tfms=[None])
dl = TabDataLoader(to.valid, bs=64, num_workers=0)

dl.show_batch()

# ## Export -

# hide
notebook2script()
