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
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter, _SingleProcessDataLoaderIter, _DatasetKind
from nbdev.export import notebook2script
from nbdev.showdoc import *
from fastai.data.all import *
from fastai.torch_basics import *
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# +
# default_exp tabular.core
# -

# export

# hide

# export
pd.set_option('mode.chained_assignment', 'raise')


# # Tabular core
#
# > Basic function to preprocess tabular data before assembling it in a `DataLoaders`.

# ## Initial preprocessing

# export
def make_date(df, date_field):
    "Make sure `df[date_field]` is of the right date type."
    field_dtype = df[date_field].dtype
    if isinstance(field_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        field_dtype = np.datetime64
    if not np.issubdtype(field_dtype, np.datetime64):
        df[date_field] = pd.to_datetime(df[date_field], infer_datetime_format=True)


df = pd.DataFrame({'date': ['2019-12-04', '2019-11-29', '2019-11-15', '2019-10-24']})
make_date(df, 'date')
test_eq(df['date'].dtype, np.dtype('datetime64[ns]'))


# export
def add_datepart(df, field_name, prefix=None, drop=True, time=False):
    "Helper function that adds columns relevant to a date in the column `field_name` of `df`."
    make_date(df, field_name)
    field = df[field_name]
    prefix = ifnone(prefix, re.sub('[Dd]ate$', '', field_name))
    attr = ['Year', 'Month', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start',
            'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time:
        attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr:
        df[prefix + n] = getattr(field.dt, n.lower())
    # Pandas removed `dt.week` in v1.1.10
    week = field.dt.isocalendar().week if hasattr(field.dt, 'isocalendar') else field.dt.week
    df.insert(3, prefix + 'Week', week)
    mask = ~field.isna()
    df[prefix + 'Elapsed'] = np.where(mask, field.values.astype(np.int64) // 10 ** 9, None)
    if drop:
        df.drop(field_name, axis=1, inplace=True)
    return df


df = pd.DataFrame({'date': ['2019-12-04', None, '2019-11-15', '2019-10-24']})
df = add_datepart(df, 'date')
test_eq(df.columns, ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start',
                     'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start', 'Elapsed'])
test_eq(df[df.Elapsed.isna()].shape, (1, 13))
df.head()


# export
def _get_elapsed(df, field_names, date_field, base_field, prefix):
    for f in field_names:
        day1 = np.timedelta64(1, 'D')
        last_date, last_base, res = np.datetime64(), None, []
        for b, v, d in zip(df[base_field].values, df[f].values, df[date_field].values):
            if last_base is None or b != last_base:
                last_date, last_base = np.datetime64(), b
            if v:
                last_date = d
            res.append(((d - last_date).astype('timedelta64[D]') / day1))
        df[prefix + f] = res
    return df


# export
def add_elapsed_times(df, field_names, date_field, base_field):
    "Add in `df` for each event in `field_names` the elapsed time according to `date_field` grouped by `base_field`"
    field_names = list(L(field_names))
    # Make sure date_field is a date and base_field a bool
    df[field_names] = df[field_names].astype('bool')
    make_date(df, date_field)

    work_df = df[field_names + [date_field, base_field]]
    work_df = work_df.sort_values([base_field, date_field])
    work_df = _get_elapsed(work_df, field_names, date_field, base_field, 'After')
    work_df = work_df.sort_values([base_field, date_field], ascending=[True, False])
    work_df = _get_elapsed(work_df, field_names, date_field, base_field, 'Before')

    for a in ['After' + f for f in field_names] + ['Before' + f for f in field_names]:
        work_df[a] = work_df[a].fillna(0).astype(int)

    for a, s in zip([True, False], ['_bw', '_fw']):
        work_df = work_df.set_index(date_field)
        tmp = (work_df[[base_field] + field_names].sort_index(ascending=a)
               .groupby(base_field).rolling(7, min_periods=1).sum())
        tmp.drop(base_field, 1, inplace=True)
        tmp.reset_index(inplace=True)
        work_df.reset_index(inplace=True)
        work_df = work_df.merge(tmp, 'left', [date_field, base_field], suffixes=['', s])
    work_df.drop(field_names, 1, inplace=True)
    return df.merge(work_df, 'left', [date_field, base_field])


df = pd.DataFrame({'date': ['2019-12-04', '2019-11-29', '2019-11-15', '2019-10-24'],
                   'event': [False, True, False, True], 'base': [1, 1, 2, 2]})
df = add_elapsed_times(df, ['event'], 'date', 'base')
df


# export
def cont_cat_split(df, max_card=20, dep_var=None):
    "Helper function that returns column names of cont and cat variables from given `df`."
    cont_names, cat_names = [], []
    for label in df:
        if label in L(dep_var):
            continue
        if df[label].dtype == int and df[label].unique().shape[0] > max_card or df[label].dtype == float:
            cont_names.append(label)
        else:
            cat_names.append(label)
    return cont_names, cat_names


# +
df = pd.DataFrame({'cat1': [1, 2, 3, 4], 'cont1': [1., 2., 3., 2.], 'cat2': ['a', 'b', 'b', 'a'],
                   'y1': [1, 0, 1, 0], 'y2': [1, 1, 1, 0]})

# Test all columns
cont, cat = cont_cat_split(df)
test_eq((cont, cat), (['cont1'], ['cat1', 'cat2', 'y1', 'y2']))

# Test exclusion of dependent variable
cont, cat = cont_cat_split(df, dep_var='y1')
test_eq((cont, cat), (['cont1'], ['cat1', 'cat2', 'y2']))

# Test exclusion of multi-label dependent variables
cont, cat = cont_cat_split(df, dep_var=['y1', 'y2'])
test_eq((cont, cat), (['cont1'], ['cat1', 'cat2']))

# Test maximal cardinality bound for int variable
cont, cat = cont_cat_split(df, max_card=2, dep_var=['y1', 'y2'])
test_eq((cont, cat), (['cat1', 'cont1'], ['cat2']))


# -

# export
def df_shrink_dtypes(df, skip=[], obj2cat=True, int2uint=False):
    "Return any possible smaller data types for DataFrame columns. Allows `object`->`category`, `int`->`uint`, and exclusion."

    # 1: Build column filter and typemap
    excl_types, skip = {'category', 'datetime64[ns]', 'bool'}, set(skip)

    typemap = {'int': [(np.dtype(x), np.iinfo(x).min, np.iinfo(x).max) for x in (np.int8, np.int16, np.int32, np.int64)],
               'uint': [(np.dtype(x), np.iinfo(x).min, np.iinfo(x).max) for x in (np.uint8, np.uint16, np.uint32, np.uint64)],
               'float': [(np.dtype(x), np.finfo(x).min, np.finfo(x).max) for x in (np.float32, np.float64, np.longdouble)]
               }
    if obj2cat:
        typemap['object'] = 'category'  # User wants to categorify dtype('Object'), which may not always save space
    else:
        excl_types.add('object')

    new_dtypes = {}
    def exclude(dt): return dt[1].name not in excl_types and dt[0] not in skip

    for c, old_t in filter(exclude, df.dtypes.items()):
        t = next((v for k, v in typemap.items() if old_t.name.startswith(k)), None)

        if isinstance(t, list):  # Find the smallest type that fits
            if int2uint and t == typemap['int'] and df[c].min() >= 0:
                t = typemap['uint']
            new_t = next((r[0] for r in t if r[1] <= df[c].min() and r[2] >= df[c].max()), None)
            if new_t and new_t == old_t:
                new_t = None
        else:
            new_t = t if isinstance(t, str) else None

        if new_t:
            new_dtypes[c] = new_t
    return new_dtypes


show_doc(df_shrink_dtypes, title_level=3)

# +
df = pd.DataFrame({'i': [-100, 0, 100], 'f': [-100.0, 0.0, 100.0], 'e': [True, False, True],
                   'date': ['2019-12-04', '2019-11-29', '2019-11-15', ]})
dt = df_shrink_dtypes(df)
test_eq(df['i'].dtype, 'int64')
test_eq(dt['i'], 'int8')

test_eq(df['f'].dtype, 'float64')
test_eq(dt['f'], 'float32')

# Default ignore 'object' and 'boolean' columns
test_eq(df['date'].dtype, 'object')
test_eq(dt['date'], 'category')

# Test categorifying 'object' type
dt2 = df_shrink_dtypes(df, obj2cat=False)
test_eq('date' not in dt2, True)


# -

# export
def df_shrink(df, skip=[], obj2cat=True, int2uint=False):
    "Reduce DataFrame memory usage, by casting to smaller types returned by `df_shrink_dtypes()`."
    dt = df_shrink_dtypes(df, skip, obj2cat=obj2cat, int2uint=int2uint)
    return df.astype(dt)


show_doc(df_shrink, title_level=3)

# `df_shrink(df)` attempts to make a DataFrame uses less memory, by fit numeric columns into smallest datatypes.  In addition:
#
#  * `boolean`, `category`, `datetime64[ns]` dtype columns are ignored.
#  * 'object' type columns are categorified, which can save a lot of memory in large dataset.  It can be turned off by `obj2cat=False`.
#  * `int2uint=True`, to fit `int` types to `uint` types, if all data in the column is >= 0.
#  * columns can be excluded by name using `excl_cols=['col1','col2']`.
#
# To get only new column data types without actually casting a DataFrame,
# use `df_shrink_dtypes()` with all the same parameters for `df_shrink()`.

# +
df = pd.DataFrame({'i': [-100, 0, 100], 'f': [-100.0, 0.0, 100.0], 'u': [0, 10, 254],
                   'date': ['2019-12-04', '2019-11-29', '2019-11-15']})
df2 = df_shrink(df, skip=['date'])

test_eq(df['i'].dtype == 'int64' and df2['i'].dtype == 'int8', True)
test_eq(df['f'].dtype == 'float64' and df2['f'].dtype == 'float32', True)
test_eq(df['u'].dtype == 'int64' and df2['u'].dtype == 'int16', True)
test_eq(df2['date'].dtype, 'object')

test_eq(df2.memory_usage().sum() < df.memory_usage().sum(), True)

# Test int => uint (when col.min() >= 0)
df3 = df_shrink(df, int2uint=True)
test_eq(df3['u'].dtype, 'uint8')  # int64 -> uint8 instead of int16

# Test excluding columns
df4 = df_shrink(df, skip=['i', 'u'])
test_eq(df['i'].dtype, df4['i'].dtype)
test_eq(df4['u'].dtype, 'int64')
# -

# Here's an example using the `ADULT_SAMPLE` dataset:

path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path / 'adult.csv')
new_df = df_shrink(df, int2uint=True)
print(f"Memory usage: {df.memory_usage().sum()} --> {new_df.memory_usage().sum()}")


# ## Tabular -

# export
class _TabIloc:
    "Get/set rows by iloc and cols by name"

    def __init__(self, to): self.to = to

    def __getitem__(self, idxs):
        df = self.to.items
        if isinstance(idxs, tuple):
            rows, cols = idxs
            cols = df.columns.isin(cols) if is_listy(cols) else df.columns.get_loc(cols)
        else:
            rows, cols = idxs, slice(None)
        return self.to.new(df.iloc[rows, cols])


# +
# export
class Tabular(CollBase, GetAttr, FilteredBase):
    "A `DataFrame` wrapper that knows which cols are cont/cat/y, and returns rows in `__getitem__`"
    _default, with_cont = 'procs', True

    def __init__(self, df, procs=None, cat_names=None, cont_names=None, y_names=None, y_block=None, splits=None,
                 do_setup=True, device=None, inplace=False, reduce_memory=True):
        if inplace and splits is not None and pd.options.mode.chained_assignment is not None:
            warn("Using inplace with splits will trigger a pandas error. Set `pd.options.mode.chained_assignment=None` to avoid it.")
        if not inplace:
            df = df.copy()
        if reduce_memory:
            df = df_shrink(df)
        if splits is not None:
            df = df.iloc[sum(splits, [])]
        self.dataloaders = delegates(self._dl_type.__init__)(self.dataloaders)
        super().__init__(df)

        self.y_names, self.device = L(y_names), device
        if y_block is None and self.y_names:
            # Make ys categorical if they're not numeric
            ys = df[self.y_names]
            if len(ys.select_dtypes(include='number').columns) != len(ys.columns):
                y_block = CategoryBlock()
            else:
                y_block = RegressionBlock()
        if y_block is not None and do_setup:
            if callable(y_block):
                y_block = y_block()
            procs = L(procs) + y_block.type_tfms
        self.cat_names, self.cont_names, self.procs = L(cat_names), L(cont_names), Pipeline(procs)
        self.split = len(df) if splits is None else len(splits[0])
        if do_setup:
            self.setup()

    def new(self, df):
        return type(self)(df, do_setup=False, reduce_memory=False, y_block=TransformBlock(),
                          **attrdict(self, 'procs', 'cat_names', 'cont_names', 'y_names', 'device'))

    def subset(self, i): return self.new(self.items[slice(0, self.split) if i == 0 else slice(self.split, len(self))])
    def copy(self): self.items = self.items.copy()
    return self
    def decode(self): return self.procs.decode(self)
    def decode_row(self, row): return self.new(pd.DataFrame(row).T).decode().items.iloc[0]
    def show(self, max_n=10, **kwargs): display_df(self.new(self.all_cols[:max_n]).decode().items)
    def setup(self): self.procs.setup(self)
    def process(self): self.procs(self)
    def loc(self): return self.items.loc
    def iloc(self): return _TabIloc(self)
    def targ(self): return self.items[self.y_names]
    def x_names(self): return self.cat_names + self.cont_names
    def n_subsets(self): return 2
    def y(self): return self[self.y_names[0]]
    def new_empty(self): return self.new(pd.DataFrame({}, columns=self.items.columns))

    def to_device(self, d=None):
        self.device = d
        return self

    def all_col_names(self):
        ys = [n for n in self.y_names if n in self.items.columns]
        return self.x_names + self.y_names if len(ys) == len(self.y_names) else self.x_names


properties(Tabular, 'loc', 'iloc', 'targ', 'all_col_names', 'n_subsets', 'x_names', 'y')


# -

# * `df`: A `DataFrame` of your data
# * `cat_names`: Your categorical `x` variables
# * `cont_names`: Your continuous `x` variables
# * `y_names`: Your dependent `y` variables
#   * Note: Mixed y's such as Regression and Classification is not currently supported, however multiple regression or classification outputs is
# * `y_block`: How to sub-categorize the type of `y_names` (`CategoryBlock` or `RegressionBlock`)
# * `splits`: How to split your data
# * `do_setup`: A parameter for if `Tabular` will run the data through the `procs` upon initialization
# * `device`: `cuda` or `cpu`
# * `inplace`: If `True`, `Tabular` will not keep a separate copy of your original `DataFrame` in memory. You should ensure `pd.options.mode.chained_assignment` is `None` before setting this
# * `reduce_memory`: `fastai` will attempt to reduce the overall memory usage by the inputted `DataFrame` with `df_shrink`

# export
class TabularPandas(Tabular):
    "A `Tabular` object with transforms"

    def transform(self, cols, f, all_col=True):
        if not all_col:
            cols = [c for c in cols if c in self.items.columns]
        if len(cols) > 0:
            self[cols] = self[cols].transform(f)


# +
# export
def _add_prop(cls, nm):
    @property
    def f(o): return o[list(getattr(o, nm + '_names'))]
    @f.setter
    def fset(o, v): o[getattr(o, nm + '_names')] = v
    setattr(cls, nm + 's', f)
    setattr(cls, nm + 's', fset)


_add_prop(Tabular, 'cat')
_add_prop(Tabular, 'cont')
_add_prop(Tabular, 'y')
_add_prop(Tabular, 'x')
_add_prop(Tabular, 'all_col')
# -

df = pd.DataFrame({'a': [0, 1, 2, 0, 2], 'b': [0, 0, 0, 0, 1]})
to = TabularPandas(df, cat_names='a')
t = pickle.loads(pickle.dumps(to))
test_eq(t.items, to.items)
test_eq(to.all_cols, to[['a']])


# export
class TabularProc(InplaceTransform):
    "Base class to write a non-lazy tabular processor for dataframes"

    def setup(self, items=None, train_setup=False):  # TODO: properly deal with train_setup
        super().setup(getattr(items, 'train', items), train_setup=False)
        # Procs are called as soon as data is available
        return self(items.items if isinstance(items, Datasets) else items)

    @property
    def name(self): return f"{super().name} -- {getattr(self,'__stored_args__',{})}"


# export
def _apply_cats(voc, add, c):
    if not is_categorical_dtype(c):
        return pd.Categorical(c, categories=voc[c.name][add:]).codes + add
    return c.cat.codes + add  # if is_categorical_dtype(c) else c.map(voc[c.name].o2i)


def _decode_cats(voc, c): return c.map(dict(enumerate(voc[c.name].items)))


# export
class Categorify(TabularProc):
    "Transform the categorical variables to something similar to `pd.Categorical`"
    order = 1

    def setups(self, to):
        store_attr(classes={n: CategoryMap(to.iloc[:, n].items, add_na=(n in to.cat_names)) for n in to.cat_names})

    def encodes(self, to): to.transform(to.cat_names, partial(_apply_cats, self.classes, 1))
    def decodes(self, to): to.transform(to.cat_names, partial(_decode_cats, self.classes))
    def __getitem__(self, k): return self.classes[k]


# +
# export
@Categorize
def setups(self, to: Tabular):
    if len(to.y_names) > 0:
        if self.vocab is None:
            self.vocab = CategoryMap(getattr(to, 'train', to).iloc[:, to.y_names[0]].items, strict=True)
        else:
            self.vocab = CategoryMap(self.vocab, sort=False, add_na=self.add_na)
        self.c = len(self.vocab)
    return self(to)


@Categorize
def encodes(self, to: Tabular):
    to.transform(to.y_names, partial(_apply_cats, {n: self.vocab for n in to.y_names}, 0), all_col=False)
    return to


@Categorize
def decodes(self, to: Tabular):
    to.transform(to.y_names, partial(_decode_cats, {n: self.vocab for n in to.y_names}), all_col=False)
    return to


# -

show_doc(Categorify, title_level=3)

df = pd.DataFrame({'a': [0, 1, 2, 0, 2]})
to = TabularPandas(df, Categorify, 'a')
cat = to.procs.categorify
test_eq(cat['a'], ['#na#', 0, 1, 2])
test_eq(to['a'], [1, 2, 3, 1, 3])
to.show()

df1 = pd.DataFrame({'a': [1, 0, 3, -1, 2]})
to1 = to.new(df1)
to1.process()
# Values that weren't in the training df are sent to 0 (na)
test_eq(to1['a'], [2, 1, 0, 0, 3])
to2 = cat.decode(to1)
test_eq(to2['a'], [1, 0, '#na#', '#na#', 2])

# test with splits
cat = Categorify()
df = pd.DataFrame({'a': [0, 1, 2, 3, 2]})
to = TabularPandas(df, cat, 'a', splits=[[0, 1, 2], [3, 4]])
test_eq(cat['a'], ['#na#', 0, 1, 2])
test_eq(to['a'], [1, 2, 3, 0, 3])

df = pd.DataFrame({'a': pd.Categorical(['M', 'H', 'L', 'M'], categories=['H', 'M', 'L'], ordered=True)})
to = TabularPandas(df, Categorify, 'a')
cat = to.procs.categorify
test_eq(cat['a'], ['#na#', 'H', 'M', 'L'])
test_eq(to.items.a, [2, 1, 3, 2])
to2 = cat.decode(to)
test_eq(to2['a'], ['M', 'H', 'L', 'M'])

# test with targets
cat = Categorify()
df = pd.DataFrame({'a': [0, 1, 2, 3, 2], 'b': ['a', 'b', 'a', 'b', 'b']})
to = TabularPandas(df, cat, 'a', splits=[[0, 1, 2], [3, 4]], y_names='b')
test_eq(to.vocab, ['a', 'b'])
test_eq(to['b'], [0, 1, 0, 1, 1])
to2 = to.procs.decode(to)
test_eq(to2['b'], ['a', 'b', 'a', 'b', 'b'])

cat = Categorify()
df = pd.DataFrame({'a': [0, 1, 2, 3, 2], 'b': ['a', 'b', 'a', 'b', 'b']})
to = TabularPandas(df, cat, 'a', splits=[[0, 1, 2], [3, 4]], y_names='b')
test_eq(to.vocab, ['a', 'b'])
test_eq(to['b'], [0, 1, 0, 1, 1])
to2 = to.procs.decode(to)
test_eq(to2['b'], ['a', 'b', 'a', 'b', 'b'])

# test with targets and train
cat = Categorify()
df = pd.DataFrame({'a': [0, 1, 2, 3, 2], 'b': ['a', 'b', 'a', 'c', 'b']})
to = TabularPandas(df, cat, 'a', splits=[[0, 1, 2], [3, 4]], y_names='b')
test_eq(to.vocab, ['a', 'b'])


# +
# export
@Normalize
def setups(self, to: Tabular):
    store_attr(means=dict(getattr(to, 'train', to).conts.mean()),
               stds=dict(getattr(to, 'train', to).conts.std(ddof=0) + 1e-7))
    return self(to)


@Normalize
def encodes(self, to: Tabular):
    to.conts = (to.conts - self.means) / self.stds
    return to


@Normalize
def decodes(self, to: Tabular):
    to.conts = (to.conts * self.stds) + self.means
    return to


# -

norm = Normalize()
df = pd.DataFrame({'a': [0, 1, 2, 3, 4]})
to = TabularPandas(df, norm, cont_names='a')
x = np.array([0, 1, 2, 3, 4])
m, s = x.mean(), x.std()
test_eq(norm.means['a'], m)
test_close(norm.stds['a'], s)
test_close(to['a'].values, (x - m) / s)

df1 = pd.DataFrame({'a': [5, 6, 7]})
to1 = to.new(df1)
to1.process()
test_close(to1['a'].values, (np.array([5, 6, 7]) - m) / s)
to2 = norm.decode(to1)
test_close(to2['a'].values, [5, 6, 7])

norm = Normalize()
df = pd.DataFrame({'a': [0, 1, 2, 3, 4]})
to = TabularPandas(df, norm, cont_names='a', splits=[[0, 1, 2], [3, 4]])
x = np.array([0, 1, 2])
m, s = x.mean(), x.std()
test_eq(norm.means['a'], m)
test_close(norm.stds['a'], s)
test_close(to['a'].values, (np.array([0, 1, 2, 3, 4]) - m) / s)


# export
class FillStrategy:
    "Namespace containing the various filling strategies."
    def median(c, fill): return c.median()
    def constant(c, fill): return fill
    def mode(c, fill): return c.dropna().value_counts().idxmax()


# Currently, filling with the `median`, a `constant`, and the `mode` are supported.

# export
class FillMissing(TabularProc):
    "Fill the missing values in continuous columns."

    def __init__(self, fill_strategy=FillStrategy.median, add_col=True, fill_vals=None):
        if fill_vals is None:
            fill_vals = defaultdict(int)
        store_attr()

    def setups(self, dsets):
        missing = pd.isnull(dsets.conts).any()
        store_attr(na_dict={n: self.fill_strategy(dsets[n], self.fill_vals[n])
                            for n in missing[missing].keys()})
        self.fill_strategy = self.fill_strategy.__name__

    def encodes(self, to):
        missing = pd.isnull(to.conts)
        for n in missing.any()[missing.any()].keys():
            assert n in self.na_dict, f"nan values in `{n}` but not in setup training set"
        for n in self.na_dict.keys():
            to[n].fillna(self.na_dict[n], inplace=True)
            if self.add_col:
                to.loc[:, n + '_na'] = missing[n]
                if n + '_na' not in to.cat_names:
                    to.cat_names.append(n + '_na')


show_doc(FillMissing, title_level=3)

# +
fill1, fill2, fill3 = (FillMissing(fill_strategy=s)
                       for s in [FillStrategy.median, FillStrategy.constant, FillStrategy.mode])
df = pd.DataFrame({'a': [0, 1, np.nan, 1, 2, 3, 4]})
df1 = df.copy()
df2 = df.copy()
tos = (TabularPandas(df, fill1, cont_names='a'),
       TabularPandas(df1, fill2, cont_names='a'),
       TabularPandas(df2, fill3, cont_names='a'))
test_eq(fill1.na_dict, {'a': 1.5})
test_eq(fill2.na_dict, {'a': 0})
test_eq(fill3.na_dict, {'a': 1.0})

for t in tos:
    test_eq(t.cat_names, ['a_na'])

for to_, v in zip(tos, [1.5, 0., 1.]):
    test_eq(to_['a'].values, np.array([0, 1, v, 1, 2, 3, 4]))
    test_eq(to_['a_na'].values, np.array([0, 0, 1, 0, 0, 0, 0]))
# -

fill = FillMissing()
df = pd.DataFrame({'a': [0, 1, np.nan, 1, 2, 3, 4], 'b': [0, 1, 2, 3, 4, 5, 6]})
to = TabularPandas(df, fill, cont_names=['a', 'b'])
test_eq(fill.na_dict, {'a': 1.5})
test_eq(to.cat_names, ['a_na'])
test_eq(to['a'].values, np.array([0, 1, 1.5, 1, 2, 3, 4]))
test_eq(to['a_na'].values, np.array([0, 0, 1, 0, 0, 0, 0]))
test_eq(to['b'].values, np.array([0, 1, 2, 3, 4, 5, 6]))

# ## TabularPandas Pipelines -

# +
procs = [Normalize, Categorify, FillMissing, noop]
df = pd.DataFrame({'a': [0, 1, 2, 1, 1, 2, 0], 'b': [0, 1, np.nan, 1, 2, 3, 4]})
to = TabularPandas(df, procs, cat_names='a', cont_names='b')

# Test setup and apply on df_main
test_eq(to.cat_names, ['a', 'b_na'])
test_eq(to['a'], [1, 2, 3, 2, 2, 3, 1])
test_eq(to['b_na'], [1, 1, 2, 1, 1, 1, 1])
x = np.array([0, 1, 1.5, 1, 2, 3, 4])
m, s = x.mean(), x.std()
test_close(to['b'].values, (x - m) / s)
test_eq(to.classes, {'a': ['#na#', 0, 1, 2], 'b_na': ['#na#', False, True]})

# +
# Test apply on y_names
df = pd.DataFrame({'a': [0, 1, 2, 1, 1, 2, 0], 'b': [0, 1, np.nan, 1, 2, 3, 4], 'c': ['b', 'a', 'b', 'a', 'a', 'b', 'a']})
to = TabularPandas(df, procs, 'a', 'b', y_names='c')

test_eq(to.cat_names, ['a', 'b_na'])
test_eq(to['a'], [1, 2, 3, 2, 2, 3, 1])
test_eq(to['b_na'], [1, 1, 2, 1, 1, 1, 1])
test_eq(to['c'], [1, 0, 1, 0, 0, 1, 0])
x = np.array([0, 1, 1.5, 1, 2, 3, 4])
m, s = x.mean(), x.std()
test_close(to['b'].values, (x - m) / s)
test_eq(to.classes, {'a': ['#na#', 0, 1, 2], 'b_na': ['#na#', False, True]})
test_eq(to.vocab, ['a', 'b'])

# +
df = pd.DataFrame({'a': [0, 1, 2, 1, 1, 2, 0], 'b': [0, 1, np.nan, 1, 2, 3, 4], 'c': ['b', 'a', 'b', 'a', 'a', 'b', 'a']})
to = TabularPandas(df, procs, 'a', 'b', y_names='c')

test_eq(to.cat_names, ['a', 'b_na'])
test_eq(to['a'], [1, 2, 3, 2, 2, 3, 1])
test_eq(df.a.dtype, int)
test_eq(to['b_na'], [1, 1, 2, 1, 1, 1, 1])
test_eq(to['c'], [1, 0, 1, 0, 0, 1, 0])

# +
df = pd.DataFrame({'a': [0, 1, 2, 1, 1, 2, 0], 'b': [0, np.nan, 1, 1, 2, 3, 4], 'c': ['b', 'a', 'b', 'a', 'a', 'b', 'a']})
to = TabularPandas(df, procs, cat_names='a', cont_names='b', y_names='c', splits=[[0, 1, 4, 6], [2, 3, 5]])

test_eq(to.cat_names, ['a', 'b_na'])
test_eq(to['a'], [1, 2, 2, 1, 0, 2, 0])
test_eq(df.a.dtype, int)
test_eq(to['b_na'], [1, 2, 1, 1, 1, 1, 1])
test_eq(to['c'], [1, 0, 0, 0, 1, 0, 1])


# -

# export
def _maybe_expand(o): return o[:, None] if o.ndim == 1 else o


# export
class ReadTabBatch(ItemTransform):
    def __init__(self, to): self.to = to

    def encodes(self, to):
        if not to.with_cont:
            res = (tensor(to.cats).long(),)
        else:
            res = (tensor(to.cats).long(), tensor(to.conts).float())
        ys = [n for n in to.y_names if n in to.items.columns]
        if len(ys) == len(to.y_names):
            res = res + (tensor(to.targ),)
        if to.device is not None:
            res = to_device(res, to.device)
        return res

    def decodes(self, o):
        o = [_maybe_expand(o_) for o_ in to_np(o) if o_.size != 0]
        vals = np.concatenate(o, axis=1)
        try:
            df = pd.DataFrame(vals, columns=self.to.all_col_names)
        except:
            df = pd.DataFrame(vals, columns=self.to.x_names)
        to = self.to.new(df)
        return to


# export
@typedispatch
def show_batch(x: Tabular, y, its, max_n=10, ctxs=None):
    x.show()


_loaders = (_MultiProcessingDataLoaderIter, _SingleProcessDataLoaderIter)


# +
# export
@delegates()
class TabDataLoader(TfmdDL):
    "A transformed `DataLoader` for Tabular data"
    do_item = noops

    def __init__(self, dataset, bs=16, shuffle=False, after_batch=None, num_workers=0, **kwargs):
        if after_batch is None:
            after_batch = L(TransformBlock().batch_tfms) + ReadTabBatch(dataset)
        super().__init__(dataset, bs=bs, shuffle=shuffle, after_batch=after_batch, num_workers=num_workers, **kwargs)

    def create_batch(self, b): return self.dataset.iloc[b]


TabularPandas._dl_type = TabDataLoader
# -

# ## Integration example
#
# For a more in-depth explanation, see the [tabular tutorial](http://docs.fast.ai/tutorial.tabular)

path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path / 'adult.csv')
df_main, df_test = df.iloc[:10000].copy(), df.iloc[10000:].copy()
df_test.drop('salary', axis=1, inplace=True)
df_main.head()

cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
cont_names = ['age', 'fnlwgt', 'education-num']
procs = [Categorify, FillMissing, Normalize]
splits = RandomSplitter()(range_of(df_main))

to = TabularPandas(df_main, procs, cat_names, cont_names, y_names="salary", splits=splits)

dls = to.dataloaders()
dls.valid.show_batch()

to.show()

# We can decode any set of transformed data by calling `to.decode_row` with our raw data:

row = to.items.iloc[0]
to.decode_row(row)

to_tst = to.new(df_test)
to_tst.process()
to_tst.items.head()

tst_dl = dls.valid.new(to_tst)
tst_dl.show_batch()


# ## Other target types

# ### Multi-label categories

# #### one-hot encoded label

def _mock_multi_label(df):
    sal, sex, white = [], [], []
    for row in df.itertuples():
        sal.append(row.salary == '>=50k')
        sex.append(row.sex == ' Male')
        white.append(row.race == ' White')
    df['salary'] = np.array(sal)
    df['male'] = np.array(sex)
    df['white'] = np.array(white)
    return df


path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path / 'adult.csv')
df_main, df_test = df.iloc[:10000].copy(), df.iloc[10000:].copy()
df_main = _mock_multi_label(df_main)

df_main.head()


# +
# export
@EncodedMultiCategorize
def setups(self, to: Tabular):
    self.c = len(self.vocab)
    return self(to)


@EncodedMultiCategorize
def encodes(self, to: Tabular): return to


@EncodedMultiCategorize
def decodes(self, to: Tabular):
    to.transform(to.y_names, lambda c: c == 1)
    return to


# -

cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
cont_names = ['age', 'fnlwgt', 'education-num']
procs = [Categorify, FillMissing, Normalize]
splits = RandomSplitter()(range_of(df_main))
y_names = ["salary", "male", "white"]

# %time to = TabularPandas(df_main, procs, cat_names, cont_names, y_names=y_names, y_block=MultiCategoryBlock(encoded=True, vocab=y_names), splits=splits)

dls = to.dataloaders()
dls.valid.show_batch()


# #### Not one-hot encoded

def _mock_multi_label(df):
    targ = []
    for row in df.itertuples():
        labels = []
        if row.salary == '>=50k':
            labels.append('>50k')
        if row.sex == ' Male':
            labels.append('male')
        if row.race == ' White':
            labels.append('white')
        targ.append(' '.join(labels))
    df['target'] = np.array(targ)
    return df


path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path / 'adult.csv')
df_main, df_test = df.iloc[:10000].copy(), df.iloc[10000:].copy()
df_main = _mock_multi_label(df_main)

df_main.head()


# +
@MultiCategorize
def encodes(self, to: Tabular):
    #to.transform(to.y_names, partial(_apply_cats, {n: self.vocab for n in to.y_names}, 0))
    return to


@MultiCategorize
def decodes(self, to: Tabular):
    #to.transform(to.y_names, partial(_decode_cats, {n: self.vocab for n in to.y_names}))
    return to


# -

cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
cont_names = ['age', 'fnlwgt', 'education-num']
procs = [Categorify, FillMissing, Normalize]
splits = RandomSplitter()(range_of(df_main))

# %time to = TabularPandas(df_main, procs, cat_names, cont_names, y_names="target", y_block=MultiCategoryBlock(), splits=splits)

to.procs[2].vocab


# ### Regression

# +
# export
@RegressionSetup
def setups(self, to: Tabular):
    if self.c is not None:
        return
    self.c = len(to.y_names)
    return to


@RegressionSetup
def encodes(self, to: Tabular): return to


@RegressionSetup
def decodes(self, to: Tabular): return to


# -

path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path / 'adult.csv')
df_main, df_test = df.iloc[:10000].copy(), df.iloc[10000:].copy()
df_main = _mock_multi_label(df_main)

cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
cont_names = ['fnlwgt', 'education-num']
procs = [Categorify, FillMissing, Normalize]
splits = RandomSplitter()(range_of(df_main))

# %time to = TabularPandas(df_main, procs, cat_names, cont_names, y_names='age', splits=splits)

to.procs[-1].means

dls = to.dataloaders()
dls.valid.show_batch()


# ## Not being used now - for multi-modal

# +
class TensorTabular(fastuple):
    def get_ctxs(self, max_n=10, **kwargs):
        n_samples = min(self[0].shape[0], max_n)
        df = pd.DataFrame(index=range(n_samples))
        return [df.iloc[i] for i in range(n_samples)]

    def display(self, ctxs): display_df(pd.DataFrame(ctxs))


class TabularLine(pd.Series):
    "A line of a dataframe that knows how to show itself"

    def show(self, ctx=None, **kwargs): return self if ctx is None else ctx.append(self)


class ReadTabLine(ItemTransform):
    def __init__(self, proc): self.proc = proc

    def encodes(self, row):
        cats, conts = (o.map(row.__getitem__) for o in (self.proc.cat_names, self.proc.cont_names))
        return TensorTabular(tensor(cats).long(), tensor(conts).float())

    def decodes(self, o):
        to = TabularPandas(o, self.proc.cat_names, self.proc.cont_names, self.proc.y_names)
        to = self.proc.decode(to)
        return TabularLine(pd.Series({c: v for v, c in zip(to.items[0] + to.items[1], self.proc.cat_names + self.proc.cont_names)}))


class ReadTabTarget(ItemTransform):
    def __init__(self, proc): self.proc = proc
    def encodes(self, row): return row[self.proc.y_names].astype(np.int64)
    def decodes(self, o): return Category(self.proc.classes[self.proc.y_names][o])


# +
# tds = TfmdDS(to.items, tfms=[[ReadTabLine(proc)], ReadTabTarget(proc)])
# enc = tds[1]
# test_eq(enc[0][0], tensor([2,1]))
# test_close(enc[0][1], tensor([-0.628828]))
# test_eq(enc[1], 1)

# dec = tds.decode(enc)
# assert isinstance(dec[0], TabularLine)
# test_close(dec[0], pd.Series({'a': 1, 'b_na': False, 'b': 1}))
# test_eq(dec[1], 'a')

# test_stdout(lambda: print(show_at(tds, 1)), """a               1
# b_na        False
# b               1
# category        a
# dtype: object""")
# -

# ## Export -

# hide
notebook2script()
