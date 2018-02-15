
# coding: utf-8

# # Deep learning for Bulldozers

get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')


get_ipython().magic(u'matplotlib inline')

from fastai.imports import *
from fastai.torch_imports import *
from fastai.dataset import *
from fastai.learner import *
from fastai.structured import *
from fastai.column_data import *


# # Load in our data from last lesson

dep = 'SalePrice'
PATH = "data/bulldozers/"
df_raw = pd.read_feather('tmp/raw')
keep_cols = list(np.load('tmp/keep_cols.npy'))


df_raw.loc[df_raw.YearMade < 1950, 'YearMade'] = 1950
df_raw['age'] = df_raw.saleYear - df_raw.YearMade
df_raw = df_raw[keep_cols + ['age', dep]].copy()
df_indep = df_raw.drop(dep, axis=1)

n_valid = 12000
n_trn = len(df_raw) - n_valid


cat_flds = [n for n in df_indep.columns if df_raw[n].nunique() < n_trn / 50]
' '.join(cat_flds)


for o in ['saleElapsed', 'saleDayofyear', 'saleDay', 'age', 'YearMade']: cat_flds.remove(o)
[n for n in df_indep.drop(cat_flds, axis=1).columns if not is_numeric_dtype(df_raw[n])]


for n in cat_flds: df_raw[n] = df_raw[n].astype('category').cat.as_ordered()

cont_flds = [n for n in df_indep.columns if n not in cat_flds]
' '.join(cont_flds)


df_raw = df_raw[cat_flds + cont_flds + [dep]]
df, y, mapper = proc_df(df_raw, 'SalePrice', do_scale=True)

val_idx = list(range(n_trn, len(df)))


md = ColumnarModelData.from_data_frame(PATH, val_idx, df, y, cat_flds=cat_flds, bs=64)


df.head()


# # Model

def rmse(x, y): return math.sqrt(((x - y)**2).mean())


emb_c = {n: len(c.cat.categories) + 1 for n, c in df_raw[cat_flds].items()}
emb_c


emb_szs = [(c, min(50, (c + 1) // 2)) for _, c in emb_c.items()]
metrics = [rmse]


y_range = (0, np.max(y) * 1.2)


m = md.get_learner(emb_szs, len(cont_flds), 0.05, 1, [500, 250], [0.5, 0.05],
                   y_range=y_range, use_bn=True)


m.lr_find()


m.sched.plot(1300)


lr = 1e-3; wd = 1e-7


m.fit(lr, 2, wd, cycle_len=1, cycle_mult=2)


m.fit(lr, 2, wd, cycle_len=2, cycle_mult=2)


math.sqrt(0.0487)
