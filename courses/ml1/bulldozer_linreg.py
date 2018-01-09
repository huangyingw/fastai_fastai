
# coding: utf-8

# # A Linear Model for Bulldozers

get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')


get_ipython().magic(u'matplotlib inline')

from fastai.imports import *
from fastai.structured import *
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV


set_plot_sizes(12, 14, 16)


# ## Load in our data from last lesson

PATH = "data/bulldozers/"

df_raw = pd.read_feather('tmp/raw')


df_raw['age'] = df_raw.saleYear - df_raw.YearMade


df, y, nas, mapper = proc_df(df_raw, 'SalePrice', max_n_cat=10, do_scale=True)


def split_vals(a, n): return a[:n], a[n:]


n_valid = 12000
n_trn = len(df) - n_valid
y_train, y_valid = split_vals(y, n_trn)
raw_train, raw_valid = split_vals(df_raw, n_trn)


def rmse(x, y): return math.sqrt(((x - y)**2).mean())


# # Linear regression for Bulldozers

# ## Data scaling

df.describe().transpose()


X_train, X_valid = split_vals(df, n_trn)


m = LinearRegression().fit(X_train, y_train)
m.score(X_valid, y_valid)


m.score(X_train, y_train)


preds = m.predict(X_valid)


rmse(preds, y_valid)


plt.scatter(preds, y_valid, alpha=0.1, s=2)


# ## Feature selection from RF

keep_cols = list(np.load('tmp/keep_cols.npy'))
', '.join(keep_cols)


df_sub = df_raw[keep_cols + ['age', 'SalePrice']]


df, y, mapper, nas = proc_df(df_sub, 'SalePrice', max_n_cat=10, do_scale=True)


X_train, X_valid = split_vals(df, n_trn)


m = LinearRegression().fit(X_train, y_train)
m.score(X_valid, y_valid)


rmse(m.predict(X_valid), y_valid)


from operator import itemgetter


sorted(list(zip(X_valid.columns, m.coef_)), key=itemgetter(1))


m = LassoCV().fit(X_train, y_train)
m.score(X_valid, y_valid)


rmse(m.predict(X_valid), y_valid)


m.alpha_


coefs = sorted(list(zip(X_valid.columns, m.coef_)), key=itemgetter(1))
coefs


skip = [n for n, c in coefs if abs(c) < 0.01]


df.drop(skip, axis=1, inplace=True)

# for n,c in df.items():
#     if '_' not in n: df[n+'2'] = df[n]**2


X_train, X_valid = split_vals(df, n_trn)


m = LassoCV().fit(X_train, y_train)
m.score(X_valid, y_valid)


rmse(m.predict(X_valid), y_valid)


coefs = sorted(list(zip(X_valid.columns, m.coef_)), key=itemgetter(1))
coefs


np.savez(f'{PATH}tmp/regr_resid', m.predict(X_train), m.predict(X_valid))
