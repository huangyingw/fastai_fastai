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

from fastai.tabular.all import *

# # Rossmann

# ## Data preparation

# To create the feature-engineered train_clean and test_clean from the Kaggle competition data, run `rossman_data_clean.ipynb`. One important step that deals with time series is this:
#
# ```python
# add_datepart(train, "Date", drop=False)
# add_datepart(test, "Date", drop=False)
# ```

path = Config().data / 'rossmann'
train_df = pd.read_pickle(path / 'train_clean')

train_df.head().T

n = len(train_df)
n

# ### Experimenting with a sample

idx = np.random.permutation(range(n))[:2000]
idx.sort()
small_df = train_df.iloc[idx]
small_cont_vars = ['CompetitionDistance', 'Mean_Humidity']
small_cat_vars = ['Store', 'DayOfWeek', 'PromoInterval']
small_df = small_df[small_cat_vars + small_cont_vars + ['Sales']].reset_index(drop=True)

small_df.head()

small_df.iloc[1000:].head()

splits = [list(range(1000)), list(range(1000, 2000))]
to = TabularPandas(small_df.copy(), Categorify, cat_names=small_cat_vars, cont_names=small_cont_vars, splits=splits)

to.train.items.head()

to.valid.items.head()

to.classes['DayOfWeek']

splits = [list(range(1000)), list(range(1000, 2000))]
to = TabularPandas(small_df.copy(), FillMissing, cat_names=small_cat_vars, cont_names=small_cont_vars, splits=splits)

to.train.items[to.train.items['CompetitionDistance_na'] == True]

# ### Preparing full data set

train_df = pd.read_pickle(path / 'train_clean')
test_df = pd.read_pickle(path / 'test_clean')

len(train_df), len(test_df)

procs = [FillMissing, Categorify, Normalize]

# +
dep_var = 'Sales'
cat_names = ['Store', 'DayOfWeek', 'Year', 'Month', 'Day', 'StateHoliday', 'StoreType', 'Assortment',
             'PromoInterval', 'CompetitionOpenSinceYear', 'Promo2SinceYear', 'State', 'Week', 'Events', 'Promo_fw',
             'Promo_bw', 'StateHoliday_fw', 'StateHoliday_bw', 'SchoolHoliday_fw', 'SchoolHoliday_bw']

cont_names = ['CompetitionDistance', 'Max_TemperatureC', 'Mean_TemperatureC', 'Min_TemperatureC',
              'Max_Humidity', 'Mean_Humidity', 'Min_Humidity', 'Max_Wind_SpeedKm_h', 'Mean_Wind_SpeedKm_h',
              'CloudCover', 'trend', 'trend_DE', 'AfterStateHoliday', 'BeforeStateHoliday', 'Promo', 'SchoolHoliday']
# -

dep_var = 'Sales'
df = train_df[cat_names + cont_names + [dep_var, 'Date']].copy()

test_df['Date'].min(), test_df['Date'].max()

cut = train_df['Date'][(train_df['Date'] == train_df['Date'][len(test_df)])].index.max()
cut

splits = (list(range(cut, len(train_df))), list(range(cut)))

train_df[dep_var].head()

train_df[dep_var] = np.log(train_df[dep_var])
#train_df = train_df.iloc[:100000]

# +
#cut = 20000
# -

splits = (list(range(cut, len(train_df))), list(range(cut)))

# %time to = TabularPandas(train_df, procs, cat_names, cont_names, dep_var, y_block=TransformBlock(), splits=splits)

dls = to.dataloaders(bs=512, path=path)

dls.show_batch()

# ## Model

max_log_y = np.log(1.2) + np.max(train_df['Sales'])
y_range = (0, max_log_y)

dls.c = 1

learn = tabular_learner(dls, layers=[1000, 500], loss_func=MSELossFlat(),
                        config=tabular_config(ps=[0.001, 0.01], embed_p=0.04, y_range=y_range),
                        metrics=exp_rmspe)

learn.model

len(dls.train_ds.cont_names)

learn.lr_find()

learn.fit_one_cycle(5, 3e-3, wd=0.2)

# (10th place in the competition was 0.108)

learn.recorder.plot_loss(skip_start=1000)

# (10th place in the competition was 0.108)

# ## Inference on the test set

test_to = to.new(test_df)
test_to.process()

test_dls = test_to.dataloaders(bs=512, path=path, shuffle_train=False)

learn.metrics = []

tst_preds, _ = learn.get_preds(dl=test_dls.train)

np.exp(tst_preds.numpy()).T.shape

test_df["Sales"] = np.exp(tst_preds.numpy()).T[0]

test_df[["Id", "Sales"]] = test_df[["Id", "Sales"]].astype("int")
test_df[["Id", "Sales"]].to_csv("rossmann_submission.csv", index=False)

# This submission scored 3rd on the private leaderboard.
