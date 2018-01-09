
# coding: utf-8

# # Structured and time series data

# This notebook contains an implementation of the third place result in the Rossman Kaggle competition as detailed in Guo/Berkhahn's [Entity Embeddings of Categorical Variables](https://arxiv.org/abs/1604.06737).
#
# The motivation behind exploring this architecture is it's relevance to
# real-world application. Most data used for decision making day-to-day in
# industry is structured and/or time-series data. Here we explore the
# end-to-end process of using neural networks with practical structured
# data problems.

get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u'reload_ext autoreload')
get_ipython().magic(u'autoreload 2')


from fastai.structured import *
from fastai.column_data import *
np.set_printoptions(threshold=50, edgeitems=20)

PATH = 'data/rossmann/'


# ## Create datasets

# In addition to the provided data, we will be using external datasets put together by participants in the Kaggle competition. You can download all of them [here](http://files.fast.ai/part2/lesson14/rossmann.tgz).
#
# For completeness, the implementation used to put them together is
# included below.

def concat_csvs(dirname):
    path = f'{PATH}{dirname}'
    filenames = glob.glob(f"{path}/*.csv")

    wrote_header = False
    with open(f"{path}.csv", "w") as outputfile:
        for filename in filenames:
            name = filename.split(".")[0]
            with open(filename) as f:
                line = f.readline()
                if not wrote_header:
                    wrote_header = True
                    outputfile.write("file," + line)
                for line in f:
                    outputfile.write(name + "," + line)
                outputfile.write("\n")


# concat_csvs('googletrend')
# concat_csvs('weather')


# Feature Space:
# * train: Training set provided by competition
# * store: List of stores
# * store_states: mapping of store to the German state they are in
# * List of German state names
# * googletrend: trend of certain google keywords over time, found by users to correlate well w/ given data
# * weather: weather
# * test: testing set

table_names = ['train', 'store', 'store_states', 'state_names',
               'googletrend', 'weather', 'test']


# We'll be using the popular data manipulation framework `pandas`. Among other things, pandas allows you to manipulate tables/data frames in python as one would in a database.
#
# We're going to go ahead and load all of our csv's as dataframes into the
# list `tables`.

tables = [pd.read_csv(f'{PATH}{fname}.csv', low_memory=False) for fname in table_names]


from IPython.display import HTML


# We can use `head()` to get a quick look at the contents of each table:
# * train: Contains store information on a daily basis, tracks things like sales, customers, whether that day was a holdiay, etc.
# * store: general info about the store including competition, etc.
# * store_states: maps store to state it is in
# * state_names: Maps state abbreviations to names
# * googletrend: trend data for particular week/state
# * weather: weather conditions for each state
# * test: Same as training table, w/o sales and customers
#

for t in tables:
    display(t.head())


# This is very representative of a typical industry dataset.
#
# The following returns summarized aggregate information to each table
# accross each field.

for t in tables:
    display(DataFrameSummary(t).summary())


# ## Data Cleaning / Feature Engineering

# As a structured data problem, we necessarily have to go through all the
# cleaning and feature engineering, even though we're using a neural
# network.

train, store, store_states, state_names, googletrend, weather, test = tables


len(train), len(test)


# We turn state Holidays to booleans, to make them more convenient for
# modeling. We can do calculations on pandas fields using notation very
# similar (often identical) to numpy.

train.StateHoliday = train.StateHoliday != '0'
test.StateHoliday = test.StateHoliday != '0'


# `join_df` is a function for joining tables on specific fields. By default, we'll be doing a left outer join of `right` on the `left` argument using the given fields for each table.
#
# Pandas does joins using the `merge` method. The `suffixes` argument
# describes the naming convention for duplicate fields. We've elected to
# leave the duplicate field names on the left untouched, and append a
# "\_y" to those on the right.

def join_df(left, right, left_on, right_on=None, suffix='_y'):
    if right_on is None:
        right_on = left_on
    return left.merge(right, how='left', left_on=left_on, right_on=right_on,
                      suffixes=("", suffix))


# Join weather/state names.

weather = join_df(weather, state_names, "file", "StateName")


# In pandas you can add new columns to a dataframe by simply defining it. We'll do this for googletrends by extracting dates and state names from the given data and adding those columns.
#
# We're also going to replace all instances of state name 'NI' to match
# the usage in the rest of the data: 'HB,NI'. This is a good opportunity
# to highlight pandas indexing. We can use `.loc[rows, cols]` to select a
# list of rows and a list of columns from the dataframe. In this case,
# we're selecting rows w/ statename 'NI' by using a boolean list
# `googletrend.State=='NI'` and selecting "State".

googletrend['Date'] = googletrend.week.str.split(' - ', expand=True)[0]
googletrend['State'] = googletrend.file.str.split('_', expand=True)[2]
googletrend.loc[googletrend.State == 'NI', "State"] = 'HB,NI'


# The following extracts particular date fields from a complete datetime for the purpose of constructing categoricals.
#
# You should *always* consider this feature extraction step when working
# with date-time. Without expanding your date-time into these additional
# fields, you can't capture any trend/cyclical behavior as a function of
# time at any of these granularities. We'll add to every table with a date
# field.

add_datepart(weather, "Date", drop=False)
add_datepart(googletrend, "Date", drop=False)
add_datepart(train, "Date", drop=False)
add_datepart(test, "Date", drop=False)


# The Google trends data has a special category for the whole of the US -
# we'll pull that out so we can use it explicitly.

trend_de = googletrend[googletrend.file == 'Rossmann_DE']


# Now we can outer join all of our data into a single dataframe. Recall that in outer joins everytime a value in the joining field on the left table does not have a corresponding value on the right table, the corresponding row in the new table has Null values for all right table fields. One way to check that all records are consistent and complete is to check for Null values post-join, as we do here.
#
# *Aside*: Why note just do an inner join?
# If you are assuming that all records are complete and match on the field
# you desire, an inner join will do the same thing as an outer join.
# However, in the event you are wrong or a mistake is made, an outer join
# followed by a null-check will catch it. (Comparing before/after # of
# rows for inner join is equivalent, but requires keeping track of
# before/after row #'s. Outer join is easier.)

store = join_df(store, store_states, "Store")
len(store[store.State.isnull()])


joined = join_df(train, store, "Store")
joined_test = join_df(test, store, "Store")
len(joined[joined.StoreType.isnull()]), len(
    joined_test[joined_test.StoreType.isnull()])


joined = join_df(joined, googletrend, ["State", "Year", "Week"])
joined_test = join_df(joined_test, googletrend, ["State", "Year", "Week"])
len(joined[joined.trend.isnull()]), len(
    joined_test[joined_test.trend.isnull()])


joined = joined.merge(trend_de, 'left', ["Year", "Week"], suffixes=('', '_DE'))
joined_test = joined_test.merge(
    trend_de, 'left', [
        "Year", "Week"], suffixes=(
            '', '_DE'))
len(joined[joined.trend_DE.isnull()]), len(
    joined_test[joined_test.trend_DE.isnull()])


joined = join_df(joined, weather, ["State", "Date"])
joined_test = join_df(joined_test, weather, ["State", "Date"])
len(joined[joined.Mean_TemperatureC.isnull()]), len(
    joined_test[joined_test.Mean_TemperatureC.isnull()])


for df in (joined, joined_test):
    for c in df.columns:
        if c.endswith('_y'):
            if c in df.columns:
                df.drop(c, inplace=True, axis=1)


# Next we'll fill in missing values to avoid complications with `NA`'s.
# `NA` (not available) is how Pandas indicates missing values; many models
# have problems when missing values are present, so it's always important
# to think about how to deal with them. In these cases, we are picking an
# arbitrary *signal value* that doesn't otherwise appear in the data.

for df in (joined, joined_test):
    df['CompetitionOpenSinceYear'] = df.CompetitionOpenSinceYear.fillna(
        1900).astype(np.int32)
    df['CompetitionOpenSinceMonth'] = df.CompetitionOpenSinceMonth.fillna(
        1).astype(np.int32)
    df['Promo2SinceYear'] = df.Promo2SinceYear.fillna(1900).astype(np.int32)
    df['Promo2SinceWeek'] = df.Promo2SinceWeek.fillna(1).astype(np.int32)


# Next we'll extract features "CompetitionOpenSince" and
# "CompetitionDaysOpen". Note the use of `apply()` in mapping a function
# across dataframe values.

for df in (joined, joined_test):
    df["CompetitionOpenSince"] = pd.to_datetime(dict(year=df.CompetitionOpenSinceYear,
                                                     month=df.CompetitionOpenSinceMonth, day=15))
    df["CompetitionDaysOpen"] = df.Date.subtract(
        df.CompetitionOpenSince).dt.days


# We'll replace some erroneous / outlying data.

for df in (joined, joined_test):
    df.loc[df.CompetitionDaysOpen < 0, "CompetitionDaysOpen"] = 0
    df.loc[df.CompetitionOpenSinceYear < 1990, "CompetitionDaysOpen"] = 0


# We add "CompetitionMonthsOpen" field, limiting the maximum to 2 years to
# limit number of unique categories.

for df in (joined, joined_test):
    df["CompetitionMonthsOpen"] = df["CompetitionDaysOpen"] // 30
    df.loc[df.CompetitionMonthsOpen > 24, "CompetitionMonthsOpen"] = 24
joined.CompetitionMonthsOpen.unique()


# Same process for Promo dates.

for df in (joined, joined_test):
    df["Promo2Since"] = pd.to_datetime(df.apply(lambda x: Week(
        x.Promo2SinceYear, x.Promo2SinceWeek).monday(), axis=1).astype(pd.datetime))
    df["Promo2Days"] = df.Date.subtract(df["Promo2Since"]).dt.days


for df in (joined, joined_test):
    df.loc[df.Promo2Days < 0, "Promo2Days"] = 0
    df.loc[df.Promo2SinceYear < 1990, "Promo2Days"] = 0
    df["Promo2Weeks"] = df["Promo2Days"] // 7
    df.loc[df.Promo2Weeks < 0, "Promo2Weeks"] = 0
    df.loc[df.Promo2Weeks > 25, "Promo2Weeks"] = 25
    df.Promo2Weeks.unique()


joined.to_feather(f'{PATH}joined')
joined_test.to_feather(f'{PATH}joined_test')


# ## Durations

# It is common when working with time series data to extract data that explains relationships across rows as opposed to columns, e.g.:
# * Running averages
# * Time until next event
# * Time since last event
#
# This is often difficult to do with most table manipulation frameworks, since they are designed to work with relationships across columns. As such, we've created a class to handle this type of data.
#
# We'll define a function `get_elapsed` for cumulative counting across a sorted dataframe. Given a particular field `fld` to monitor, this function will start tracking time since the last occurrence of that field. When the field is seen again, the counter is set to zero.
#
# Upon initialization, this will result in datetime na's until the field
# is encountered. This is reset every time a new store is seen. We'll see
# how to use this shortly.

def get_elapsed(fld, pre):
    day1 = np.timedelta64(1, 'D')
    last_date = np.datetime64()
    last_store = 0
    res = []

    for s, v, d in zip(df.Store.values, df[fld].values, df.Date.values):
        if s != last_store:
            last_date = np.datetime64()
            last_store = s
        if v:
            last_date = d
        res.append(
            ((d - last_date).astype('timedelta64[D]') / day1).astype(int))
    df[pre + fld] = res


# We'll be applying this to a subset of columns:

columns = ["Date", "Store", "Promo", "StateHoliday", "SchoolHoliday"]


df = train[columns]


df = test[columns]


# Let's walk through an example.
#
# Say we're looking at School Holiday. We'll first sort by Store, then Date, and then call `add_elapsed('SchoolHoliday', 'After')`:
# This will apply to each row with School Holiday:
# * A applied to every row of the dataframe in order of store and date
# * Will add to the dataframe the days since seeing a School Holiday
# * If we sort in the other direction, this will count the days until another holiday.

fld = 'SchoolHoliday'
df = df.sort_values(['Store', 'Date'])
get_elapsed(fld, 'After')
df = df.sort_values(['Store', 'Date'], ascending=[True, False])
get_elapsed(fld, 'Before')


# We'll do this for two more fields.

fld = 'StateHoliday'
df = df.sort_values(['Store', 'Date'])
get_elapsed(fld, 'After')
df = df.sort_values(['Store', 'Date'], ascending=[True, False])
get_elapsed(fld, 'Before')


fld = 'Promo'
df = df.sort_values(['Store', 'Date'])
get_elapsed(fld, 'After')
df = df.sort_values(['Store', 'Date'], ascending=[True, False])
get_elapsed(fld, 'Before')


# We're going to set the active index to Date.

df = df.set_index("Date")


# Then set null values from elapsed field calculations to 0.

columns = ['SchoolHoliday', 'StateHoliday', 'Promo']


for o in ['Before', 'After']:
    for p in columns:
        a = o + p
        df[a] = df[a].fillna(0)


# Next we'll demonstrate window functions in pandas to calculate rolling quantities.
#
# Here we're sorting by date (`sort_index()`) and counting the number of
# events of interest (`sum()`) defined in `columns` in the following week
# (`rolling()`), grouped by Store (`groupby()`). We do the same in the
# opposite direction.

bwd = df[['Store'] + columns].sort_index().groupby("Store").rolling(7,
                                                                    min_periods=1).sum()


fwd = df[['Store'] + columns].sort_index(ascending=False
                                         ).groupby("Store").rolling(7, min_periods=1).sum()


# Next we want to drop the Store indices grouped together in the window function.
#
# Often in pandas, there is an option to do this in place. This is time
# and memory efficient when working with large datasets.

bwd.drop('Store', 1, inplace=True)
bwd.reset_index(inplace=True)


fwd.drop('Store', 1, inplace=True)
fwd.reset_index(inplace=True)


df.reset_index(inplace=True)


# Now we'll merge these values onto the df.

df = df.merge(bwd, 'left', ['Date', 'Store'], suffixes=['', '_bw'])
df = df.merge(fwd, 'left', ['Date', 'Store'], suffixes=['', '_fw'])


df.drop(columns, 1, inplace=True)


df.head()


# It's usually a good idea to back up large tables of extracted / wrangled
# features before you join them onto another one, that way you can go back
# to it easily if you need to make changes to it.

df.to_feather(f'{PATH}df')


df = pd.read_feather(f'{PATH}df')


df["Date"] = pd.to_datetime(df.Date)


df.columns


joined = join_df(joined, df, ['Store', 'Date'])


joined_test = join_df(joined_test, df, ['Store', 'Date'])


# The authors also removed all instances where the store had zero sale /
# was closed. We speculate that this may have cost them a higher standing
# in the competition. One reason this may be the case is that a little
# exploratory data analysis reveals that there are often periods where
# stores are closed, typically for refurbishment. Before and after these
# periods, there are naturally spikes in sales that one might expect. By
# ommitting this data from their training, the authors gave up the ability
# to leverage information about these periods to predict this otherwise
# volatile behavior.

joined = joined[joined.Sales != 0]


# We'll back this up as well.

joined.reset_index(inplace=True)
joined_test.reset_index(inplace=True)


joined.to_feather(f'{PATH}joined')
joined_test.to_feather(f'{PATH}joined_test')


# We now have our final set of engineered features.
#
# While these steps were explicitly outlined in the paper, these are all
# fairly typical feature engineering steps for dealing with time series
# data and are practical in any similar setting.

# ## Create features

joined = pd.read_feather(f'{PATH}joined')
joined_test = pd.read_feather(f'{PATH}joined_test')


joined.head().T.head(40)


# Now that we've engineered all our features, we need to convert to input compatible with a neural network.
#
# This includes converting categorical variables into contiguous integers
# or one-hot encodings, normalizing continuous features to standard
# normal, etc...

cat_vars = ['Store', 'DayOfWeek', 'Year', 'Month', 'Day', 'StateHoliday', 'CompetitionMonthsOpen',
            'Promo2Weeks', 'StoreType', 'Assortment', 'PromoInterval', 'CompetitionOpenSinceYear', 'Promo2SinceYear',
            'State', 'Week', 'Events', 'Promo_fw', 'Promo_bw', 'StateHoliday_fw', 'StateHoliday_bw',
            'SchoolHoliday_fw', 'SchoolHoliday_bw']

contin_vars = ['CompetitionDistance', 'Max_TemperatureC', 'Mean_TemperatureC', 'Min_TemperatureC',
               'Max_Humidity', 'Mean_Humidity', 'Min_Humidity', 'Max_Wind_SpeedKm_h',
               'Mean_Wind_SpeedKm_h', 'CloudCover', 'trend', 'trend_DE',
               'AfterStateHoliday', 'BeforeStateHoliday', 'Promo', 'SchoolHoliday']

n = len(joined)
n


dep = 'Sales'
joined = joined[cat_vars + contin_vars + [dep, 'Date']].copy()


joined_test[dep] = 0
joined_test = joined_test[cat_vars + contin_vars + [dep, 'Date', 'Id']].copy()


for v in cat_vars:
    joined[v] = joined[v].astype('category').cat.as_ordered()


apply_cats(joined_test, joined)


for v in contin_vars:
    joined[v] = joined[v].astype('float32')
    joined_test[v] = joined_test[v].astype('float32')


# We're going to run on a sample.

idxs = get_cv_idxs(n, val_pct=150000 / n)
joined_samp = joined.iloc[idxs].set_index("Date")
samp_size = len(joined_samp)
samp_size


# To run on the full dataset, use this instead:

samp_size = n
joined_samp = joined.set_index("Date")


# We can now process our data...

joined_samp.head(2)


df, y, nas, mapper = proc_df(joined_samp, 'Sales', do_scale=True)
yl = np.log(y)


joined_test = joined_test.set_index("Date")


df_test, _, nas, mapper = proc_df(joined_test, 'Sales', do_scale=True, skip_flds=['Id'],
                                  mapper=mapper, na_dict=nas)


df.head(2)


# In time series data, cross-validation is not random. Instead, our holdout data is generally the most recent data, as it would be in real application. This issue is discussed in detail in [this post](http://www.fast.ai/2017/11/13/validation-sets/) on our web site.
#
# One approach is to take the last 25% of rows (sorted by date) as our
# validation set.

train_ratio = 0.75
# train_ratio = 0.9
train_size = int(samp_size * train_ratio)
train_size
val_idx = list(range(train_size, len(df)))


# An even better option for picking a validation set is using the exact
# same length of time period as the test set uses - this is implemented
# here:

val_idx = np.flatnonzero(
    (df.index <= datetime.datetime(2014, 9, 17)) & (df.index >= datetime.datetime(2014, 8, 1)))


val_idx = [0]


# ## DL

# We're ready to put together our models.
#
# Root-mean-squared percent error is the metric Kaggle used for this
# competition.

def inv_y(a): return np.exp(a)


def exp_rmspe(y_pred, targ):
    targ = inv_y(targ)
    pct_var = (targ - inv_y(y_pred)) / targ
    return math.sqrt((pct_var**2).mean())


max_log_y = np.max(yl)
y_range = (0, max_log_y * 1.2)


# We can create a ModelData object directly from out data frame.

md = ColumnarModelData.from_data_frame(PATH, val_idx, df, yl.astype(np.float32), cat_flds=cat_vars, bs=128,
                                       test_df=df_test)


# Some categorical variables have a lot more levels than others. Store, in
# particular, has over a thousand!

cat_sz = [(c, len(joined_samp[c].cat.categories) + 1) for c in cat_vars]


cat_sz


# We use the *cardinality* of each variable (that is, its number of unique
# values) to decide how large to make its *embeddings*. Each level will be
# associated with a vector with length defined as below.

emb_szs = [(c, min(50, (c + 1) // 2)) for _, c in cat_sz]


emb_szs


m = md.get_learner(emb_szs, len(df.columns) - len(cat_vars),
                   0.04, 1, [1000, 500], [0.001, 0.01], y_range=y_range)
lr = 1e-3


m.lr_find()


m.sched.plot(100)


# ### Sample

m = md.get_learner(emb_szs, len(df.columns) - len(cat_vars),
                   0.04, 1, [1000, 500], [0.001, 0.01], y_range=y_range)
lr = 1e-3


m.fit(lr, 3, metrics=[exp_rmspe])


m.fit(lr, 5, metrics=[exp_rmspe], cycle_len=1)


m.fit(lr, 2, metrics=[exp_rmspe], cycle_len=4)


# ### All

m = md.get_learner(emb_szs, len(df.columns) - len(cat_vars),
                   0.04, 1, [1000, 500], [0.001, 0.01], y_range=y_range)
lr = 1e-3


m.fit(lr, 1, metrics=[exp_rmspe])


m.fit(lr, 3, metrics=[exp_rmspe])


m.fit(lr, 3, metrics=[exp_rmspe], cycle_len=1)


# ### Test

m = md.get_learner(emb_szs, len(df.columns) - len(cat_vars),
                   0.04, 1, [1000, 500], [0.001, 0.01], y_range=y_range)
lr = 1e-3


m.fit(lr, 3, metrics=[exp_rmspe])


m.fit(lr, 3, metrics=[exp_rmspe], cycle_len=1)


m.save('val0')


m.load('val0')


x, y = m.predict_with_targs()


exp_rmspe(x, y)


pred_test = m.predict(True)


pred_test = np.exp(pred_test)


joined_test['Sales'] = pred_test


csv_fn = f'{PATH}tmp/sub.csv'


joined_test[['Id', 'Sales']].to_csv(csv_fn, index=False)


FileLink(csv_fn)


# ## RF

from sklearn.ensemble import RandomForestRegressor


((val, trn), (y_val, y_trn)) = split_by_idx(val_idx, df.values, yl)


m = RandomForestRegressor(n_estimators=40, max_features=0.99, min_samples_leaf=2,
                          n_jobs=-1, oob_score=True)
m.fit(trn, y_trn)


preds = m.predict(val)
m.score(trn, y_trn), m.score(val, y_val), m.oob_score_, exp_rmspe(preds, y_val)
