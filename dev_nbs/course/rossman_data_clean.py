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

# # Rossman data preparation

# To illustrate the techniques we need to apply before feeding all the data to a Deep Learning model, we are going to take the example of the [Rossmann sales Kaggle competition](https://www.kaggle.com/c/rossmann-store-sales). Given a wide range of information about a store, we are going to try predict their sale number on a given day. This is very useful to be able to manage stock properly and be able to properly satisfy the demand without wasting anything. The official training set was giving a lot of informations about various stores in Germany, but it was also allowed to use additional data, as long as it was made public and available to all participants.
#
# We are going to reproduce most of the steps of one of the winning teams that they highlighted in [Entity Embeddings of Categorical Variables](https://arxiv.org/pdf/1604.06737.pdf). In addition to the official data, teams in the top of the leaderboard also used information about the weather, the states of the stores or the Google trends of those days. We have assembled all that additional data in one file available for download [here](http://files.fast.ai/part2/lesson14/rossmann.tgz) if you want to replicate those steps.

# ### A first look at the data

# First things first, let's import everything we will need.

from fastai.tabular.all import *

# If you have download the previous file and decompressed it in a folder named rossmann in the fastai data folder, you should see the following list of files with this instruction:

path = Config().data / 'rossmann'
path.ls()

# The data that comes from Kaggle is in 'train.csv', 'test.csv', 'store.csv' and 'sample_submission.csv'. The other files are the additional data we were talking about. Let's start by loading everything using pandas.

table_names = ['train', 'store', 'store_states', 'state_names', 'googletrend', 'weather', 'test']
tables = [pd.read_csv(path / f'{fname}.csv', low_memory=False) for fname in table_names]
train, store, store_states, state_names, googletrend, weather, test = tables

# To get an idea of the amount of data available, let's just look at the length of the training and test tables.

len(train), len(test)

# So we have more than one million records available. Let's have a look at what's inside:

train.head()

# The `Store` column contains the id of the stores, then we are given the id of the day of the week, the exact date, if the store was open on that day, if there were any promotion in that store during that day, and if it was a state or school holiday. The `Customers` column is given as an indication, and the `Sales` column is what we will try to predict.
#
# If we look at the test table, we have the same columns, minus `Sales` and `Customers`, and it looks like we will have to predict on dates that are after the ones of the train table.

test.head()

# The other table given by Kaggle contains some information specific to the stores: their type, what the competition looks like, if they are engaged in a permanent promotion program, and if so since then.

store.head().T

# Now let's have a quick look at our four additional dataframes. `store_states` just gives us the abbreviated name of the sate of each store.

store_states.head()

# We can match them to their real names with `state_names`.

state_names.head()

# Which is going to be necessary if we want to use the `weather` table:

weather.head().T

# Lastly the googletrend table gives us the trend of the brand in each state and in the whole of Germany.

googletrend.head()


# Before we apply the fastai preprocessing, we will need to join the store table and the additional ones with our training and test table. Then, as we saw in our first example in chapter 1, we will need to split our variables between categorical and continuous. Before we do that, though, there is one type of variable that is a bit different from the others: dates.
#
# We could turn each particular day in a category but there are cyclical information in dates we would miss if we did that. We already have the day of the week in our tables, but maybe the day of the month also bears some significance. People might be more inclined to go shopping at the beginning or the end of the month. The number of the week/month is also important to detect seasonal influences.
#
# Then we will try to exctract meaningful information from those dates. For instance promotions on their own are important inputs, but maybe the number of running weeks with promotion is another useful information as it will influence customers. A state holiday in itself is important, but it's more significant to know if we are the day before or after such a holiday as it will impact sales. All of those might seem very specific to this dataset, but you can actually apply them in any tabular data containing time information.
#
# This first step is called feature-engineering and is extremely important: your model will try to extract useful information from your data but any extra help you can give it in advance is going to make training easier, and the final result better. In Kaggle Competitions using tabular data, it's often the way people prepared their data that makes the difference in the final leaderboard, not the exact model used.

# ### Feature Engineering

# #### Merging tables

# To merge tables together, we will use this little helper function that relies on the pandas library. It will merge the tables `left` and `right` by looking at the column(s) which names are in `left_on` and `right_on`: the information in `right` will be added to the rows of the tables in `left` when the data in `left_on` inside `left` is the same as the data in `right_on` inside `right`. If `left_on` and `right_on` are the same, we don't have to pass `right_on`. We keep the fields in `right` that have the same names as fields in `left` and add a `_y` suffix (by default) to those field names.

def join_df(left, right, left_on, right_on=None, suffix='_y'):
    if right_on is None:
        right_on = left_on
    return left.merge(right, how='left', left_on=left_on, right_on=right_on,
                      suffixes=("", suffix))


# First, let's replace the state names in the weather table by the abbreviations, since that's what is used in the other tables.

weather = join_df(weather, state_names, "file", "StateName")
weather[['file', 'Date', 'State', 'StateName']].head()

# To double-check the merge happened without incident, we can check that every row has a `State` with this line:

len(weather[weather.State.isnull()])

# We can now safely remove the columns with the state names (`file` and `StateName`) since they we'll use the short codes.

weather.drop(columns=['file', 'StateName'], inplace=True)

# To add the weather informations to our `store` table, we first use the table `store_states` to match a store code with the corresponding state, then we merge with our weather table.

store = join_df(store, store_states, 'Store')
store = join_df(store, weather, 'State')

# And again, we can check if the merge went well by looking if new NaNs where introduced.

len(store[store.Mean_TemperatureC.isnull()])

# Next, we want to join the `googletrend` table to this `store` table. If you remember from our previous look at it, it's not exactly in the same format:

googletrend.head()

# We will need to change the column with the states and the columns with the dates:
# - in the column `fil`, the state names contain `Rossmann_DE_XX` with `XX` being the code of the state, so we want to remove `Rossmann_DE`. We will do this by creating a new column containing the last part of a split of the string by '\_'.
# - in the column `week`, we will extract the date corresponding to the beginning of the week in a new column by taking the last part of a split on ' - '.
#
# In pandas, creating a new column is very easy: you just have to define them.

googletrend['Date'] = googletrend.week.str.split(' - ', expand=True)[0]
googletrend['State'] = googletrend.file.str.split('_', expand=True)[2]
googletrend.head()

# Let's check everything went well by looking at the values in the new `State` column of our `googletrend` table.

store['State'].unique(), googletrend['State'].unique()

# We have two additional values in the second (`None` and 'SL') but this isn't a problem since they'll be ignored when we join. One problem however is that 'HB,NI' in the first table is named 'NI' in the second one, so we need to change that.

googletrend.loc[googletrend.State == 'NI', "State"] = 'HB,NI'

# Why do we have a `None` in state? As we said before, there is a global trend for Germany that corresponds to `Rosmann_DE` in the field `file`. For those, the previous split failed which gave the `None` value. We will keep this global trend and put it in a new column.

trend_de = googletrend[googletrend.file == 'Rossmann_DE'][['Date', 'trend']]

# Then we can merge it with the rest of our trends, by adding the suffix '\_DE' to know it's the general trend.

googletrend = join_df(googletrend, trend_de, 'Date', suffix='_DE')

# Then at this stage, we can remove the columns `file` and `week`since they won't be useful anymore, as well as the rows where `State` is `None` (since they correspond to the global trend that we saved in another column).

googletrend.drop(columns=['file', 'week'], axis=1, inplace=True)
googletrend = googletrend[~googletrend['State'].isnull()]

# The last thing missing to be able to join this with or store table is to extract the week from the date in this table and in the store table: we need to join them on week values since each trend is given for the full week that starts on the indicated date. This is linked to the next topic in feature engineering: extracting dateparts.

# #### Adding dateparts

# If your table contains dates, you will need to split the information there in several column for your Deep Learning model to be able to train properly. There is the basic stuff, such as the day number, week number, month number or year number, but anything that can be relevant to your problem is also useful. Is it the beginning or the end of the month? Is it a holiday?
#
# To help with this, the fastai library as a convenience function called `add_datepart`. It will take a dataframe and a column you indicate, try to read it as a date, then add all those new columns. If we go back to our `googletrend` table, we now have gour columns.

googletrend.head()

# If we add the dateparts, we will gain a lot more

googletrend = add_datepart(googletrend, 'Date', drop=False)

googletrend.head().T

# We chose the option `drop=False` as we want to keep the `Date` column  for now. Another option is to add the `time` part of the date, but it's not relevant to our problem here.
#
# Now we can join our Google trends with the information in the `store` table, it's just a join on \['Week', 'Year'\] once we apply `add_datepart` to that table. Note that we only keep the initial columns of `googletrend` with `Week` and `Year` to avoid all the duplicates.

googletrend = googletrend[['trend', 'State', 'trend_DE', 'Week', 'Year']]
store = add_datepart(store, 'Date', drop=False)
store = join_df(store, googletrend, ['Week', 'Year', 'State'])

# At this stage, `store` contains all the information about the stores, the weather on that day and the Google trends applicable. We only have to join it with our training and test table. We have to use `make_date` before being able to execute that merge, to convert the `Date` column of `train` and `test` to proper date format.

make_date(train, 'Date')
make_date(test, 'Date')
train_fe = join_df(train, store, ['Store', 'Date'])
test_fe = join_df(test, store, ['Store', 'Date'])

# #### Elapsed times

# Another feature that can be useful is the elapsed time before/after a certain event occurs. For instance the number of days since the last promotion or before the next school holiday. Like for the date parts, there is a fastai convenience function that will automatically add them.
#
# One thing to take into account here is that you will need to use that function on the whole time series you have, even the test data: there might be a school holiday that takes place during the training data and it's going to impact those new features in the test data.

all_ftrs = train_fe.append(test_fe, sort=False)

# We will consider the elapsed times for three events: 'Promo', 'StateHoliday' and 'SchoolHoliday'. Note that those must correspondon to booleans in your dataframe. 'Promo' and 'SchoolHoliday' already are (only 0s and 1s) but 'StateHoliday' has multiple values.

all_ftrs['StateHoliday'].unique()

# If we refer to the explanation on Kaggle, 'b' is for Easter, 'c' for Christmas and 'a' for the other holidays. We will just converts this into a boolean that flags any holiday.

all_ftrs.StateHoliday = all_ftrs.StateHoliday != '0'

# Now we can add, for each store, the number of days since or until the next promotion, state or school holiday. This will take a little while since the whole table is big.

all_ftrs = add_elapsed_times(all_ftrs, ['Promo', 'StateHoliday', 'SchoolHoliday'],
                             date_field='Date', base_field='Store')

# It added a four new features. If we look at 'StateHoliday' for instance:

[c for c in all_ftrs.columns if 'StateHoliday' in c]

# The column 'AfterStateHoliday' contains the number of days since the last state holiday, 'BeforeStateHoliday' the number of days until the next one. As for 'StateHoliday_bw' and 'StateHoliday_fw', they contain the number of state holidays in the past or future seven days respectively. The same four columns have been added for 'Promo' and 'SchoolHoliday'.
#
# Now that we have added those features, we can split again our tables between the training and the test one.

train_df = all_ftrs.iloc[:len(train_fe)]
test_df = all_ftrs.iloc[len(train_fe):]

# One last thing the authors of this winning solution did was to remove the rows with no sales, which correspond to exceptional closures of the stores. This might not have been a good idea since even if we don't have access to the same features in the test data, it can explain why we have some spikes in the training data.

train_df = train_df[train_df.Sales != 0.]

# We will use those for training but since all those steps took a bit of time, it's a good idea to save our progress until now. We will just pickle those tables on the hard drive.

train_df.to_pickle(path / 'train_clean')
test_df.to_pickle(path / 'test_clean')
