from IPython.display import display
from fastai.imports import *
from fastai.structured import *
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
import feather
PATH = "data/bulldozers/"
# The good news is that modern machine learning can be distilled down to a couple of key techniques that are of very wide applicability. Recent studies have shown that the vast majority of datasets can be best modeled with just two methods:
#
# - *Ensembles of decision trees* (i.e. Random Forests and Gradient Boosting Machines), mainly for structured data (such as you might find in a database table at most companies)
# - *Multi-layered neural networks learnt with SGD* (i.e. shallow and/or deep learning), mainly for unstructured data (such as audio, vision, and natural language)
#
# In this course we'll be doing a deep dive into random forests, and simple models learnt with SGD. You'll be learning about gradient boosting and deep learning in part 2.
# ### ...this dataset
# We will be looking at the Blue Book for Bulldozers Kaggle Competition: "The goal of the contest is to predict the sale price of a particular piece of heavy equiment at auction based on it's usage, equipment type, and configuration.  The data is sourced from auction result postings and includes information on usage and equipment configurations."
#
# This is a very common type of dataset and prediciton problem, and similar to what you may see in your project or workplace.
# ### ...Kaggle Competitions
# Kaggle is an awesome resource for aspiring data scientists or anyone looking to improve their machine learning skills.  There is nothing like being able to get hands-on practice and receiving real-time feedback to help you improve your skills.
#
# Kaggle provides:
#
# 1. Interesting data sets
# 2. Feedback on how you're doing
# 3. A leader board to see what's good, what's possible, and what's state-of-art.
# 4. Blog posts by winning contestants share useful tips and techniques.
# ## The data
# ### Look at the data
# Kaggle provides info about some of the fields of our dataset; on the [Kaggle Data info](https://www.kaggle.com/c/bluebook-for-bulldozers/data) page they say the following:
#
# For this competition, you are predicting the sale price of bulldozers sold at auctions. The data for this competition is split into three parts:
#
# - **Train.csv** is the training set, which contains data through the end of 2011.
# - **Valid.csv** is the validation set, which contains data from January 1, 2012 - April 30, 2012. You make predictions on this set throughout the majority of the competition. Your score on this set is used to create the public leaderboard.
# - **Test.csv** is the test set, which won't be released until the last week of the competition. It contains data from May 1, 2012 - November 2012. Your score on the test set determines your final rank for the competition.
#
# The key fields are in train.csv are:
#
# - SalesID: the unique identifier of the sale
# - MachineID: the unique identifier of a machine.  A machine can be sold multiple times
# - saleprice: what the machine sold for at auction (only provided in train.csv)
# - saledate: the date of the sale
df_raw = pd.read_csv(f'{PATH}Train.csv', low_memory=False,
                     parse_dates=["saledate"])
# In any sort of data science work, it's **important to look at your data**, to make sure you understand the format, how it's stored, what type of values it holds, etc. Even if you've read descriptions about your data, the actual data may not be what you expect.


def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000):
        print(df)


display_all(df_raw.tail().T)
display_all(df_raw.describe(include='all').T)
# It's important to note what metric is being used for a project. Generally, selecting the metric(s) is an important part of the project setup. However, in this case Kaggle tells us what metric to use: RMSLE (root mean squared log error) between the actual and predicted auction prices. Therefore we take the log of the prices, so that RMSE will give us what we need.
df_raw.SalePrice = np.log(df_raw.SalePrice)
# ### Initial processing
m = RandomForestRegressor(n_jobs=-1)
# The following code is supposed to fail due to string values in the input data
# m.fit(df_raw.drop('SalePrice', axis=1), df_raw.SalePrice)
# This dataset contains a mix of **continuous** and **categorical** variables.
#
# The following method extracts particular date fields from a complete datetime for the purpose of constructing categoricals.  You should always consider this feature extraction step when working with date-time. Without expanding your date-time into these additional fields, you can't capture any trend/cyclical behavior as a function of time at any of these granularities.
add_datepart(df_raw, 'saledate')
df_raw.saleYear.head()
# The categorical variables are currently stored as strings, which is inefficient, and doesn't provide the numeric coding required for a random forest. Therefore we call `train_cats` to convert strings to pandas categories.
train_cats(df_raw)
# We can specify the order to use for categorical variables if we wish:
df_raw.UsageBand.cat.categories
df_raw.UsageBand.cat.set_categories(['High', 'Medium', 'Low'], ordered=True, inplace=True)
# Normally, pandas will continue displaying the text categories, while treating them as numerical data internally. Optionally, we can replace the text categories with numbers, which will make this variable non-categorical, like so:.
df_raw.UsageBand = df_raw.UsageBand.cat.codes
# We're still not quite done - for instance we have lots of missing values, which we can't pass directly to a random forest.
display_all(df_raw.isnull().sum().sort_index() / len(df_raw))
# But let's save this file for now, since it's already in format can we be stored and accessed efficiently.
os.makedirs('tmp', exist_ok=True)
df_raw.to_feather('tmp/bulldozers-raw')
# ### Pre-processing
# In the future we can simply read it from this fast format.
df_raw = feather.read_dataframe('tmp/bulldozers-raw')
# We'll replace categories with their numeric codes, handle missing continuous values, and split the dependent variable into a separate variable.
df, y, nas = proc_df(df_raw, 'SalePrice')
# We now have something we can pass to a random forest!
m = RandomForestRegressor(n_jobs=-1)
m.fit(df, y)
m.score(df, y)
# In statistics, the coefficient of determination, denoted R2 or r2 and pronounced "R squared", is the proportion of the variance in the dependent variable that is predictable from the independent variable(s). https://en.wikipedia.org/wiki/Coefficient_of_determination
# Wow, an r^2 of 0.98 - that's great, right? Well, perhaps not...
#
# Possibly **the most important idea** in machine learning is that of having separate training & validation data sets. As motivation, suppose you don't divide up your data, but instead use all of it.  And suppose you have lots of parameters:
#
# The error for the pictured data points is lowest for the model on the far right (the blue curve passes through the red points almost perfectly), yet it's not the best choice.  Why is that?  If you were to gather some new data points, they most likely would not be on that curve in the graph on the right, but would be closer to the curve in the middle graph.
#
# This illustrates how using all our data can lead to **overfitting**. A validation set helps diagnose this problem.


def split_vals(a, n): return a[:n].copy(), a[n:].copy()


n_valid = 12000  # same as Kaggle's test set size
n_trn = len(df) - n_valid
raw_train, raw_valid = split_vals(df_raw, n_trn)
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)
X_train.shape, y_train.shape, X_valid.shape
# # Random Forests
# ## Base model
# Let's try our model again, this time with separate training and validation sets.


def rmse(x, y): return math.sqrt(((x - y)**2).mean())


def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
           m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'):
        res.append(m.oob_score_)
    print(res)


m = RandomForestRegressor(n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
# An r^2 in the high-80's isn't bad at all (and the RMSLE puts us around rank 100 of 470 on the Kaggle leaderboard), but we can see from the validation set score that we're over-fitting badly. To understand this issue, let's simplify things down to a single small tree.
# ## Speeding things up
df_trn, y_trn, nas = proc_df(df_raw, 'SalePrice', subset=30000, na_dict=nas)
X_train, _ = split_vals(df_trn, 20000)
y_train, _ = split_vals(y_trn, 20000)
m = RandomForestRegressor(n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
# ## Single tree
m = RandomForestRegressor(n_estimators=1, max_depth=3, bootstrap=False, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
draw_tree(m.estimators_[0], df_trn, precision=3)
# Let's see what happens if we create a bigger tree.
m = RandomForestRegressor(n_estimators=1, bootstrap=False, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
# The training set result looks great! But the validation set is worse than our original model. This is why we need to use *bagging* of multiple trees to get more generalizable results.
# ## Bagging
# ### Intro to bagging
# To learn about bagging in random forests, let's start with our basic model again.
m = RandomForestRegressor(n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
# We'll grab the predictions for each individual tree, and look at one example.
preds = np.stack([t.predict(X_valid) for t in m.estimators_])
preds[:, 0], np.mean(preds[:, 0]), y_valid[0]
preds.shape
plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i + 1], axis=0)) for i in range(10)])
plt.show()
# The shape of this curve suggests that adding more trees isn't going to help us much. Let's check. (Compare this to our original model on a sample)
m = RandomForestRegressor(n_estimators=20, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
m = RandomForestRegressor(n_estimators=40, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
m = RandomForestRegressor(n_estimators=80, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
# ### Out-of-bag (OOB) score
# Is our validation set worse than our training set because we're over-fitting, or because the validation set is for a different time period, or a bit of both? With the existing information we've shown, we can't tell. However, random forests have a very clever trick called *out-of-bag (OOB) error* which can handle this (and more!)
#
# The idea is to calculate error on the training set, but only include the trees in the calculation of a row's error where that row was *not* included in training that tree. This allows us to see whether the model is over-fitting, without needing a separate validation set.
#
# This also has the benefit of allowing us to see whether our model generalizes, even if we only have a small amount of data so want to avoid separating some out to create a validation set.
#
# This is as simple as adding one more parameter to our model constructor. We print the OOB error last in our `print_score` function below.
m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
# This shows that our validation set time difference is making an impact, as is model over-fitting.
# ## Reducing over-fitting
# ### Subsampling
# It turns out that one of the easiest ways to avoid over-fitting is also one of the best ways to speed up analysis: *subsampling*. Let's return to using our full dataset, so that we can demonstrate the impact of this technique.
df_trn, y_trn, nas = proc_df(df_raw, 'SalePrice')
X_train, X_valid = split_vals(df_trn, n_trn)
y_train, y_valid = split_vals(y_trn, n_trn)
# The basic idea is this: rather than limit the total amount of data that our model can access, let's instead limit it to a *different* random subset per tree. That way, given enough trees, the model can still see *all* the data, but for each individual tree it'll be just as fast as if we had cut down our dataset as before.
set_rf_samples(20000)
m = RandomForestRegressor(n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
# Since each additional tree allows the model to see more data, this approach can make additional trees more useful.
m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
# ### Tree building parameters
# We revert to using a full bootstrap sample in order to show the impact of other over-fitting avoidance methods.
reset_rf_samples()
# Let's get a baseline for this full set to compare to.


def dectree_max_depth(tree):
    children_left = tree.children_left
    children_right = tree.children_right

    def walk(node_id):
        if (children_left[node_id] != children_right[node_id]):
            left_max = 1 + walk(children_left[node_id])
            right_max = 1 + walk(children_right[node_id])
            return max(left_max, right_max)
        else:  # leaf
            return 1
    root_node_id = 0
    return walk(root_node_id)


m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
t = m.estimators_[0].tree_
dectree_max_depth(t)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
t = m.estimators_[0].tree_
dectree_max_depth(t)
# Another way to reduce over-fitting is to grow our trees less deeply. We do this by specifying (with `min_samples_leaf`) that we require some minimum number of rows in every leaf node. This has two benefits:
#
# - There are less decision rules for each leaf node; simpler models should generalize better
# - The predictions are made by averaging more rows in the leaf node, resulting in less volatility
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
# We can also increase the amount of variation amongst the trees by not only use a sample of rows for each tree, but to also using a sample of *columns* for each *split*. We do this by specifying `max_features`, which is the proportion of features to randomly select from at each split.
# - None
# - 0.5
# - 'sqrt'
# - 1, 3, 5, 10, 25, 100
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
# We can't compare our results directly with the Kaggle competition, since it used a different validation set (and we can no longer to submit to this competition) - but we can at least see that we're getting similar results to the winners based on the dataset we have.
#
# The sklearn docs [show an example](http://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html) of different `max_features` methods with increasing numbers of trees - as you see, using a subset of features on each split requires using more trees, but results in better models:
# ![sklearn max_features chart](http://scikit-learn.org/stable/_images/sphx_glr_plot_ensemble_oob_001.png)
