# coding: utf-8
# # Random Forest from scratch!
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
from fastai.imports import *
from fastai.structured import *
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics
# ## Load in our data from last lesson
PATH = "data/bulldozers/"
df_raw = pd.read_feather('tmp/bulldozers-raw')
df_trn, y_trn, nas = proc_df(df_raw, 'SalePrice')
def split_vals(a, n): return a[:n], a[n:]
n_valid = 12000
n_trn = len(df_trn) - n_valid
X_train, X_valid = split_vals(df_trn, n_trn)
y_train, y_valid = split_vals(y_trn, n_trn)
raw_train, raw_valid = split_vals(df_raw, n_trn)
x_sub = X_train[['YearMade', 'MachineHoursCurrentMeter']]
# ## Basic data structures
class TreeEnsemble():
    def __init__(self, x, y, n_trees, sample_sz, min_leaf=5):
        np.random.seed(42)
        self.x, self.y, self.sample_sz, self.min_leaf = x, y, sample_sz, min_leaf
        self.trees = [self.create_tree() for i in range(n_trees)]

    def create_tree(self):
        rnd_idxs = np.random.permutation(len(self.y))[:self.sample_sz]
        return DecisionTree(self.x.iloc[rnd_idxs], self.y[rnd_idxs], min_leaf=self.min_leaf)
        
    def predict(self, x):
        return np.mean([t.predict(x) for t in self.trees], axis=0)
class DecisionTree():
    def __init__(self, x, y, idxs=None, min_leaf=5):
        self.x, self.y, self.idxs, self.min_leaf = x, y, idxs, min_leaf
m = TreeEnsemble(X_train, y_train, n_trees=10, sample_sz=1000, min_leaf=3)
m.trees[0]
class DecisionTree():
    def __init__(self, x, y, idxs=None, min_leaf=5):
        if idxs is None: idxs = np.arange(len(y))
        self.x, self.y, self.idxs, self.min_leaf = x, y, idxs, min_leaf
        self.n, self.c = len(idxs), x.shape[1]
        self.val = np.mean(y[idxs])
        self.score = float('inf')
        self.find_varsplit()
        
    # This just does one decision; we'll make it recursive later
    def find_varsplit(self):
        for i in range(self.c): self.find_better_split(i)
            
    # We'll write this later!
    def find_better_split(self, var_idx): pass
    
    @property
    def split_name(self): return self.x.columns[self.var_idx]
    
    @property
    def split_col(self): return self.x.values[self.idxs, self.var_idx]

    @property
    def is_leaf(self): return self.score == float('inf')
    
    def __repr__(self):
        s = f'n: {self.n}; val:{self.val}'
        if not self.is_leaf:
            s += f'; score:{self.score}; split:{self.split}; var:{self.split_name}'
        return s
m = TreeEnsemble(X_train, y_train, n_trees=10, sample_sz=1000, min_leaf=3)
m.trees[0]
m.trees[0].idxs
# ## Single branch
# ### Find best split given variable
ens = TreeEnsemble(x_sub, y_train, 1, 1000)
tree = ens.trees[0]
x_samp, y_samp = tree.x, tree.y
x_samp.columns
tree
m = RandomForestRegressor(n_estimators=1, max_depth=1, bootstrap=False)
m.fit(x_samp, y_samp)
draw_tree(m.estimators_[0], x_samp, precision=2)
def find_better_split(self, var_idx):
    x, y = self.x.values[self.idxs, var_idx], self.y[self.idxs]
    for i in range(self.n):
        lhs = x <= x[i]
        rhs = x > x[i]
        if rhs.sum() < self.min_leaf or lhs.sum() < self.min_leaf: continue
        lhs_std = y[lhs].std()
        rhs_std = y[rhs].std()
        curr_score = lhs_std * lhs.sum() + rhs_std * rhs.sum()
        if curr_score < self.score:
            self.var_idx, self.score, self.split = var_idx, curr_score, x[i]
get_ipython().run_line_magic('timeit', 'find_better_split(tree,1)')
tree
find_better_split(tree, 0); tree
# ### Speeding things up
tree = TreeEnsemble(x_sub, y_train, 1, 1000).trees[0]
def std_agg(cnt, s1, s2): return math.sqrt((s2 / cnt) - (s1 / cnt)**2)
def find_better_split(self, var_idx):
    x, y = self.x.values[self.idxs, var_idx], self.y[self.idxs]
    
    sort_idx = np.argsort(x)
    sort_y, sort_x = y[sort_idx], x[sort_idx]
    rhs_cnt, rhs_sum, rhs_sum2 = self.n, sort_y.sum(), (sort_y**2).sum()
    lhs_cnt, lhs_sum, lhs_sum2 = 0, 0., 0.
    for i in range(0, self.n - self.min_leaf):
        xi, yi = sort_x[i], sort_y[i]
        lhs_cnt += 1; rhs_cnt -= 1
        lhs_sum += yi; rhs_sum -= yi
        lhs_sum2 += yi**2; rhs_sum2 -= yi**2
        if i < self.min_leaf - 1 or xi == sort_x[i + 1]:
            continue
            
        lhs_std = std_agg(lhs_cnt, lhs_sum, lhs_sum2)
        rhs_std = std_agg(rhs_cnt, rhs_sum, rhs_sum2)
        curr_score = lhs_std * lhs_cnt + rhs_std * rhs_cnt
        if curr_score < self.score:
            self.var_idx, self.score, self.split = var_idx, curr_score, xi
get_ipython().run_line_magic('timeit', 'find_better_split(tree,1)')
tree
find_better_split(tree, 0); tree
DecisionTree.find_better_split = find_better_split
tree = TreeEnsemble(x_sub, y_train, 1, 1000).trees[0]; tree
# ## Full single tree
m = RandomForestRegressor(n_estimators=1, max_depth=2, bootstrap=False)
m.fit(x_samp, y_samp)
draw_tree(m.estimators_[0], x_samp, precision=2)
def find_varsplit(self):
    for i in range(self.c): self.find_better_split(i)
    if self.is_leaf: return
    x = self.split_col
    lhs = np.nonzero(x <= self.split)[0]
    rhs = np.nonzero(x > self.split)[0]
    self.lhs = DecisionTree(self.x, self.y, self.idxs[lhs])
    self.rhs = DecisionTree(self.x, self.y, self.idxs[rhs])
DecisionTree.find_varsplit = find_varsplit
tree = TreeEnsemble(x_sub, y_train, 1, 1000).trees[0]; tree
tree.lhs
tree.rhs
tree.lhs.lhs
tree.lhs.rhs
# ## Predictions
cols = ['MachineID', 'YearMade', 'MachineHoursCurrentMeter', 'ProductSize', 'Enclosure',
        'Coupler_System', 'saleYear']
get_ipython().run_line_magic('time', 'tree = TreeEnsemble(X_train[cols], y_train, 1, 1000).trees[0]')
x_samp, y_samp = tree.x, tree.y
m = RandomForestRegressor(n_estimators=1, max_depth=3, bootstrap=False)
m.fit(x_samp, y_samp)
draw_tree(m.estimators_[0], x_samp, precision=2, ratio=0.9, size=7)
def predict(self, x): return np.array([self.predict_row(xi) for xi in x])
DecisionTree.predict = predict
if something:
    x = do1()
else:
    x = do2()
x = do1() if something else do2()
x = something ? do1(): do2()
def predict_row(self, xi):
    if self.is_leaf: return self.val
    t = self.lhs if xi[self.var_idx] <= self.split else self.rhs
    return t.predict_row(xi)
DecisionTree.predict_row = predict_row
get_ipython().run_line_magic('time', 'preds = tree.predict(X_valid[cols].values)')
plt.scatter(preds, y_valid, alpha=0.05)
metrics.r2_score(preds, y_valid)
m = RandomForestRegressor(n_estimators=1, min_samples_leaf=5, bootstrap=False)
get_ipython().run_line_magic('time', 'm.fit(x_samp, y_samp)')
preds = m.predict(X_valid[cols].values)
plt.scatter(preds, y_valid, alpha=0.05)
metrics.r2_score(preds, y_valid)
# # Putting it together
class TreeEnsemble():
    def __init__(self, x, y, n_trees, sample_sz, min_leaf=5):
        np.random.seed(42)
        self.x, self.y, self.sample_sz, self.min_leaf = x, y, sample_sz, min_leaf
        self.trees = [self.create_tree() for i in range(n_trees)]

    def create_tree(self):
        idxs = np.random.permutation(len(self.y))[:self.sample_sz]
        return DecisionTree(self.x.iloc[idxs], self.y[idxs],
                    idxs=np.array(range(self.sample_sz)), min_leaf=self.min_leaf)
        
    def predict(self, x):
        return np.mean([t.predict(x) for t in self.trees], axis=0)
def std_agg(cnt, s1, s2): return math.sqrt((s2 / cnt) - (s1 / cnt)**2)
class DecisionTree():
    def __init__(self, x, y, idxs, min_leaf=5):
        self.x, self.y, self.idxs, self.min_leaf = x, y, idxs, min_leaf
        self.n, self.c = len(idxs), x.shape[1]
        self.val = np.mean(y[idxs])
        self.score = float('inf')
        self.find_varsplit()
        
    def find_varsplit(self):
        for i in range(self.c): self.find_better_split(i)
        if self.score == float('inf'): return
        x = self.split_col
        lhs = np.nonzero(x <= self.split)[0]
        rhs = np.nonzero(x > self.split)[0]
        self.lhs = DecisionTree(self.x, self.y, self.idxs[lhs])
        self.rhs = DecisionTree(self.x, self.y, self.idxs[rhs])

    def find_better_split(self, var_idx):
        x, y = self.x.values[self.idxs, var_idx], self.y[self.idxs]
        sort_idx = np.argsort(x)
        sort_y, sort_x = y[sort_idx], x[sort_idx]
        rhs_cnt, rhs_sum, rhs_sum2 = self.n, sort_y.sum(), (sort_y**2).sum()
        lhs_cnt, lhs_sum, lhs_sum2 = 0, 0., 0.
        for i in range(0, self.n - self.min_leaf):
            xi, yi = sort_x[i], sort_y[i]
            lhs_cnt += 1; rhs_cnt -= 1
            lhs_sum += yi; rhs_sum -= yi
            lhs_sum2 += yi**2; rhs_sum2 -= yi**2
            if i < self.min_leaf - 1 or xi == sort_x[i + 1]:
                continue
            lhs_std = std_agg(lhs_cnt, lhs_sum, lhs_sum2)
            rhs_std = std_agg(rhs_cnt, rhs_sum, rhs_sum2)
            curr_score = lhs_std * lhs_cnt + rhs_std * rhs_cnt
            if curr_score < self.score:
                self.var_idx, self.score, self.split = var_idx, curr_score, xi

    @property
    def split_name(self): return self.x.columns[self.var_idx]
    
    @property
    def split_col(self): return self.x.values[self.idxs, self.var_idx]

    @property
    def is_leaf(self): return self.score == float('inf')
    
    def __repr__(self):
        s = f'n: {self.n}; val:{self.val}'
        if not self.is_leaf:
            s += f'; score:{self.score}; split:{self.split}; var:{self.split_name}'
        return s

    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi):
        if self.is_leaf: return self.val
        t = self.lhs if xi[self.var_idx] <= self.split else self.rhs
        return t.predict_row(xi)
ens = TreeEnsemble(X_train[cols], y_train, 5, 1000)
preds = ens.predict(X_valid[cols].values)
plt.scatter(y_valid, preds, alpha=0.1, s=6);
metrics.r2_score(y_valid, preds)
get_ipython().run_line_magic('load_ext', 'Cython')
def fib1(n):
    a, b = 0, 1
    while b < n:
        a, b = b, a + b
get_ipython().run_cell_magic('cython', '', 'def fib2(n):\n    a, b = 0, 1\n    while b < n:\n        a, b = b, a + b')
get_ipython().run_cell_magic('cython', '', 'def fib3(int n):\n    cdef int b = 1\n    cdef int a = 0\n    cdef int t = 0\n    while b < n:\n        t = a\n        a = b\n        b = t + b')
get_ipython().run_line_magic('timeit', 'fib1(50)')
get_ipython().run_line_magic('timeit', 'fib2(50)')
get_ipython().run_line_magic('timeit', 'fib3(50)')
