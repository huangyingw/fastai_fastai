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
from nbdev.export import notebook2script
from nbdev.showdoc import *
from fastai.tabular.data import *
from fastai.tabular.model import *
from fastai.tabular.core import *
from fastai.basics import *
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# export


# hide


# +
# default_exp tabular.learner
# -

# # Tabular learner
#
# > The function to immediately get a `Learner` ready to train for tabular data

# The main function you probably want to use in this module is `tabular_learner`. It will automatically create a `TabularModel` suitable for your data and infer the right loss function. See the [tabular tutorial](http://docs.fast.ai/tutorial.tabular) for an example of use in context.

# ## Main functions

# export
@log_args(but_as=Learner.__init__)
class TabularLearner(Learner):
    "`Learner` for tabular data"

    def predict(self, row):
        "Predict on a Pandas Series"
        dl = self.dls.test_dl(row.to_frame().T)
        dl.dataset.conts = dl.dataset.conts.astype(np.float32)
        inp, preds, _, dec_preds = self.get_preds(dl=dl, with_input=True, with_decoded=True)
        b = (*tuplify(inp), *tuplify(dec_preds))
        full_dec = self.dls.decode(b)
        return full_dec, dec_preds[0], preds[0]


show_doc(TabularLearner, title_level=3)


# It works exactly as a normal `Learner`, the only difference is that it implements a `predict` method specific to work on a row of data.

# export
@log_args(to_return=True, but_as=Learner.__init__)
@delegates(Learner.__init__)
def tabular_learner(dls, layers=None, emb_szs=None, config=None, n_out=None, y_range=None, **kwargs):
    "Get a `Learner` using `dls`, with `metrics`, including a `TabularModel` created using the remaining params."
    if config is None:
        config = tabular_config()
    if layers is None:
        layers = [200, 100]
    to = dls.train_ds
    emb_szs = get_emb_sz(dls.train_ds, {} if emb_szs is None else emb_szs)
    if n_out is None:
        n_out = get_c(dls)
    assert n_out, "`n_out` is not defined, and could not be inferred from data, set `dls.c` or pass `n_out`"
    if y_range is None and 'y_range' in config:
        y_range = config.pop('y_range')
    model = TabularModel(emb_szs, len(dls.cont_names), n_out, layers, y_range=y_range, **config)
    return TabularLearner(dls, model, **kwargs)


# If your data was built with fastai, you probably won't need to pass anything to `emb_szs` unless you want to change the default of the library (produced by `get_emb_sz`), same for `n_out` which should be automatically inferred. `layers` will default to `[200,100]` and is passed to `TabularModel` along with the `config`.
#
# Use `tabular_config` to create a `config` and customize the model used. There is just easy access to `y_range` because this argument is often used.
#
# All the other arguments are passed to `Learner`.

path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path / 'adult.csv')
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
cont_names = ['age', 'fnlwgt', 'education-num']
procs = [Categorify, FillMissing, Normalize]
dls = TabularDataLoaders.from_df(df, path, procs=procs, cat_names=cat_names, cont_names=cont_names,
                                 y_names="salary", valid_idx=list(range(800, 1000)), bs=64)
learn = tabular_learner(dls)

show_doc(TabularLearner.predict)

# We can pass in an individual row of data into our `TabularLearner`'s `predict` method. It's output is slightly different from the other `predict` methods, as this one will always return the input as well:

row, clas, probs = learn.predict(df.iloc[0])

row.show()

clas, probs

# +
# hide
# test y_range is passed
learn = tabular_learner(dls, y_range=(0, 32))
assert isinstance(learn.model.layers[-1], SigmoidRange)
test_eq(learn.model.layers[-1].low, 0)
test_eq(learn.model.layers[-1].high, 32)

learn = tabular_learner(dls, config=tabular_config(y_range=(0, 32)))
assert isinstance(learn.model.layers[-1], SigmoidRange)
test_eq(learn.model.layers[-1].low, 0)
test_eq(learn.model.layers[-1].high, 32)


# -

# export
@typedispatch
def show_results(x: Tabular, y: Tabular, samples, outs, ctxs=None, max_n=10, **kwargs):
    df = x.all_cols[:max_n]
    for n in x.y_names:
        df[n + '_pred'] = y[n][:max_n].values
    display_df(df)


# ## Export -

# hide
notebook2script()
