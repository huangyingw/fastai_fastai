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
from fastai.text.all import *
from fastai.callback.all import *
from fastai.basics import *
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab


# +
# all_slow
# -

# # Tutorial - Assemble the data on the wikitext dataset
#
# > Using `Datasets`, `Pipeline`, `TfmdLists` and `Transform` in text

# In this tutorial, we explore the mid-level API for data collection in the text application. We will use the bases introduced in the [pets tutorial](http://docs.fast.ai/tutorial.pets) so you should be familiar with `Transform`, `Pipeline`, `TfmdLists` and `Datasets` already.

# ## Data

path = untar_data(URLs.WIKITEXT_TINY)

# The dataset comes with the articles in two csv files, so we read it and concatenate them in one dataframe.

df_train = pd.read_csv(path / 'train.csv', header=None)
df_valid = pd.read_csv(path / 'test.csv', header=None)
df_all = pd.concat([df_train, df_valid])

df_all.head()

# We could tokenize it based on spaces to compare (as is usually done) but here we'll use the standard fastai tokenizer.

splits = [list(range_of(df_train)), list(range(len(df_train), len(df_all)))]
tfms = [attrgetter("text"), Tokenizer.from_df(0), Numericalize()]
dsets = Datasets(df_all, [tfms], splits=splits, dl_type=LMDataLoader)

bs, sl = 104, 72
dls = dsets.dataloaders(bs=bs, seq_len=sl)

dls.show_batch(max_n=3)

# ## Model

config = awd_lstm_lm_config.copy()
config.update({'input_p': 0.6, 'output_p': 0.4, 'weight_p': 0.5, 'embed_p': 0.1, 'hidden_p': 0.2})
model = get_language_model(AWD_LSTM, len(dls.vocab), config=config)

opt_func = partial(Adam, wd=0.1, eps=1e-7)
cbs = [MixedPrecision(), GradientClip(0.1)] + rnn_cbs(alpha=2, beta=1)

learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), opt_func=opt_func, cbs=cbs, metrics=[accuracy, Perplexity()])

learn.fit_one_cycle(1, 5e-3, moms=(0.8, 0.7, 0.8), div=10)

# +
#learn.fit_one_cycle(90, 5e-3, moms=(0.8,0.7,0.8), div=10)
# -
