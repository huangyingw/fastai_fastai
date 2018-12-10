
# coding: utf-8

# get_ipython().run_line_magic('reload_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
# get_ipython().run_line_magic('matplotlib', 'inline')

from fastai.learner import *

import torchtext
from torchtext import data

from fastai.rnn_reg import *
from fastai.rnn_train import *
from fastai.nlp import *
from fastai.lm_rnn import *

import dill as pickle
import spacy


# ## Language modeling

# ### Data

# The [large movie view dataset](http://ai.stanford.edu/~amaas/data/sentiment/) contains a collection of 50,000 reviews from IMDB. The dataset contains an even number of positive and negative reviews. The authors considered only highly polarized reviews. A negative review has a score ≤ 4 out of 10, and a positive review has a score ≥ 7 out of 10. Neutral reviews are not included in the dataset. The dataset is divided into training and test sets. The training set is the same 25,000 labeled reviews.
#
# The **sentiment classification task** consists of predicting the polarity (positive or negative) of a given text.
#
# However, before we try to classify *sentiment*, we will simply try to create a *language model*; that is, a model that can predict the next word in a sentence. Why? Because our model first needs to understand the structure of English, before we can expect it to recognize positive vs negative sentiment.
#
# So our plan of attack is the same as we used for Dogs v Cats: pretrain a model to do one thing (predict the next word), and fine tune it to do something else (classify sentiment).
#
# Unfortunately, there are no good pretrained language models available to download, so we need to create our own. To follow along with this notebook, we suggest downloading the dataset from [this location](http://files.fast.ai/data/aclImdb.tgz) on files.fast.ai.

os.chdir(os.path.dirname(os.path.realpath(__file__)))
PATH = 'data/aclImdb/'

TRN_PATH = 'train/all/'
VAL_PATH = 'test/all/'
TRN = f'{PATH}{TRN_PATH}'
VAL = f'{PATH}{VAL_PATH}'

# get_ipython().run_line_magic('ls', '{PATH}')


# Let's look inside the training folder...

trn_files = get_ipython().getoutput('ls {TRN}')
trn_files[:10]


# ...and at an example review.

review = get_ipython().getoutput('cat {TRN}{trn_files[6]}')
review[0]


# Sounds like I'd really enjoy *Zombiegeddon*...
#
# Now we'll check how many words are in the dataset.

# get_ipython().system("find {TRN} -name '*.txt' | xargs cat | wc -w")


# get_ipython().system("find {VAL} -name '*.txt' | xargs cat | wc -w")


# Before we can analyze text, we must first *tokenize* it. This refers to the process of splitting a sentence into an array of words (or more generally, into an array of *tokens*).

# *Note:* If you get an error like:
#
#     Can't find model 'en'. It doesn't seem to be a shortcut link, a Python package or a valid path to a data directory.
#
# then you need to install the Spacy language model by running this command on the command-line:
#
#     $ python -m spacy download en

spacy_tok = spacy.load('en')


' '.join([sent.string.strip() for sent in spacy_tok(review[0])])


# We use Pytorch's [torchtext](https://github.com/pytorch/text) library to preprocess our data, telling it to use the wonderful [spacy](https://spacy.io/) library to handle tokenization.
#
# First, we create a torchtext *field*, which describes how to preprocess a piece of text - in this case, we tell torchtext to make everything lowercase, and tokenize it with spacy.

TEXT = data.Field(lower=True, tokenize="spacy")


# fastai works closely with torchtext. We create a ModelData object for language modeling by taking advantage of `LanguageModelData`, passing it our torchtext field object, and the paths to our training, test, and validation sets. In this case, we don't have a separate test set, so we'll just use `VAL_PATH` for that too.
#
# As well as the usual `bs` (batch size) parameter, we also now have `bptt`; this define how many words are processing at a time in each row of the mini-batch. More importantly, it defines how many 'layers' we will backprop through. Making this number higher will increase time and memory requirements, but will improve the model's ability to handle long sentences.

bs = 64; bptt = 70


FILES = dict(train=TRN_PATH, validation=VAL_PATH, test=VAL_PATH)
md = LanguageModelData.from_text_files(PATH, TEXT, **FILES, bs=bs, bptt=bptt, min_freq=10)


# After building our `ModelData` object, it automatically fills the `TEXT` object with a very important attribute: `TEXT.vocab`. This is a *vocabulary*, which stores which words (or *tokens*) have been seen in the text, and how each word will be mapped to a unique integer id. We'll need to use this information again later, so we save it.
#
# *(Technical note: python's standard `Pickle` library can't handle this correctly, so at the top of this notebook we used the `dill` library instead and imported it as `pickle`)*.

pickle.dump(TEXT, open(f'{PATH}models/TEXT.pkl', 'wb'))


# Here are the: # batches; # unique tokens in the vocab; # tokens in the training set; # sentences

len(md.trn_dl), md.nt, len(md.trn_ds), len(md.trn_ds[0].text)


# This is the start of the mapping from integer IDs to unique tokens.

# 'itos': 'int-to-string'
TEXT.vocab.itos[:12]


# 'stoi': 'string to int'
TEXT.vocab.stoi['the']


# Note that in a `LanguageModelData` object there is only one item in each dataset: all the words of the text joined together.

md.trn_ds[0].text[:12]


# torchtext will handle turning this words into integer IDs for us automatically.

TEXT.numericalize([md.trn_ds[0].text[:12]])


# Our `LanguageModelData` object will create batches with 64 columns (that's our batch size), and varying sequence lengths of around 80 tokens (that's our `bptt` parameter - *backprop through time*).
#
# Each batch also contains the exact same data as labels, but one word later in the text - since we're trying to always predict the next word. The labels are flattened into a 1d array.

next(iter(md.trn_dl))


# ### Train

# We have a number of parameters to set - we'll learn more about these later, but you should find these values suitable for many problems.

em_sz = 200  # size of each embedding vector
nh = 500     # number of hidden activations per layer
nl = 3       # number of layers


# Researchers have found that large amounts of *momentum* (which we'll learn about later) don't work well with these kinds of *RNN* models, so we create a version of the *Adam* optimizer with less momentum than it's default of `0.9`.

opt_fn = partial(optim.Adam, betas=(0.7, 0.99))


# fastai uses a variant of the state of the art [AWD LSTM Language Model](https://arxiv.org/abs/1708.02182) developed by Stephen Merity. A key feature of this model is that it provides excellent regularization through [Dropout](https://en.wikipedia.org/wiki/Convolutional_neural_network#Dropout). There is no simple way known (yet!) to find the best values of the dropout parameters below - you just have to experiment...
#
# However, the other parameters (`alpha`, `beta`, and `clip`) shouldn't generally need tuning.

learner = md.get_model(opt_fn, em_sz, nh, nl,
               dropouti=0.05, dropout=0.05, wdrop=0.1, dropoute=0.02, dropouth=0.05)
learner.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
learner.clip = 0.3


# As you can see below, I gradually tuned the language model in a few stages. I possibly could have trained it further (it wasn't yet overfitting), but I didn't have time to experiment more. Maybe you can see if you can train it to a better accuracy! (I used `lr_find` to find a good learning rate, but didn't save the output in this notebook. Feel free to try running it yourself now.)

learner.fit(3e-3, 4, wds=1e-6, cycle_len=1, cycle_mult=2)


learner.save_encoder('adam1_enc')


learner.load_encoder('adam1_enc')


learner.fit(3e-3, 1, wds=1e-6, cycle_len=10)


# In the sentiment analysis section, we'll just need half of the language model - the *encoder*, so we save that part.

learner.save_encoder('adam3_10_enc')


learner.load_encoder('adam3_10_enc')


# Language modeling accuracy is generally measured using the metric *perplexity*, which is simply `exp()` of the loss function we used.

math.exp(4.165)


pickle.dump(TEXT, open(f'{PATH}models/TEXT.pkl', 'wb'))


# ### Test

# We can play around with our language model a bit to check it seems to be working OK. First, let's create a short bit of text to 'prime' a set of predictions. We'll use our torchtext field to numericalize it so we can feed it to our language model.

m = learner.model
ss = """. So, it wasn't quite was I was expecting, but I really liked it anyway! The best"""
s = [TEXT.preprocess(ss)]
t = TEXT.numericalize(s)
' '.join(s[0])


# We haven't yet added methods to make it easy to test a language model, so we'll need to manually go through the steps.

# Set batch size to 1
m[0].bs = 1
# Turn off dropout
m.eval()
# Reset hidden state
m.reset()
# Get predictions from model
res, *_ = m(t)
# Put the batch size back to what it was
m[0].bs = bs


# Let's see what the top 10 predictions were for the next word after our short text:

nexts = torch.topk(res[-1], 10)[1]
[TEXT.vocab.itos[o] for o in to_np(nexts)]


# ...and let's see if our model can generate a bit more text all by itself!

print(ss, "\n")
for i in range(50):
    n = res[-1].topk(2)[1]
    n = n[1] if n.data[0] == 0 else n[0]
    print(TEXT.vocab.itos[n.data[0]], end=' ')
    res, *_ = m(n[0].unsqueeze(0))
print('...')


# ### Sentiment

# We'll need to the saved vocab from the language model, since we need to ensure the same words map to the same IDs.

TEXT = pickle.load(open(f'{PATH}models/TEXT.pkl', 'rb'))


# `sequential=False` tells torchtext that a text field should be tokenized (in this case, we just want to store the 'positive' or 'negative' single label).
#
# `splits` is a torchtext method that creates train, test, and validation sets. The IMDB dataset is built into torchtext, so we can take advantage of that. Take a look at `lang_model-arxiv.ipynb` to see how to define your own fastai/torchtext datasets.

IMDB_LABEL = data.Field(sequential=False)
splits = torchtext.datasets.IMDB.splits(TEXT, IMDB_LABEL, 'data/')


t = splits[0].examples[0]


t.label, ' '.join(t.text[:16])


# fastai can create a ModelData object directly from torchtext splits.

md2 = TextData.from_splits(PATH, splits, bs)


m3 = md2.get_model(opt_fn, 1500, bptt, emb_sz=em_sz, n_hid=nh, n_layers=nl,
           dropout=0.1, dropouti=0.4, wdrop=0.5, dropoute=0.05, dropouth=0.3)
m3.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
m3.load_encoder(f'adam3_10_enc')


# Because we're fine-tuning a pretrained model, we'll use differential learning rates, and also increase the max gradient for clipping, to allow the SGDR to work better.

m3.clip = 25.
lrs = np.array([1e-4, 1e-4, 1e-4, 1e-3, 1e-2])


m3.freeze_to(-1)
m3.fit(lrs / 2, 1, metrics=[accuracy])
m3.unfreeze()
m3.fit(lrs, 1, metrics=[accuracy], cycle_len=1)


m3.fit(lrs, 7, metrics=[accuracy], cycle_len=2, cycle_save_name='imdb2')


m3.load_cycle('imdb2', 4)


accuracy_np(*m3.predict_with_targs())


# A recent paper from Bradbury et al, [Learned in translation: contextualized word vectors](https://einstein.ai/research/learned-in-translation-contextualized-word-vectors), has a handy summary of the latest academic research in solving this IMDB sentiment analysis problem. Many of the latest algorithms shown are tuned for this specific problem.
#
# ![image.png](attachment:image.png)
#
# As you see, we just got a new state of the art result in sentiment analysis, decreasing the error from 5.9% to 5.5%! You should be able to get similarly world-class results on other NLP classification problems using the same basic steps.
#
# There are many opportunities to further improve this, although we won't be able to get to them until part 2 of this course...

# ### End
