# coding: utf-8
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
from fastai.nlp import *
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from torchtext import vocab, data, datasets
# ## IMBD dataset and the sentiment classification task
# The [large movie view dataset](http://ai.stanford.edu/~amaas/data/sentiment/) contains a collection of 50,000 reviews from IMDB. The dataset contains an even number of positive and negative reviews. The authors considered only highly polarized reviews. A negative review has a score ≤ 4 out of 10, and a positive review has a score ≥ 7 out of 10. Neutral reviews are not included in the dataset. The dataset is divided into training and test sets. The training set is the same 25,000 labeled reviews.
#
# The **sentiment classification task** consists of predicting the polarity (positive or negative) of a given text.
# To get the dataset, in your terminal run the following commands:
#
# `wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz`
#
# `gunzip aclImdb_v1.tar.gz`
#
# `tar -xvf aclImdb_v1.tar`
# ### Tokenizing
sl = 1000
vocab_size = 200000
PATH = 'data/aclImdb/'
names = ['neg', 'pos']
trn, trn_y = texts_labels_from_folders(f'{PATH}train', names)
val, val_y = texts_labels_from_folders(f'{PATH}test', names)
# Here is the text of the first review:
trn[0]
trn_y[0]
# [`CountVectorizer`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) converts a collection of text documents to a matrix of token counts (part of `sklearn.feature_extraction.text`). Here is how you specify parameters to the CountVectorizer. We will be working with the top 200000 unigrams, bigrams and trigrams.
veczr = CountVectorizer(ngram_range=(1, 3), tokenizer=tokenize, max_features=vocab_size)
# In the next line `fit_transform(trn)` computes the vocabulary and other hyparameters learned from the training set. It also transforms the training set. Since we have to apply the *same transformation* to your validation set, the second line uses just the method `transform(val)`. `trn_term_doc` and `val_term_doc` are sparse matrices. `trn_term_doc[i]` represents training document $i$ and it is binary (it has a $1$ for each vocabulary n-gram present in document $i$  and $0$ otherwise).
trn_term_doc = veczr.fit_transform(trn)
val_term_doc = veczr.transform(val)
trn_term_doc.shape
veczr.get_params()
# here is the vocabulary
vocab = veczr.get_feature_names()
vocab[50:55]
# ## Weighted Naive Bayes
# Our first model is a version of logistic regression with Naive Bayes features described [here](https://www.aclweb.org/anthology/P12-2018). For every document we compute binarized features as described above. Each feature if multiplied by a log-count ratio (see below for explanation). A logitic regression model is then trained to predict sentiment.
# Here is how to define **log-count ratio** for a feature $f$:
#
# $\text{log-count ratio} = \log \frac{\text{ratio of feature $f$ in positive documents}}{\text{ratio of feature $f$ in negative documents}}$
#
# where ratio of feature $f$ in positive documents is the number of times a positive document has a feature divided by the number of positive documents.
# Here is how we get a model from a bag of words
md = TextClassifierData.from_bow(trn_term_doc, trn_y, val_term_doc, val_y, sl)
learner = md.dotprod_nb_learner()
learner.fit(0.02, 1, wds=1e-5, cycle_len=1)
learner = md.dotprod_nb_learner()
learner.fit(0.02, 1, wds=1e-6)
# ### unigram
# Here is use `CountVectorizer` with a different set of parameters. In particular ngram_range by default is set to (1, 1)so we will get unigram features. Note that we are specifiying our own `tokenize` function.
veczr = CountVectorizer(tokenizer=tokenize)
trn_term_doc = veczr.fit_transform(trn)
val_term_doc = veczr.transform(val)
# Here is how to compute the $\text{log-count ratio}$ `r`.
x = trn_term_doc
y = trn_y
p = x[y == 1].sum(0) + 1
q = x[y == 0].sum(0) + 1
r = np.log((p / p.sum()) / (q / q.sum()))
b = np.log(len(p) / len(q))
# Here is the formula for Naive Bayes.
pre_preds = val_term_doc @ r.T + b
preds = pre_preds.T > 0
(preds == val_y).mean()
pre_preds = val_term_doc.sign() @ r.T + b
preds = pre_preds.T > 0
(preds == val_y).mean()
# Here is how we can fit regularized logistic regression where the features are the unigrams.
m = LogisticRegression(C=0.1, fit_intercept=False, dual=True)
m.fit(x, y)
preds = m.predict(val_term_doc)
(preds == val_y).mean()
# ### bigram with NB features
# Similar to the model before but with bigram features.
veczr = CountVectorizer(ngram_range=(1, 2), tokenizer=tokenize)
trn_term_doc = veczr.fit_transform(trn)
val_term_doc = veczr.transform(val)
y = trn_y
x = trn_term_doc.sign()
val_x = val_term_doc.sign()
p = x[y == 1].sum(0) + 1
q = x[y == 0].sum(0) + 1
r = np.log((p / p.sum()) / (q / q.sum()))
b = np.log(len(p) / len(q))
# Here we fit regularized logistic regression where the features are the bigrams. Bigrams are giving us 2% boost.
m = LogisticRegression(C=0.1, fit_intercept=False)
m.fit(x, y);
preds = m.predict(val_x)
(preds.T == val_y).mean()
# Here is the $\text{log-count ratio}$ `r`.
r
# Here we fit regularized logistic regression where the features are the bigrams multiplied by the $\text{log-count ratio}$. We are getting an extra boost for the normalization.
x_nb = x.multiply(r)
m = LogisticRegression(dual=True, C=1, fit_intercept=False)
m.fit(x_nb, y);
w = m.coef_.T
preds = (val_x_nb @ w + m.intercept_) > 0
(preds.T == val_y).mean()
# This is an interpolation between Naive Bayes the regulaized logistic regression approach.
beta = 0.25
val_x_nb = val_x.multiply(r)
w = (1 - beta) * m.coef_.mean() + beta * m.coef_.T
preds = (val_x_nb @ w + m.intercept_) > 0
(preds.T == val_y).mean()
w2 = w.T[0] * r.A1
preds = (val_x @ w2 + m.intercept_) > 0
(preds.T == val_y).mean()
# ## References
# * Baselines and Bigrams: Simple, Good Sentiment and Topic Classification. Sida Wang and Christopher D. Manning [pdf](https://www.aclweb.org/anthology/P12-2018)
# ### Unused helpers
class EzLSTM(nn.LSTM):
    def __init__(self, input_size, hidden_size, *args, **kwargs):
        super().__init__(input_size, hidden_size, *args, **kwargs)
        self.num_dirs = 2 if self.bidirectional else 1
        self.input_size = input_size
        self.hidden_size = hidden_size
        
    def forward(self, x):
        h0 = c0 = Variable(torch.zeros(self.num_dirs, x.size(1), self.hidden_size)).cuda()
        outp, _ = super().forward(x, (h0, c0))
        return outp[-1]
def init_wgts(m, last_l=-2):
    c = list(m.children())
    for l in c:
        if isinstance(l, nn.Embedding):
            l.weight.data.uniform_(-0.05, 0.05)
        elif isinstance(l, (nn.Linear, nn.Conv1d)):
            xavier_uniform(l.weight.data, gain=calculate_gain('relu'))
            l.bias.data.zero_()
    xavier_uniform(c[last_l].weight.data, gain=calculate_gain('linear'));
class SeqSize(nn.Sequential):
    def forward(self, x):
        for l in self.children():
            x = l(x)
            print(x.size())
        return x
# ### End
