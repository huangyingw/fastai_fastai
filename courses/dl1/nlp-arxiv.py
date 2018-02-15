
# coding: utf-8

get_ipython().magic(u'reload_ext autoreload')
get_ipython().magic(u'autoreload 2')
get_ipython().magic(u'matplotlib inline')

from fastai.nlp import *
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from torchtext import vocab, data, datasets
import pandas as pd


sl = 1000
vocab_size = 200000


PATH = '/data2/datasets/part1/arxiv/arxiv.csv'

df = pd.read_csv(PATH)
df.head()


df['txt'] = df.category + ' ' + df.title + '\n' + df.summary


print(df.iloc[0].txt)


n = len(df); n


val_idx = get_cv_idxs(n, val_pct=0.1)
((val, trn), (val_y, trn_y)) = split_by_idx(val_idx, df.txt.values, df.tweeted.values)


# ## Ngram logistic regression

veczr = CountVectorizer(ngram_range=(1, 3), tokenizer=tokenize)
trn_term_doc = veczr.fit_transform(trn)
val_term_doc = veczr.transform(val)
trn_term_doc.shape, trn_term_doc.sum()


y = trn_y
x = trn_term_doc.sign()
val_x = val_term_doc.sign()


p = x[np.argwhere(y != 0)[:, 0]].sum(0) + 1
q = x[np.argwhere(y == 0)[:, 0]].sum(0) + 1
r = np.log((p / p.sum()) / (q / q.sum()))
b = np.log(len(p) / len(q))


pre_preds = val_term_doc @ r.T + b
preds = pre_preds.T > 0
(preds == val_y).mean()


m = LogisticRegression(C=0.1, fit_intercept=False)
m.fit(x, y);

preds = m.predict(val_x)
(preds.T == val_y).mean()


probs = m.predict_proba(val_x)[:, 1]


from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

precision, recall, _ = precision_recall_curve(val_y, probs)
average_precision = average_precision_score(val_y, probs)

plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve: AUC={0:0.2f}'.format(average_precision));


recall[precision >= 0.6][0]


df_val = df.iloc[sorted(val_idx)]


incorrect_yes = np.where((preds != val_y) & (val_y == 0))[0]
most_incorrect_yes = np.argsort(-probs[incorrect_yes])
txts = df_val.iloc[incorrect_yes[most_incorrect_yes[:10]]]
txts[["link", "title", "summary"]]


' '.join(txts.link.values)


incorrect_no = np.where((preds != val_y) & (val_y == 1))[0]
most_incorrect_no = np.argsort(probs[incorrect_no])
txts = df_val.iloc[incorrect_no[most_incorrect_no[:10]]]


txts[["link", "title", "summary"]]


' '.join(txts.link.values)


to_review = np.where((preds > 0.8) & (val_y == 0))[0]
to_review_idx = np.argsort(-probs[to_review])
txts = df_val.iloc[to_review[to_review_idx]]


txt_html = ('<li><a href="http://' + txts.link + '">' + txts.title.str.replace('\n', ' ') + '</a>: '
    + txts.summary.str.replace('\n', ' ') + '</li>').values


full_html = (f"""<!DOCTYPE html>
<html>
<head><title>Brundage Bot Backfill</title></head>
<body>
<ul>
{os.linesep.join(txt_html)}
</ul>
</body>
</html>""")


# ## Learner

veczr = CountVectorizer(ngram_range=(1, 3), tokenizer=tokenize, max_features=vocab_size)

trn_term_doc = veczr.fit_transform(trn)
val_term_doc = veczr.transform(val)


trn_term_doc.shape, trn_term_doc.sum()


md = TextClassifierData.from_bow(trn_term_doc, trn_y, val_term_doc, val_y, sl)


learner = md.dotprod_nb_learner(r_adj=20)


learner.fit(0.02, 4, wds=1e-6, cycle_len=1)


from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

def prec_at_6(preds, targs):
    precision, recall, _ = precision_recall_curve(targs[:, 1], preds[:, 1])
    return recall[precision >= 0.6][0]


prec_at_6(*learner.predict_with_targs())
