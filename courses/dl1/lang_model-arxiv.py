
# coding: utf-8

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from fastai.model import fit
from fastai.dataset import *

import torchtext
from torchtext import vocab, data
from torchtext.datasets import language_modeling

from fastai.rnn_reg import *
from fastai.rnn_train import *
from fastai.nlp import *
from fastai.lm_rnn import *

import dill as pickle


bs, bptt = 64, 70


# ## Language modeling

# ### Data

PATH = '/data2/datasets/part1/arxiv/'

df_mb = pd.read_csv(f'{PATH}arxiv.csv')
df_all = pd.read_pickle(f'{PATH}all_arxiv.pickle')


def get_txt(df):
    return '<CAT> ' + df.category.str.replace(r'[\.\-]', '') + ' <SUMM> ' + df.summary + ' <TITLE> ' + df.title
df_mb['txt'] = get_txt(df_mb)
df_all['txt'] = get_txt(df_all)
n = len(df_all); n


os.makedirs(f'{PATH}trn/yes', exist_ok=True)
os.makedirs(f'{PATH}val/yes', exist_ok=True)
os.makedirs(f'{PATH}trn/no', exist_ok=True)
os.makedirs(f'{PATH}val/no', exist_ok=True)
os.makedirs(f'{PATH}all/trn', exist_ok=True)
os.makedirs(f'{PATH}all/val', exist_ok=True)
os.makedirs(f'{PATH}models', exist_ok=True)


for (i, (_, r)) in enumerate(df_all.iterrows()):
    dset = 'trn' if random.random() > 0.1 else 'val'
    open(f'{PATH}all/{dset}/{i}.txt', 'w').write(r['txt'])


for (i, (_, r)) in enumerate(df_mb.iterrows()):
    lbl = 'yes' if r.tweeted else 'no'
    dset = 'trn' if random.random() > 0.1 else 'val'
    open(f'{PATH}{dset}/{lbl}/{i}.txt', 'w').write(r['txt'])


from spacy.symbols import ORTH

my_tok = spacy.load('en')

my_tok.tokenizer.add_special_case('<SUMM>', [{ORTH: '<SUMM>'}])
my_tok.tokenizer.add_special_case('<CAT>', [{ORTH: '<CAT>'}])
my_tok.tokenizer.add_special_case('<TITLE>', [{ORTH: '<TITLE>'}])
my_tok.tokenizer.add_special_case('<BR />', [{ORTH: '<BR />'}])
my_tok.tokenizer.add_special_case('<BR>', [{ORTH: '<BR>'}])

def my_spacy_tok(x): return [tok.text for tok in my_tok.tokenizer(x)]


TEXT = data.Field(lower=True, tokenize=my_spacy_tok)
FILES = dict(train='trn', validation='val', test='val')
md = LanguageModelData(f'{PATH}all/', TEXT, **FILES, bs=bs, bptt=bptt, min_freq=10)
pickle.dump(TEXT, open(f'{PATH}models/TEXT.pkl', 'wb'))


len(md.trn_dl), md.nt, len(md.trn_ds), len(md.trn_ds[0].text)


TEXT.vocab.itos[:12]


' '.join(md.trn_ds[0].text[:150])


# ### Train

em_sz = 200
nh = 500
nl = 3
opt_fn = partial(optim.Adam, betas=(0.7, 0.99))


learner = md.get_model(opt_fn, em_sz, nh, nl,
    dropout=0.05, dropouth=0.1, dropouti=0.05, dropoute=0.02, wdrop=0.2)
# dropout=0.4, dropouth=0.3, dropouti=0.65, dropoute=0.1, wdrop=0.5
#                dropouti=0.05, dropout=0.05, wdrop=0.1, dropoute=0.02, dropouth=0.05)
learner.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
learner.clip = 0.3


learner.fit(3e-3, 1, wds=1e-6)


learner.fit(3e-3, 3, wds=1e-6, cycle_len=1, cycle_mult=2)


learner.save_encoder('adam2_enc')


learner.fit(3e-3, 10, wds=1e-6, cycle_len=5, cycle_save_name='adam3_10')


learner.save_encoder('adam3_10_enc')


learner.fit(3e-3, 8, wds=1e-6, cycle_len=10, cycle_save_name='adam3_5')


learner.fit(3e-3, 1, wds=1e-6, cycle_len=20, cycle_save_name='adam3_20')


learner.save_encoder('adam3_20_enc')


learner.save('adam3_20')


# ### Test

def proc_str(s): return TEXT.preprocess(TEXT.tokenize(s))
def num_str(s): return TEXT.numericalize([proc_str(s)])


m = learner.model


s = """<CAT> cscv <SUMM> algorithms that"""


def sample_model(m, s, l=50):
    t = num_str(s)
    m[0].bs = 1
    m.eval()
    m.reset()
    res, *_ = m(t)
    print('...', end='')

    for i in range(l):
        n = res[-1].topk(2)[1]
        n = n[1] if n.data[0] == 0 else n[0]
        word = TEXT.vocab.itos[n.data[0]]
        print(word, end=' ')
        if word == '<eos>': break
        res, *_ = m(n[0].unsqueeze(0))

    m[0].bs = bs


sample_model(m, "<CAT> csni <SUMM> algorithms that")


sample_model(m, "<CAT> cscv <SUMM> algorithms that")


sample_model(m, "<CAT> cscv <SUMM> algorithms. <TITLE> on ")


sample_model(m, "<CAT> csni <SUMM> algorithms. <TITLE> on ")


sample_model(m, "<CAT> cscv <SUMM> algorithms. <TITLE> towards ")


sample_model(m, "<CAT> csni <SUMM> algorithms. <TITLE> towards ")


# ### Sentiment

TEXT = pickle.load(open(f'{PATH}models/TEXT.pkl', 'rb'))


class ArxivDataset(torchtext.data.Dataset):
    def __init__(self, path, text_field, label_field, **kwargs):
        fields = [('text', text_field), ('label', label_field)]
        examples = []
        for label in ['yes', 'no']:
            for fname in glob(os.path.join(path, label, '*.txt')):
                with open(fname, 'r') as f: text = f.readline()
                examples.append(data.Example.fromlist([text, label], fields))
        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex): return len(ex.text)
    
    @classmethod
    def splits(cls, text_field, label_field, root='.data',
               train='train', test='test', **kwargs):
        return super().splits(
            root, text_field=text_field, label_field=label_field,
            train=train, validation=None, test=test, **kwargs)


ARX_LABEL = data.Field(sequential=False)
splits = ArxivDataset.splits(TEXT, ARX_LABEL, PATH, train='trn', test='val')


md2 = TextData.from_splits(PATH, splits, bs)


#            dropout=0.3, dropouti=0.4, wdrop=0.3, dropoute=0.05, dropouth=0.2)


from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

def prec_at_6(preds, targs):
    precision, recall, _ = precision_recall_curve(targs == 2, preds[:, 2])
    print(recall[precision >= 0.6][0])
    return recall[precision >= 0.6][0]


# dropout=0.4, dropouth=0.3, dropouti=0.65, dropoute=0.1, wdrop=0.5
m3 = md2.get_model(opt_fn, 1500, bptt, emb_sz=em_sz, n_hid=nh, n_layers=nl,
           dropout=0.1, dropouti=0.65, wdrop=0.5, dropoute=0.1, dropouth=0.3)
m3.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
m3.clip = 25.


m3.load_encoder(f'adam3_20_enc')
lrs = np.array([1e-4, 1e-3, 1e-2])


m3.freeze_to(-1)
m3.fit(lrs / 2, 1, metrics=[accuracy])
m3.unfreeze()
m3.fit(lrs, 1, metrics=[accuracy], cycle_len=1)


m3.fit(lrs, 2, metrics=[accuracy], cycle_len=4, cycle_save_name='imdb2')


prec_at_6(*m3.predict_with_targs())


m3.fit(lrs, 4, metrics=[accuracy], cycle_len=2, cycle_save_name='imdb2')


prec_at_6(*m3.predict_with_targs())
