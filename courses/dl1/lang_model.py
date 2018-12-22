# coding: utf-8
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
from fastai.imports import *
from fastai.torch_imports import *
from fastai.core import *
from fastai.model import fit
from fastai.dataset import *
import torchtext
from torchtext import vocab, data
from torchtext.datasets import language_modeling
from fastai.rnn_reg import *
from fastai.rnn_train import *
from fastai.nlp import *
from fastai.lm_rnn import *
# ## Language modeling
# ### Data
PATH = 'data/wikitext-2/'
get_ipython().run_line_magic('ls', '{PATH}')
get_ipython().system('head -5 {PATH}wiki.train.tokens')
get_ipython().system('wc -lwc {PATH}wiki.train.tokens')
get_ipython().system('wc -lwc {PATH}wiki.valid.tokens')
TEXT = data.Field(lower=True)
FILES = dict(train='wiki.train.tokens', validation='wiki.valid.tokens', test='wiki.test.tokens')
bs, bptt = 80, 70
md = LanguageModelData(PATH, TEXT, **FILES, bs=bs, bptt=bptt, min_freq=10)
len(md.trn_dl), md.nt, len(md.trn_ds), len(md.trn_ds[0].text)
#md.trn_ds[0].text[:12], next(iter(md.trn_dl))
# ### Train
em_sz = 200
nh = 500
nl = 3
learner = md.get_model(SGD_Momentum(0.7), bs, em_sz, nh, nl)
reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
clip = 0.3
learner.fit(10, 1, wds=1e-6, reg_fn=reg_fn, clip=clip)
learner.fit(10, 6, wds=1e-6, reg_fn=reg_fn, cycle_len=1, cycle_mult=2, clip=clip)
learner.save('lm_420')
learner.fit(10, 6, wds=1e-6, reg_fn=reg_fn, cycle_len=1, cycle_mult=2, clip=clip)
learner.save('lm_419')
learner.fit(10, 6, wds=1e-6, reg_fn=reg_fn, cycle_len=1, cycle_mult=2, clip=clip)
learner.save('lm_418')
math.exp(4.17)
# ### Test
m = learner.model
s = [""". <eos> The game began development in 2010 , carrying over a large portion of the work 
done on Valkyria Chronicles II . While it retained the standard features of """.split()]
t = TEXT.numericalize(s)
m[0].bs = 1
m.reset(False)
res, *_ = m(t)
nexts = torch.topk(res[-1], 10)[1]
[TEXT.vocab.itos[o] for o in to_np(nexts)]
for i in range(20):
    n = res[-1].topk(2)[1]
    n = n[1] if n.data[0] == 0 else n[0]
    print(TEXT.vocab.itos[n.data[0]], end=' ')
    res, *_ = m(n[0].unsqueeze(0))
# ### End
