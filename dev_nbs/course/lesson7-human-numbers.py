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

# # Human numbers

from fastai.text.all import *

bs = 64

# ## Data

path = untar_data(URLs.HUMAN_NUMBERS)
path.ls()


def readnums(d): return ', '.join(o.strip() for o in open(path / d).readlines())


train_txt = readnums('train.txt')
train_txt[:80]

valid_txt = readnums('valid.txt')
valid_txt[-80:]

train_tok = tokenize1(train_txt)
valid_tok = tokenize1(valid_txt)

dsets = Datasets([train_tok, valid_tok], tfms=Numericalize, dl_type=LMDataLoader, splits=[[0], [1]])

dls = dsets.dataloaders(bs=bs, val_bs=bs)

dsets.show((dsets.train[0][0][:80],))

len(dsets.valid[0][0])

len(dls.valid)

dls.seq_len, len(dls.valid)

13017 / 72 / bs

it = iter(dls.valid)
x1, y1 = next(it)
x2, y2 = next(it)
x3, y3 = next(it)
it.close()

x1.numel() + x2.numel() + x3.numel()

# This is the closes multiple of 64 below 13017

x1.shape, y1.shape

x2.shape, y2.shape

x1[0]

y1[0]

v = dls.vocab

' '.join([v[x] for x in x1[0]])

' '.join([v[x] for x in y1[0]])

' '.join([v[x] for x in x2[0]])

' '.join([v[x] for x in x3[0]])

' '.join([v[x] for x in x1[1]])

' '.join([v[x] for x in x2[1]])

' '.join([v[x] for x in x3[1]])

' '.join([v[x] for x in x3[-1]])

# ## Single fully connected model

dls = dsets.dataloaders(bs=bs, seq_len=3)

x, y = dls.one_batch()
x.shape, y.shape

nv = len(v)
nv

nh = 64


def loss4(input, target): return F.cross_entropy(input, target[:, -1])
def acc4(input, target): return accuracy(input, target[:, -1])


class Model0(Module):
    def __init__(self):
        self.i_h = nn.Embedding(nv, nh)  # green arrow
        self.h_h = nn.Linear(nh, nh)     # brown arrow
        self.h_o = nn.Linear(nh, nv)     # blue arrow
        self.bn = nn.BatchNorm1d(nh)

    def forward(self, x):
        h = self.bn(F.relu(self.h_h(self.i_h(x[:, 0]))))
        if x.shape[1] > 1:
            h = h + self.i_h(x[:, 1])
            h = self.bn(F.relu(self.h_h(h)))
        if x.shape[1] > 2:
            h = h + self.i_h(x[:, 2])
            h = self.bn(F.relu(self.h_h(h)))
        return self.h_o(h)


learn = Learner(dls, Model0(), loss_func=loss4, metrics=acc4)

learn.fit_one_cycle(6, 1e-4)


# ## Same thing with a loop

class Model1(Module):
    def __init__(self):
        self.i_h = nn.Embedding(nv, nh)  # green arrow
        self.h_h = nn.Linear(nh, nh)     # brown arrow
        self.h_o = nn.Linear(nh, nv)     # blue arrow
        self.bn = nn.BatchNorm1d(nh)

    def forward(self, x):
        h = torch.zeros(x.shape[0], nh).to(device=x.device)
        for i in range(x.shape[1]):
            h = h + self.i_h(x[:, i])
            h = self.bn(F.relu(self.h_h(h)))
        return self.h_o(h)


learn = Learner(dls, Model1(), loss_func=loss4, metrics=acc4)

learn.fit_one_cycle(6, 1e-4)

# ## Multi fully connected model

dls = dsets.dataloaders(bs=bs, seq_len=20)

x, y = dls.one_batch()
x.shape, y.shape


class Model2(Module):
    def __init__(self):
        self.i_h = nn.Embedding(nv, nh)
        self.h_h = nn.Linear(nh, nh)
        self.h_o = nn.Linear(nh, nv)
        self.bn = nn.BatchNorm1d(nh)

    def forward(self, x):
        h = torch.zeros(x.shape[0], nh).to(device=x.device)
        res = []
        for i in range(x.shape[1]):
            h = h + self.i_h(x[:, i])
            h = F.relu(self.h_h(h))
            res.append(self.h_o(self.bn(h)))
        return torch.stack(res, dim=1)


learn = Learner(dls, Model2(), loss_func=CrossEntropyLossFlat(), metrics=accuracy)

learn.fit_one_cycle(10, 1e-4, pct_start=0.1)


# ## Maintain state

class Model3(Module):
    def __init__(self):
        self.i_h = nn.Embedding(nv, nh)
        self.h_h = nn.Linear(nh, nh)
        self.h_o = nn.Linear(nh, nv)
        self.bn = nn.BatchNorm1d(nh)
        self.h = torch.zeros(bs, nh).cuda()

    def forward(self, x):
        res = []
        if x.shape[0] != self.h.shape[0]:
            self.h = torch.zeros(x.shape[0], nh).cuda()
        h = self.h
        for i in range(x.shape[1]):
            h = h + self.i_h(x[:, i])
            h = F.relu(self.h_h(h))
            res.append(self.bn(h))
        self.h = h.detach()
        res = torch.stack(res, dim=1)
        res = self.h_o(res)
        return res

    def reset(self): self.h = torch.zeros(bs, nh).cuda()


learn = Learner(dls, Model3(), metrics=accuracy, loss_func=CrossEntropyLossFlat())

learn.fit_one_cycle(20, 3e-3)


# ## nn.RNN

class Model4(Module):
    def __init__(self):
        self.i_h = nn.Embedding(nv, nh)
        self.rnn = nn.RNN(nh, nh, batch_first=True)
        self.h_o = nn.Linear(nh, nv)
        self.bn = BatchNorm1dFlat(nh)
        self.h = torch.zeros(1, bs, nh).cuda()

    def forward(self, x):
        if x.shape[0] != self.h.shape[1]:
            self.h = torch.zeros(1, x.shape[0], nh).cuda()
        res, h = self.rnn(self.i_h(x), self.h)
        self.h = h.detach()
        return self.h_o(self.bn(res))


learn = Learner(dls, Model4(), loss_func=CrossEntropyLossFlat(), metrics=accuracy)

learn.fit_one_cycle(20, 3e-3)


# ## 2-layer GRU

class Model5(Module):
    def __init__(self):
        self.i_h = nn.Embedding(nv, nh)
        self.rnn = nn.GRU(nh, nh, 2, batch_first=True)
        self.h_o = nn.Linear(nh, nv)
        self.bn = BatchNorm1dFlat(nh)
        self.h = torch.zeros(2, bs, nh).cuda()

    def forward(self, x):
        if x.shape[0] != self.h.shape[1]:
            self.h = torch.zeros(2, x.shape[0], nh).cuda()
        res, h = self.rnn(self.i_h(x), self.h)
        self.h = h.detach()
        return self.h_o(self.bn(res))


learn = Learner(dls, Model5(), loss_func=CrossEntropyLossFlat(), metrics=accuracy)

learn.fit_one_cycle(10, 1e-2)

# ## fin
