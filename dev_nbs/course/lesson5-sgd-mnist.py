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

# %matplotlib inline
from fastai.callback.all import *
from torch.utils.data import TensorDataset
from fastai.basics import *
import gzip

# ## MNIST SGD

# Get the 'pickled' MNIST dataset from http://deeplearning.net/data/mnist/mnist.pkl.gz. We're going to treat it as a standard flat dataset with fully connected layers, rather than using a CNN.

path = Config().data / 'mnist'

path.ls()

with gzip.open(path / 'mnist.pkl.gz', 'rb') as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')

plt.imshow(x_train[0].reshape((28, 28)), cmap="gray")
x_train.shape

x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))
n, c = x_train.shape
x_train.shape, y_train.min(), y_train.max()

# In lesson2-sgd we did these things ourselves:
#
# ```python
# x = torch.ones(n,2)
# def mse(y_hat, y): return ((y_hat-y)**2).mean()
# y_hat = x@a
# ```
#
# Now instead we'll use PyTorch's functions to do it for us, and also to handle mini-batches (which we didn't do last time, since our dataset was so small).


bs = 64
train_ds = TensorDataset(x_train, y_train)
valid_ds = TensorDataset(x_valid, y_valid)
train_dl = TfmdDL(train_ds, bs=bs, shuffle=True)
valid_dl = TfmdDL(valid_ds, bs=2 * bs)
dls = DataLoaders(train_dl, valid_dl)

x, y = dls.one_batch()
x.shape, y.shape


class Mnist_Logistic(Module):
    def __init__(self): self.lin = nn.Linear(784, 10, bias=True)
    def forward(self, xb): return self.lin(xb)


model = Mnist_Logistic().cuda()

model

model.lin

model(x).shape

[p.shape for p in model.parameters()]

lr = 2e-2

loss_func = nn.CrossEntropyLoss()


def update(x, y, lr):
    wd = 1e-5
    y_hat = model(x)
    # weight decay
    w2 = 0.
    for p in model.parameters():
        w2 += (p**2).sum()
    # add to regular loss
    loss = loss_func(y_hat, y) + w2 * wd
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            p.sub_(lr * p.grad)
            p.grad.zero_()
    return loss.item()


losses = [update(x, y, lr) for x, y in dls.train]

plt.plot(losses)


class Mnist_NN(Module):
    def __init__(self):
        self.lin1 = nn.Linear(784, 50, bias=True)
        self.lin2 = nn.Linear(50, 10, bias=True)

    def forward(self, xb):
        x = self.lin1(xb)
        x = F.relu(x)
        return self.lin2(x)


model = Mnist_NN().cuda()

losses = [update(x, y, lr) for x, y in dls.train]

plt.plot(losses)

model = Mnist_NN().cuda()


def update(x, y, lr):
    opt = torch.optim.Adam(model.parameters(), lr)
    y_hat = model(x)
    loss = loss_func(y_hat, y)
    loss.backward()
    opt.step()
    opt.zero_grad()
    return loss.item()


losses = [update(x, y, 1e-3) for x, y in dls.train]

plt.plot(losses)

learn = Learner(dls, Mnist_NN(), loss_func=loss_func, metrics=accuracy)


learn.lr_find()

learn.fit_one_cycle(1, 1e-2)

learn.recorder.plot_sched()

learn.recorder.plot_loss()

# ## fin
