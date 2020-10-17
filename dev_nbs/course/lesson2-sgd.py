# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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

# +
# %matplotlib inline

# fastai v1 backward compatibility
from matplotlib import animation, rc
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np

def tensor(*argv): return torch.tensor(argv)

# TEST
assert torch.all(tensor(1, 2) == torch.tensor([1, 2])), 'Backward compatibility with fastai v1'
# -

# In this part of the lecture we explain Stochastic Gradient Descent (SGD) which is an **optimization** method commonly used in neural networks. We will illustrate the concepts with concrete examples.

# #  Linear Regression problem

# The goal of linear regression is to fit a line to a set of points.

n = 100

x = torch.ones(n, 2)
x[:, 0].uniform_(-1., 1)
x[:5]

a = tensor(3., 2)
a

y = x@a + torch.rand(n)

plt.scatter(x[:, 0], y)


# You want to find **parameters** (weights) `a` such that you minimize the *error* between the points and the line `x@a`. Note that here `a` is unknown. For a regression problem the most common *error function* or *loss function* is the **mean squared error**.

def mse(y_hat, y): return ((y_hat - y)**2).mean()


# Suppose we believe `a = (-1.0,1.0)` then we can compute `y_hat` which is our *prediction* and then compute our error.

a = tensor(-1., 1)

y_hat = x@a
mse(y_hat, y)

plt.scatter(x[:, 0], y)
plt.scatter(x[:, 0], y_hat)

# So far we have specified the *model* (linear regression) and the *evaluation criteria* (or *loss function*). Now we need to handle *optimization*; that is, how do we find the best values for `a`? How do we find the best *fitting* linear regression.

# # Gradient Descent

# We would like to find the values of `a` that minimize `mse_loss`.
#
# **Gradient descent** is an algorithm that minimizes functions. Given a function defined by a set of parameters, gradient descent starts with an initial set of parameter values and iteratively moves toward a set of parameter values that minimize the function. This iterative minimization is achieved by taking steps in the negative direction of the function gradient.
#
# Here is gradient descent implemented in [PyTorch](http://pytorch.org/).

a = nn.Parameter(a)
a


def update():
    y_hat = x@a
    loss = mse(y, y_hat)
    if t % 10 == 0:
        print(loss)
    loss.backward()
    with torch.no_grad():
        a.sub_(lr * a.grad)
        a.grad.zero_()


lr = 1e-1
for t in range(100):
    update()

plt.scatter(x[:, 0], y)
plt.scatter(x[:, 0], (x@a).detach())

# ## Animate it!

rc('animation', html='jshtml')

# +
a = nn.Parameter(tensor(-1., 1))

fig = plt.figure()
plt.scatter(x[:, 0], y, c='orange')
line, = plt.plot(x[:, 0], (x@a).detach())
plt.close()

def animate(i):
    update()
    line.set_ydata((x@a).detach())
    return line,

animation.FuncAnimation(fig, animate, np.arange(0, 100), interval=20)
# -

# In practice, we don't calculate on the whole file at once, but we use *mini-batches*.

# ## Vocab

# - Learning rate
# - Epoch
# - Minibatch
# - SGD
# - Model / Architecture
# - Parameters
# - Loss function
#
# For classification problems, we use *cross entropy loss*, also known as *negative log likelihood loss*. This penalizes incorrect confident predictions, and correct unconfident predictions.
