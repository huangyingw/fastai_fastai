
# coding: utf-8

# ## AdamW/SGDW benchmarking

# This is to benchmark an implementation of https://arxiv.org/abs/1711.05101

PATH = "/home/as/datasets/fastai.cifar10/cifar10/"


get_ipython().magic('matplotlib inline')
get_ipython().magic('reload_ext autoreload')
get_ipython().magic('autoreload 2')


# This file contains all the main external libs we'll use
from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# ### Common stuff for all experiments

def Get_SGD_Momentum(momentum=0.9):
    return lambda *args, **kwargs: optim.SGD(*
                                             args, momentum=momentum, **kwargs)


def Get_Adam():
    return lambda *args, **kwargs: optim.Adam(*args, **kwargs)


import pickle


def save_list(fname, l):
    with open(fname, "wb") as fp:
        pickle.dump(l, fp)


def read_list(fname):
    with open(fname, "rb") as fp:
        return pickle.load(fp)


# ### Section 1: Plot loss trends of various scenarios

# ### This is a common function which does the training.
#
# The only thing it asks for is the optimizer, and the initial LR for that
# optimizer. Hence we are comparing optimizers keeping all things same.

def experiment(optimizer, PATH, lr=1e-3, find_lr=False, use_wd_sched=False, wds=None, do_unfreeze=False,
               norm_wds=False, wds_sched_mult=None):
    sz = 32
    bs = 120
    arch = resnet152
    cycle_len = 2
    cycle_mult = 2
    num_cycles = 4
    lr = lr

    if wds is None:
        weight_decay = 0.025  # As used in the paper https://arxiv.org/abs/1711.05101
    else:
        weight_decay = wds

    tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
    data = ImageClassifierData.from_paths(PATH, tfms=tfms, bs=bs)
    learn = ConvLearner.pretrained(
        arch, data, precompute=True, xtra_fc=[
            1024, 512], opt_fn=optimizer)

    if find_lr:
        lrf = learn.lr_find()
        learn.sched.plot()
        return

    learn.fit(lr, 1, wds=weight_decay, use_wd_sched=use_wd_sched,
              norm_wds=norm_wds, wds_sched_mult=wds_sched_mult)
    print('Now with precompute as False')
    if do_unfreeze:
        learn.unfreeze()

    learn.precompute = False
    learn.fit(lr, num_cycles, wds=weight_decay, use_wd_sched=use_wd_sched, cycle_len=cycle_len,
              cycle_mult=cycle_mult, norm_wds=norm_wds, wds_sched_mult=wds_sched_mult)

    loss = learn.sched.losses
    fig = plt.figure(figsize=(10, 5))
    plt.plot(loss)
    plt.show()
    learn.sched.plot_lr()

    return learn.sched.losses, learn


def f(x): return np.array(x.layer_opt.lrs) / x.init_lrs


# ### SGDR/SGDW

get_ipython().run_cell_magic(
    'time',
    '',
    'sgdm = Get_SGD_Momentum()\nloss_sgdm = experiment(sgdm, PATH=PATH, find_lr=True)')


get_ipython().run_cell_magic(
    'time',
    '',
    "# Without weight decay\nsgdm = Get_SGD_Momentum()\nloss_sgdm, _ = experiment(sgdm, PATH=PATH, lr=1e-2)\nsave_list('sgdm_loss.txt', loss_sgdm)")


get_ipython().run_cell_magic('time', '', "# With weight decay\nsgdmw = Get_SGD_Momentum()\nloss_sgdmw, _ = experiment(sgdmw, PATH=PATH, lr=1e-2, use_wd_sched=True, norm_wds=True, wds_sched_mult=f)\nsave_list('sgdmw_loss.txt', loss_sgdmw)")


# ### Adam/AdamW

adam = Get_Adam()
loss_adam = experiment(adam, PATH, find_lr=True)


# **Train**

get_ipython().run_cell_magic(
    'time',
    '',
    "# Without weight decay\nadam = Get_Adam()\nloss_adam, _ = experiment(adam, PATH, 1e-3)\nsave_list('adam_loss.txt', loss_adam)")


get_ipython().run_cell_magic(
    'time',
    '',
    "# With weight decay\nadamw = Get_Adam()\nloss_adamw, _ = experiment(adamw, PATH, 1e-3, use_wd_sched=True, norm_wds=True, wds_sched_mult=f)\nsave_list('adamw_loss.txt', loss_adamw)")


# ### Differential Learning and Weight Decay

get_ipython().run_cell_magic(
    'time',
    '',
    "adamw_diff = Get_Adam()\nlr = 1e-3\nwd = 0.025\nloss_adamw_diff, _ = experiment(adamw_diff, PATH, [lr/10, lr/5, lr], wds=[wd/10, wd/5, wd], use_wd_sched=True, norm_wds=True, wds_sched_mult=f)\nsave_list('loss_adamw_diff.txt', loss_adamw_diff)")


get_ipython().run_cell_magic(
    'time',
    '',
    "sgdw_diff = Get_SGD_Momentum(0.9)\nlr = 1e-2\nwd = 0.025\nloss_sgdw_diff, _ = experiment(sgdw_diff, PATH, [lr/10, lr/5, lr], wds=[wd/10, wd/5, wd], use_wd_sched=True, norm_wds=True, wds_sched_mult=f)\nsave_list('loss_sgdw_diff.txt', loss_sgdw_diff)")


fig = plt.figure(figsize=(15, 10))
plt.plot(loss_adam, c='red', label='Adam')
plt.plot(loss_sgdm, c='blue', label='SGDM')
plt.plot(loss_adamw, c='green', label='AdamW')
plt.plot(loss_sgdmw, c='black', label='SGDW')
plt.plot(loss_adamw_diff, c='orange', label='AdamW_differential')
plt.plot(loss_sgdw_diff, c='gray', label='SGDW_differential')
plt.legend()
plt.show()


# ### Section 2: Check for regularization of overfitting

import time


def check_overfitting(optimizer, PATH, sz, bs, lr, wds, use_wd_sched=True):

    arch = resnet50
    cycle_len = 12
    cycle_mult = 2
    num_cycles = 1

    # aug_tfms=transforms_side_on, max_zoom=1.1
    tfms = tfms_from_model(arch, sz)
    data = ImageClassifierData.from_paths(PATH, tfms=tfms, bs=bs)
    # Turning off Dropout, adding 3 extra FC layers to help in overfitting.
    learn = ConvLearner.pretrained(
        arch, data, precompute=False, xtra_fc=[
            1024, 512], ps=[
            0, 0, 0], opt_fn=optimizer)

    print("==== Let's overfit the model. Expectation: training loss should reduce but validation loss should stagnate.")
    learn.unfreeze()
    learn.fit(lr, num_cycles, cycle_len=cycle_len, cycle_mult=cycle_mult)
    print("==== Let's introduce weight regularization. Expectation: training loss and validation loss should reduce. Accuracy should improve.")
    learn.fit(
        lr,
        num_cycles,
        wds=wds,
        use_wd_sched=use_wd_sched,
        cycle_len=cycle_len,
        cycle_mult=cycle_mult)
    time.sleep(5)


# **AdamW on Cats & Dogs**

get_ipython().run_cell_magic(
    'time',
    '',
    'PATH = "/home/as/datasets/fastai.dogscats/"\ncheck_overfitting(Get_Adam(), PATH=PATH, sz=224, bs=96, lr=1e-3, wds=0.25, use_wd_sched=True)')


# **AdamW on Cifar10**

get_ipython().run_cell_magic(
    'time',
    '',
    'PATH = "/home/as/datasets/fastai.cifar10/cifar10/"\ncheck_overfitting(Get_Adam(), PATH=PATH, sz=32, bs=320, lr=1e-3, wds=0.45, use_wd_sched=True)')


# **Adam on Cifar10**

get_ipython().run_cell_magic(
    'time',
    '',
    'PATH = "/home/as/datasets/fastai.cifar10/cifar10/"\ncheck_overfitting(Get_Adam(), PATH=PATH, sz=32, bs=320, lr=1e-3, wds=0.45, use_wd_sched=False)')


# ### Section 3: Tests after splitting the various parts of the regularizer

# Separate weight regularization

get_ipython().run_cell_magic(
    'time',
    '',
    "# With weight decay\nadamw = Get_Adam()\nlr = [1e-5,1e-4,1e-3]\nwd = [1e-7,1e-6,1e-5]\nloss_adamw1, learn1 = experiment(adamw, PATH, lr, use_wd_sched=True, wds=wd, do_unfreeze=True)\nsave_list('adamw_loss1.txt', loss_adamw1)")


# With weight normalization

get_ipython().run_cell_magic(
    'time',
    '',
    "# With weight decay\nadamw = Get_Adam()\nlr = [1e-5,1e-4,1e-3]\nwd = [1e-5,1e-4,1e-3]\nloss_adamw2, learn2 = experiment(adamw, PATH, lr, use_wd_sched=True, wds=wd, norm_wds=True, do_unfreeze=True)\nsave_list('adamw_loss2.txt', loss_adamw2)")


# With custom weight multiplier wds_sched_mult

get_ipython().run_cell_magic(
    'time',
    '',
    "# With weight decay\nadamw = Get_Adam()\nlr = [1e-5,1e-4,1e-3]\nwd = [1e-5,1e-4,1e-3]\nf = lambda x: np.array(x.layer_opt.lrs) / x.init_lrs\nloss_adamw3, learn3 = experiment(adamw, PATH, lr, use_wd_sched=True, wds=wd, norm_wds=True, wds_sched_mult=f, do_unfreeze=True)\nsave_list('adamw_loss3.txt', loss_adamw3)")


fig = plt.figure(figsize=(15, 10))
plt.plot(loss_adamw1)
plt.plot(loss_adamw2)
plt.plot(loss_adamw3)
plt.show()


get_ipython().run_cell_magic(
    'time',
    '',
    "# With weight decay\nadamw = Get_Adam()\nlr = [1e-5,1e-4,1e-3]\nwd = [1e-5,1e-4,1e-3]\nloss_adamw4, learn4 = experiment(adamw, PATH, lr, use_wd_sched=True, wds=wd, do_unfreeze=True)\nsave_list('adamw_loss4.txt', loss_adamw4)")
