
# coding: utf-8

# # TrainingPhase and General scheduler

# Creates a scheduler that lets you train a model with following different [`TrainingPhase`](/callbacks.general_sched.html#TrainingPhase).

from fastai.gen_doc.nbdoc import *
from fastai.callbacks.general_sched import *
from fastai import *
from fastai.vision import *


show_doc(TrainingPhase, doc_string=False)


# Create a phase for training a model during `length` iterations, following a schedule given by `lrs` and `lr_anneal`, `moms` and `mom_anneal`. More specifically, the phase will make the learning rate (or momentum) vary from the first value of `lrs` (or `moms`) to the second, following `lr_anneal` (or `mom_anneal`). If an annealing function is speficied but `lrs` or `moms` is a float, it will decay to 0. If no annealing function is specified, the default is a linear annealing if `lrs` (or `moms`) is a tuple, a constant parameter if it's a float.

show_doc(GeneralScheduler)


show_doc(GeneralScheduler.on_batch_end, doc_string=False)


# Takes a step in the current phase and prepare the hyperparameters for the next batch.

show_doc(GeneralScheduler.on_train_begin, doc_string=False)


# Initiates the hyperparameters to the start values of the first phase.

# Let's make an example by using this to code [SGD with warm restarts](https://arxiv.org/abs/1608.03983).

def fit_sgd_warm(learn, n_cycles, lr, mom, cycle_len, cycle_mult):
    n = len(learn.data.train_dl)
    phases = [TrainingPhase(n * (cycle_len * cycle_mult**i), lr, mom, lr_anneal=annealing_cos) for i in range(n_cycles)]
    sched = GeneralScheduler(learn, phases)
    learn.callbacks.append(sched)
    if cycle_mult != 1:
        total_epochs = int(cycle_len * (1 - (cycle_mult)**n_cycles) / (1 - cycle_mult))
    else: total_epochs = n_cycles * cycle_len
    learn.fit(total_epochs)


path = untar_data(URLs.MNIST_SAMPLE)
data = ImageDataBunch.from_folder(path)
learn = Learner(data, simple_cnn((3, 16, 16, 2)))
fit_sgd_warm(learn, 3, 1e-3, 0.9, 1, 2)


learn.recorder.plot_lr()


# ## Undocumented Methods - Methods moved below this line will intentionally be hidden
