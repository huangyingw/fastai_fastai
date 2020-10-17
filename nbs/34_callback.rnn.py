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

# hide
# skip
from nbdev.export import notebook2script
from nbdev.showdoc import *
from fastai.basics import *
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# export

# hide


# +
# default_exp callback.rnn
# -

# # Callback for RNN training
#
# > Callback that uses the outputs of language models to add AR and TAR regularization

# export
@docs
class ModelResetter(Callback):
    "`Callback` that resets the model at each validation/training step"
    def before_train(self): self.model.reset()
    def before_validate(self): self.model.reset()
    def after_fit(self): self.model.reset()

    _docs = dict(before_train="Reset the model before training",
                 before_validate="Reset the model before validation",
                 after_fit="Reset the model after fitting")


# export
@docs
@log_args
class RNNRegularizer(Callback):
    "`Callback` that adds AR and TAR regularization in RNN training"
    def __init__(self, alpha=0., beta=0.): self.alpha, self.beta = alpha, beta

    def after_pred(self):
        self.raw_out = self.pred[1][-1] if is_listy(self.pred[1]) else self.pred[1]
        self.out = self.pred[2][-1] if is_listy(self.pred[2]) else self.pred[2]
        self.learn.pred = self.pred[0]

    def after_loss(self):
        if not self.training:
            return
        if self.alpha != 0.:
            self.learn.loss += self.alpha * self.out.float().pow(2).mean()
        if self.beta != 0.:
            h = self.raw_out
            if len(h) > 1:
                self.learn.loss += self.beta * (h[:, 1:] - h[:, :-1]).float().pow(2).mean()

    _docs = dict(after_pred="Save the raw and dropped-out outputs and only keep the true output for loss computation",
                 after_loss="Add AR and TAR regularization")


# ## Export -

# hide
notebook2script()
