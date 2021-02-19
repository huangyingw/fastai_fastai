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
class RNNCallback(Callback):
    "Save the raw and dropped-out outputs and only keep the true output for loss computation"

    def after_pred(self): self.learn.pred, self.raw_out, self.out = [o[-1] if is_listy(o) else o for o in self.pred]


# export
class RNNRegularizer(Callback):
    "Add AR and TAR regularization"
    order, run_valid = RNNCallback.order + 1, False
    def __init__(self, alpha=0., beta=0.): store_attr()

    def after_loss(self):
        if not self.training:
            return
        if self.alpha:
            self.learn.loss_grad += self.alpha * self.rnn.out.float().pow(2).mean()
        if self.beta:
            h = self.rnn.raw_out
            if len(h) > 1:
                self.learn.loss_grad += self.beta * (h[:, 1:] - h[:, :-1]).float().pow(2).mean()


# export
def rnn_cbs(alpha=0., beta=0.):
    "All callbacks needed for (optionally regularized) RNN training"
    reg = [RNNRegularizer(alpha=alpha, beta=beta)] if alpha or beta else []
    return [ModelResetter(), RNNCallback()] + reg


# ## Export -

# hide
notebook2script()
