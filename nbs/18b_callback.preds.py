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
from fastai.test_utils import *
from nbdev.showdoc import *
from fastai.basics import *
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# +
# default_exp callback.preds
# -

# export

# hide


# # Predictions callbacks
#
# > Various callbacks to customize get_preds behaviors

# ## MCDropoutCallback
#
# > Turns on dropout during inference, allowing you to call Learner.get_preds multiple times to approximate your model uncertainty using [Monte Carlo Dropout](https://arxiv.org/pdf/1506.02142.pdf).

# export
class MCDropoutCallback(Callback):
    def before_validate(self):
        for m in [m for m in flatten_model(self.model) if 'dropout' in m.__class__.__name__.lower()]:
            m.train()

    def after_validate(self):
        for m in [m for m in flatten_model(self.model) if 'dropout' in m.__class__.__name__.lower()]:
            m.eval()


# +
learn = synth_learner()

# Call get_preds 10 times, then stack the predictions, yielding a tensor with shape [# of samples, batch_size, ...]
dist_preds = []
for i in range(10):
    preds, targs = learn.get_preds(cbs=[MCDropoutCallback()])
    dist_preds += [preds]

torch.stack(dist_preds).shape
# -

# ## Export -

# hide
notebook2script()
