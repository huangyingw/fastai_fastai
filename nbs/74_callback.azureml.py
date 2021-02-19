# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
# ---

# hide
# skip
from azureml.core.run import Run
from nbdev.showdoc import *
from fastai.learner import Callback
from fastai.basics import *
import tempfile
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# +
# all_slow
# -

# export

# hide

# default_exp callback.azureml


# # AzureML Callback
#
# Track fastai experiments with the azure machine learning plattform.
#
# ## Prerequisites
#
# Install the azureml SDK
#
#     pip install azureml-core
#
#
# ## How to use it?
#
# Add the `AzureMLCallback` to your learner.
#
#     from fastai.callback.azureml import AzureMLCallback
#
#     learn = Learner(dls, model,
#                 cbs=AzureMLCallback()
#                 )
#
# When you submit your training run with the [ScriptRunConfig](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets), the callback will automatically detect the run and log metrics.
#
# If you are running an experiment on your local machine, the callback will recognize that there is no AzureML run and print the log attempts.
#
# To save the model weights, use the usual fastai methods and save the model to the `outputs` folder, which is automatically tracked in AzureML.
#
#

# export


# export
class AzureMLCallback(Callback):
    "Log losses, metrics, model architecture summary to AzureML"
    order = Recorder.order + 1

    def before_fit(self):
        self.run = Run.get_context()

        self.run.log("n_epoch", self.learn.n_epoch)
        self.run.log("model_class", str(type(self.learn.model)))

        try:
            summary_file = Path("outputs") / 'model_summary.txt'
            with summary_file.open("w") as f:
                f.write(repr(self.learn.model))
        except:
            print('Did not log model summary. Check if your model is PyTorch model.')

    def after_batch(self):
        # log loss and opt.hypers
        if self.learn.training:
            # self.run.log('batch__smooth_loss', self.learn.smooth_loss)
            self.run.log('batch__loss', self.learn.loss)
            self.run.log('batch__train_iter', self.learn.train_iter)
            for i, h in enumerate(self.learn.opt.hypers):
                for k, v in h.items():
                    self.run.log(f'batch__opt.hypers.{k}', v)

    def after_epoch(self):
        # log metrics
        for n, v in zip(self.learn.recorder.metric_names, self.learn.recorder.log):
            if n not in ['epoch', 'time']:
                self.run.log(f'epoch__{n}', v)
            if n == 'time':
                self.run.log(f'epoch__{n}', str(v))
