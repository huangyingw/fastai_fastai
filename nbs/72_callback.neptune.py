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

# hide
# skip
import neptune
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

# +
# default_exp callback.neptune
# -

# # Neptune.ai
#
# > Integration with [neptune.ai](https://www.neptune.ai).
#
# > [Track fastai experiments](https://ui.neptune.ai/o/neptune-ai/org/fastai-integration) like in this example project.

# ## Registration

# 1. Create account: [neptune.ai/register](https://neptune.ai/register).
# 2. Export API token to the environment variable (more help [here](https://docs.neptune.ai/python-api/tutorials/get-started.html#copy-api-token)). In your terminal run:
#
# ```
# export NEPTUNE_API_TOKEN='YOUR_LONG_API_TOKEN'
# ```
#
# or append the command above to your `~/.bashrc` or `~/.bash_profile` files (**recommended**). More help is [here](https://docs.neptune.ai/python-api/tutorials/get-started.html#copy-api-token).

# ## Installation

# 1. You need to install neptune-client. In your terminal run:
#
# ```
# pip install neptune-client
# ```
#
# or (alternative installation using conda). In your terminal run:
#
# ```
# conda install neptune-client -c conda-forge
# ```
# 2. Install [psutil](https://psutil.readthedocs.io/en/latest/) to see hardware monitoring charts:
#
# ```
# pip install psutil
# ```

# ## How to use?

# Key is to call `neptune.init()` before you create `Learner()` and call `neptune_create_experiment()`, before you fit the model.
#
# Use `NeptuneCallback` in your `Learner`, like this:
#
# ```
# from fastai.callback.neptune import NeptuneCallback
#
# neptune.init('USERNAME/PROJECT_NAME')  # specify project
#
# learn = Learner(dls, model,
#                 cbs=NeptuneCallback()
#                 )
#
# neptune.create_experiment()  # start experiment
# learn.fit_one_cycle(1)
# ```

# export


# export
class NeptuneCallback(Callback):
    "Log losses, metrics, model weights, model architecture summary to neptune"
    order = Recorder.order + 1

    def __init__(self, log_model_weights=True, keep_experiment_running=False):
        self.log_model_weights = log_model_weights
        self.keep_experiment_running = keep_experiment_running
        self.experiment = None

        if neptune.project is None:
            raise ValueError('You did not initialize project in neptune.\n',
                             'Please invoke `neptune.init("USERNAME/PROJECT_NAME")` before this callback.')

    def before_fit(self):
        try:
            self.experiment = neptune.get_experiment()
        except ValueError:
            print('No active experiment. Please invoke `neptune.create_experiment()` before this callback.')

        try:
            self.experiment.set_property('n_epoch', str(self.learn.n_epoch))
            self.experiment.set_property('model_class', str(type(self.learn.model)))
        except:
            print(f'Did not log all properties. Check properties in the {neptune.get_experiment()}.')

        try:
            with tempfile.NamedTemporaryFile(mode='w') as f:
                with open(f.name, 'w') as g:
                    g.write(repr(self.learn.model))
                self.experiment.log_artifact(f.name, 'model_summary.txt')
        except:
            print('Did not log model summary. Check if your model is PyTorch model.')

        if self.log_model_weights and not hasattr(self.learn, 'save_model'):
            print('Unable to log model to Neptune.\n',
                  'Use "SaveModelCallback" to save model checkpoints that will be logged to Neptune.')

    def after_batch(self):
        # log loss and opt.hypers
        if self.learn.training:
            self.experiment.log_metric('batch__smooth_loss', self.learn.smooth_loss)
            self.experiment.log_metric('batch__loss', self.learn.loss)
            self.experiment.log_metric('batch__train_iter', self.learn.train_iter)
            for i, h in enumerate(self.learn.opt.hypers):
                for k, v in h.items():
                    self.experiment.log_metric(f'batch__opt.hypers.{k}', v)

    def after_epoch(self):
        # log metrics
        for n, v in zip(self.learn.recorder.metric_names, self.learn.recorder.log):
            if n not in ['epoch', 'time']:
                self.experiment.log_metric(f'epoch__{n}', v)
            if n == 'time':
                self.experiment.log_text(f'epoch__{n}', str(v))

        # log model weights
        if self.log_model_weights and hasattr(self.learn, 'save_model'):
            if self.learn.save_model.every_epoch:
                _file = join_path_file(f'{self.learn.save_model.fname}_{self.learn.save_model.epoch}',
                                       self.learn.path / self.learn.model_dir, ext='.pth')
            else:
                _file = join_path_file(self.learn.save_model.fname,
                                       self.learn.path / self.learn.model_dir, ext='.pth')
            self.experiment.log_artifact(_file)

    def after_fit(self):
        if not self.keep_experiment_running:
            try:
                self.experiment.stop()
            except:
                print('No neptune experiment to stop.')
        else:
            print(f'Your experiment (id: {self.experiment.id}, name: {self.experiment.name}) is left in the running state.\n',
                  'You can log more data to it, like this: `neptune.log_metric()`')
