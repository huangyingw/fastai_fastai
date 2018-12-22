# coding: utf-8
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from fastai.gen_doc.gen_notebooks import *
from pathlib import Path
# ### To update this notebook
# Run `tools/sgen_notebooks.py
# Or run below:
# You need to make sure to refresh right after
import glob
for f in Path().glob('*.ipynb'):
    generate_missing_metadata(f)
# # Metadata generated below
update_nb_metadata('tutorial.itemlist.ipynb',
    summary='Advanced tutorial, explains how to create your custom `ItemBase` or `ItemList`',
    title='Custom ItemList')
update_nb_metadata('tutorial.inference.ipynb',
    summary='Intermediate tutorial, explains how to create a Learner for inference',
    title='Inference Learner')
update_nb_metadata('tutorial.data.ipynb',
    summary="Beginner's tutorial, explains how to quickly look at your data or model predictions",
    title='Look at data')
update_nb_metadata('vision.gan.ipynb',
    summary='All the modules and callbacks necessary to train a GAN',
    title='vision.gan')
update_nb_metadata('callbacks.csv_logger.ipynb',
    summary='Callbacks that saves the tracked metrics during training',
    title='callbacks.csv_logger')
update_nb_metadata('callbacks.tracker.ipynb',
    summary='Callbacks that take decisions depending on the evolution of metrics during training',
    title='callbacks.tracker')
update_nb_metadata('torch_core.ipynb',
    summary='Basic functions using pytorch',
    title='torch_core')
update_nb_metadata('gen_doc.convert2html.ipynb',
    summary='Converting the documentation notebooks to HTML pages',
    title='gen_doc.convert2html')
update_nb_metadata('metrics.ipynb',
    summary='Useful metrics for training',
    title='metrics')
update_nb_metadata('callbacks.fp16.ipynb',
    summary='Training in mixed precision implementation',
    title='callbacks.fp16')
update_nb_metadata('callbacks.general_sched.ipynb',
    summary='Implementation of a flexible training API',
    title='callbacks.general_sched')
update_nb_metadata('text.ipynb',
    keywords='fastai',
    summary='Application to NLP, including ULMFiT fine-tuning',
    title='text')
update_nb_metadata('callback.ipynb',
    summary='Implementation of the callback system',
    title='callback')
update_nb_metadata('tabular.models.ipynb',
    keywords='fastai',
    summary='Model for training tabular/structured data',
    title='tabular.models')
update_nb_metadata('callbacks.mixup.ipynb',
    summary='Implementation of mixup',
    title='callbacks.mixup')
update_nb_metadata('applications.ipynb',
    summary='Types of problems you can apply the fastai library to',
    title='applications')
update_nb_metadata('vision.data.ipynb',
    summary='Basic dataset for computer vision and helper function to get a DataBunch',
    title='vision.data')
update_nb_metadata('overview.ipynb',
    summary='Overview of the core modules',
    title='overview')
update_nb_metadata('training.ipynb',
    keywords='fastai',
    summary='Overview of fastai training modules, including Learner, metrics, and callbacks',
    title='training')
update_nb_metadata('text.transform.ipynb',
    summary='NLP data processing; tokenizes text and creates vocab indexes',
    title='text.transform')
# do not overwrite this notebook, or changes may get lost!
# update_nb_metadata('jekyll_metadata.ipynb')
update_nb_metadata('collab.ipynb',
    summary='Application to collaborative filtering',
    title='collab')
update_nb_metadata('text.learner.ipynb',
    summary='Easy access of language models and ULMFiT',
    title='text.learner')
update_nb_metadata('gen_doc.nbdoc.ipynb',
    summary='Helper function to build the documentation',
    title='gen_doc.nbdoc')
update_nb_metadata('vision.learner.ipynb',
    summary='`Learner` support for computer vision',
    title='vision.learner')
update_nb_metadata('core.ipynb',
    summary='Basic helper functions for the fastai library',
    title='core')
update_nb_metadata('fastai_typing.ipynb',
    keywords='fastai',
    summary='Type annotations names',
    title='fastai_typing')
update_nb_metadata('gen_doc.gen_notebooks.ipynb',
    summary='Generation of documentation notebook skeletons from python module',
    title='gen_doc.gen_notebooks')
update_nb_metadata('basic_train.ipynb',
    summary='Learner class and training loop',
    title='basic_train')
update_nb_metadata('gen_doc.ipynb',
    keywords='fastai',
    summary='Documentation modules overview',
    title='gen_doc')
update_nb_metadata('callbacks.rnn.ipynb',
    summary='Implementation of a callback for RNN training',
    title='callbacks.rnn')
update_nb_metadata('callbacks.one_cycle.ipynb',
    summary='Implementation of the 1cycle policy',
    title='callbacks.one_cycle')
update_nb_metadata('vision.ipynb',
    summary='Application to Computer Vision',
    title='vision')
update_nb_metadata('vision.transform.ipynb',
    summary='List of transforms for data augmentation in CV',
    title='vision.transform')
update_nb_metadata('callbacks.lr_finder.ipynb',
    summary='Implementation of the LR Range test from Leslie Smith',
    title='callbacks.lr_finder')
update_nb_metadata('text.data.ipynb',
    summary='Basic dataset for NLP tasks and helper functions to create a DataBunch',
    title='text.data')
update_nb_metadata('text.models.ipynb',
    summary='Implementation of the AWD-LSTM and the RNN models',
    title='text.models')
update_nb_metadata('tabular.data.ipynb',
    summary='Base class to deal with tabular data and get a DataBunch',
    title='tabular.data')
update_nb_metadata('callbacks.ipynb',
    keywords='fastai',
    summary='Callbacks implemented in the fastai library',
    title='callbacks')
update_nb_metadata('train.ipynb',
    summary='Extensions to Learner that easily implement Callback',
    title='train')
update_nb_metadata('callbacks.hooks.ipynb',
    summary='Implement callbacks using hooks',
    title='callbacks.hooks')
update_nb_metadata('vision.image.ipynb',
    summary='Image class, variants and internal data augmentation pipeline',
    title='vision.image')
update_nb_metadata('vision.models.unet.ipynb',
    summary='Dynamic Unet that can use any pretrained model as a backbone.',
    title='vision.models.unet')
update_nb_metadata('vision.models.ipynb',
    keywords='fastai',
    summary='Overview of the models used for CV in fastai',
    title='vision.models')
update_nb_metadata('tabular.transform.ipynb',
    summary='Transforms to clean and preprocess tabular data',
    title='tabular.transform')
update_nb_metadata('index.ipynb',
    keywords='fastai',
    toc='false',
    title='Welcome to fastai')
update_nb_metadata('layers.ipynb',
    summary='Provides essential functions to building and modifying `Model` architectures.',
    title='layers')
update_nb_metadata('tabular.ipynb',
    keywords='fastai',
    summary='Application to tabular/structured data',
    title='tabular')
update_nb_metadata('basic_data.ipynb',
    summary='Basic classes to contain the data for model training.',
    title='basic_data')
update_nb_metadata('datasets.ipynb')
update_nb_metadata('tmp.ipynb',
    keywords='fastai')
update_nb_metadata('callbacks.tracking.ipynb')
update_nb_metadata('data_block.ipynb',
    keywords='fastai',
    summary='The data block API',
    title='data_block')
update_nb_metadata('callbacks.tracker.ipynb',
    keywords='fastai',
    summary='Callbacks that take decisions depending on the evolution of metrics during training',
    title='callbacks.tracking')
update_nb_metadata('widgets.ipynb')
update_nb_metadata('text_tmp.ipynb')
update_nb_metadata('tabular_tmp.ipynb')
update_nb_metadata('tutorial.data.ipynb')
update_nb_metadata('tutorial.itemlist.ipynb')
update_nb_metadata('tutorial.inference.ipynb')
update_nb_metadata('vision.gan.ipynb')
update_nb_metadata('utils.collect_env.ipynb')
