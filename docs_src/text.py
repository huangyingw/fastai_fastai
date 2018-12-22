# coding: utf-8
# # Text models, data, and training
from fastai.gen_doc.nbdoc import *
# The [`text`](/text.html#text) module of the fastai library contains all the necessary functions to define a Dataset suitable for the various NLP (Natural Language Processing) tasks and quickly generate models you can use for them. Specifically:
# - [`text.transform`](/text.transform.html#text.transform) contains all the scripts to preprocess your data, from raw text to token ids,
# - [`text.data`](/text.data.html#text.data) contains the definition of [`TextDataset`](/text.data.html#TextDataset), which the main class you'll need in NLP,
# - [`text.learner`](/text.learner.html#text.learner) contains helper functions to quickly create a language model or an RNN classifier.
#
# Have a look at the links above for full details of the API of each module, of read on for a quick overview.
# ## Quick Start: Training an IMDb sentiment model with *ULMFiT*
# Let's start with a quick end-to-end example of training a model. We'll train a sentiment classifier on a sample of the popular IMDb data, showing 4 steps:
#
# 1. Reading and viewing the IMDb data
# 1. Getting your data ready for modeling
# 1. Fine-tuning a language model
# 1. Building a classifier
# ### Reading and viewing the IMDb data
# First let's import everything we need for text.
from fastai.text import *
# Contrary to images in Computer Vision, text can't directly be transformed into numbers to be fed into a model. The first thing we need to do is to preprocess our data so that we change the raw texts to lists of words, or tokens (a step that is called tokenization) then transform these tokens into numbers (a step that is called numericalization). These numbers are then passed to embedding layers that wil convert them in arrays of floats before passing them through a model.
#
# You can find on the web plenty of [Word Embeddings](https://en.wikipedia.org/wiki/Word_embedding) to directly convert your tokens into floats. Those word embeddings have generally be trained on a large corpus such as wikipedia. Following the work of [ULMFiT](https://arxiv.org/abs/1801.06146), the fastai library is more focused on using pre-trained Language Models and fine-tuning them. Word embeddings are just vectors of 300 or 400 floats that represent different words, but a pretrained language model not only has those, but has also been trained to get a representation of full sentences and documents.
#
# That's why the library is structured around three steps:
#
# 1. Get your data preprocessed and ready to use in a minimum amount of code,
# 1. Create a language model with pretrained weights that you can fine-tune to your dataset,
# 1. Create other models such as classifiers on top of the encoder of the language model.
#
# To show examples, we have provided a small sample of the [IMDB dataset](https://www.imdb.com/interfaces/) which contains 1,000 reviews of movies with labels (positive or negative).
path = untar_data(URLs.IMDB_SAMPLE)
path
# Creating a dataset from your raw texts is very simple if you have it in one of those ways
# - organized it in folders in an ImageNet style
# - organized in a csv file with labels columns and a text columns
#
# Here, the sample from imdb is in a texts csv files that looks like this:
df = pd.read_csv(path / 'texts.csv')
df.head()
# ### Getting your data ready for modeling
for file in ['train_tok.npy', 'valid_tok.npy']:
    if os.path.exists(path / 'tmp' / file): os.remove(path / 'tmp' / file)
# To get a [`DataBunch`](/basic_data.html#DataBunch) quickly, there are also several factory methods depending on how our data is structured. They are all detailed in [`text.data`](/text.data.html#text.data), here we'll use the method <code>from_csv</code> of the [`TextLMDataBunch`](/text.data.html#TextLMDataBunch) (to get the data ready for a language model) and [`TextClasDataBunch`](/text.data.html#TextClasDataBunch) (to get the data ready for a text classifier) classes.
# Language model data
data_lm = TextLMDataBunch.from_csv(path, 'texts.csv')
# Classifier model data
data_clas = TextClasDataBunch.from_csv(path, 'texts.csv', vocab=data_lm.train_ds.vocab, bs=32)
# This does all the necessary preprocessing behing the scene. For the classifier, we also pass the vocabulary (mapping from ids to words) that we want to use: this is to ensure that `data_clas` will use the same dictionary as `data_lm`.
#
# Since this step can be a bit time-consuming, it's best to save the result with:
data_lm.save()
data_clas.save()
# This will create a 'tmp' directory where all the computed stuff will be stored. You can then reload those results with:
data_lm = TextLMDataBunch.load(path)
data_clas = TextClasDataBunch.load(path, bs=32)
# Note that you can load the data with different [`DataBunch`](/basic_data.html#DataBunch) parameters (batch size, `bptt`,...)
# ### Fine-tuning a language model
# We can use the `data_lm` object we created earlier to fine-tune a pretrained language model. [fast.ai](http://www.fast.ai/) has an English model available that we can download. We can create a learner object that will directly create a model, download the pretrained weights and be ready for fine-tuning.
learn = language_model_learner(data_lm, pretrained_model=URLs.WT103_1, drop_mult=0.5)
learn.fit_one_cycle(1, 1e-2)
# Like a computer vision model, we can then unfreeze the model and fine-tune it.
learn.unfreeze()
learn.fit_one_cycle(1, 1e-3)
# To evaluate your language model, you can run the [`Learner.predict`](/basic_train.html#Learner.predict) method and specify the number of words you want it to guess.
learn.predict("This is a review about", n_words=10)
# It doesn't make much sense (we have a tiny vocabulary here and didn't train much on it) but note that it respects basic grammar (which comes from the pretrained model).
#
# Finally we save the encoder to be able to use it for classification in the next section.
learn.save_encoder('ft_enc')
# ### Building a classifier
# We now use the `data_clas` object we created earlier to build a classifier with our fine-tuned encoder. The learner object can be done in a single line.
learn = text_classifier_learner(data_clas, drop_mult=0.5)
learn.load_encoder('ft_enc')
data_clas.show_batch()
learn.fit_one_cycle(1, 1e-2)
# Again, we can unfreeze the model and fine-tune it.
learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(5e-3 / 2., 5e-3))
learn.unfreeze()
learn.fit_one_cycle(1, slice(2e-3 / 100, 2e-3))
# Again, we can predict on a raw text by using the [`Learner.predict`](/basic_train.html#Learner.predict) method.
learn.predict("This was a great movie!")
