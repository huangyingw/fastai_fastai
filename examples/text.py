
# coding: utf-8

from fastai import *        # Quick access to most common functionality
from fastai.text import *   # Quick access to NLP functionality


# # Text example

# An example of creating a language model and then transfering to a classifier.

path = untar_data(URLs.IMDB_SAMPLE)
path


# Open and view the independent and dependent variables:

df = pd.read_csv(path / 'texts.csv', header=None)
df.head()


# Create a `DataBunch` for each of the language model and the classifier:

data_lm = TextLMDataBunch.from_csv(path, 'texts.csv')
data_clas = TextClasDataBunch.from_csv(path, 'texts.csv', vocab=data_lm.train_ds.vocab, bs=42)


# We'll fine-tune the language model. [fast.ai](http://www.fast.ai/) has a pre-trained English model available that we can download, we jsut have to specify it like this:

moms = (0.8, 0.7)


learn = language_model_learner(data_lm, pretrained_model=URLs.WT103_1)
learn.unfreeze()
learn.fit_one_cycle(4, slice(1e-2), moms=moms)


# Save our language model's encoder:

learn.save_encoder('enc')


# Fine tune it to create a classifier:

learn = text_classifier_learner(data_clas)
learn.load_encoder('enc')
learn.freeze()
learn.fit_one_cycle(4, moms=moms)


learn.unfreeze()
learn.fit_one_cycle(8, slice(1e-5, 1e-3), moms=moms)
