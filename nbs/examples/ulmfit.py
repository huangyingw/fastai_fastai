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

from fastai.text.all import *
from nbdev.showdoc import show_doc

# # ULMFiT

# ## Finetune a pretrained Language Model

# First we get our data and tokenize it.

path = untar_data(URLs.IMDB)

texts = get_files(path, extensions=['.txt'], folders=['unsup', 'train', 'test'])
len(texts)


# Then we put it in a `Datasets`. For a language model, we don't have targets, so there is only one transform to numericalize the texts. Note that `tokenize_df` returns the count of the words in the corpus to make it easy to create a vocabulary.

def read_file(f): return L(f.read_text().split(' '))


splits = RandomSplitter(valid_pct=0.1)(texts)
tfms = [Tokenizer.from_folder(path), Numericalize()]
dsets = Datasets(texts, [tfms], splits=splits, dl_type=LMDataLoader)

# Then we use that `Datasets` to create a `DataLoaders`. Here the class of `TfmdDL` we need to use is `LMDataLoader` which will concatenate all the texts in a source (with a shuffle at each epoch for the training set), split it in `bs` chunks then read continuously through it.

bs, sl = 256, 80
dbunch_lm = dsets.dataloaders(bs=bs, seq_len=sl, val_bs=bs)

dbunch_lm.show_batch()

# Then we have a convenience method to directly grab a `Learner` from it, using the `AWD_LSTM` architecture.

opt_func = partial(Adam, wd=0.1)
learn = language_model_learner(dbunch_lm, AWD_LSTM, opt_func=opt_func, metrics=[accuracy, Perplexity()], path=path)
learn = learn.to_fp16(clip=0.1)

learn.fit_one_cycle(1, 2e-2, moms=(0.8, 0.7, 0.8))

learn.save('stage1')

learn.load('stage1')

learn.unfreeze()
learn.fit_one_cycle(10, 2e-3, moms=(0.8, 0.7, 0.8))

# Once we have fine-tuned the pretrained language model to this corpus, we save the encoder since we will use it for the classifier.

learn.save_encoder('finetuned1')

# ## Use it to train a classifier

texts = get_files(path, extensions=['.txt'], folders=['train', 'test'])

splits = GrandparentSplitter(valid_name='test')(texts)

# For classification, we need to use two set of transforms: one to numericalize the texts and the other to encode the labels as categories.

x_tfms = [Tokenizer.from_folder(path), Numericalize(vocab=dbunch_lm.vocab)]
dsets = Datasets(texts, [x_tfms, [parent_label, Categorize()]], splits=splits, dl_type=SortedDL)

bs = 64

dls = dsets.dataloaders(before_batch=pad_input_chunk, bs=bs)

dls.show_batch(max_n=2)

# Then we once again have a convenience function to create a classifier from this `DataLoaders` with the `AWD_LSTM` architecture.

opt_func = partial(Adam, wd=0.1)
learn = text_classifier_learner(dls, AWD_LSTM, metrics=[accuracy], path=path, drop_mult=0.5, opt_func=opt_func)

# We load our pretrained encoder.

learn = learn.load_encoder('finetuned1')
learn = learn.to_fp16(clip=0.1)

# Then we can train with gradual unfreezing and differential learning rates.

lr = 1e-1 * bs / 128

learn.fit_one_cycle(1, lr, moms=(0.8, 0.7, 0.8), wd=0.1)

learn.freeze_to(-2)
lr /= 2
learn.fit_one_cycle(1, slice(lr / (2.6**4), lr), moms=(0.8, 0.7, 0.8), wd=0.1)

learn.freeze_to(-3)
lr /= 2
learn.fit_one_cycle(1, slice(lr / (2.6**4), lr), moms=(0.8, 0.7, 0.8), wd=0.1)

learn.unfreeze()
lr /= 5
learn.fit_one_cycle(2, slice(lr / (2.6**4), lr), moms=(0.8, 0.7, 0.8), wd=0.1)
