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

# # IMDB

# %matplotlib inline

from fastai.text.all import *
from nbdev.showdoc import *

# ## Preparing the data

# First let's download the dataset we are going to study. The [dataset](http://ai.stanford.edu/~amaas/data/sentiment/) has been curated by Andrew Maas et al. and contains a total of 100,000 reviews on IMDB. 25,000 of them are labelled as positive and negative for training, another 25,000 are labelled for testing (in both cases they are highly polarized). The remaning 50,000 is an additional unlabelled data (but we will find a use for it nonetheless).
#
# We'll begin with a sample we've prepared for you, so that things run quickly before going over the full dataset.

path = untar_data(URLs.IMDB_SAMPLE)
path.ls()

# It only contains one csv file, let's have a look at it.

df = pd.read_csv(path / 'texts.csv')
df.head()

df['text'][1]

# It contains one line per review, with the label ('negative' or 'positive'), the text and a flag to determine if it should be part of the validation set or the training set.
#
# First, we need to tokenize the texts in our dataframe, which means separate the sentences in individual tokens (often words).

# ### Tokenization

# The first step of processing we make the texts go through is to split the raw sentences into words, or more exactly tokens. The easiest way to do this would be to split the string on spaces, but we can be smarter:
#
# - we need to take care of punctuation
# - some words are contractions of two different words, like isn't or don't
# - we may need to clean some parts of our texts, if there's HTML code for instance

# The texts are truncated at 100 tokens for more readability. We can see that it did more than just split on space and punctuation symbols:
# - the "'s" are grouped together in one token
# - the contractions are separated like this: "did", "n't"
# - content has been cleaned for any HTML symbol and lower cased
# - there are several special tokens (all those that begin by xx), to replace unknown tokens (see below) or to introduce different text fields (here we only have one).

# ### Numericalization

# Once we have extracted tokens from our texts, we convert to integers by creating a list of all the words used. We only keep the ones that appear at least twice with a maximum vocabulary size of 60,000 (by default) and replace the ones that don't make the cut by the unknown token `UNK`.
#
# This is done automatically behind the scenes if we use a facotry method of `TextDataLoaders`.

dbunch_lm = TextDataLoaders.from_df(df, text_col='text', label_col='label', path=path, is_lm=True, valid_col='is_valid')

# And if we look at what a what's in our datasets, we'll see the numericalized text as a representation:

dbunch_lm.train_ds[0]

# The correspondence is stored in the vocab attribute of our `DataLoaders`

dbunch_lm.vocab[:20]

dbunch_lm.show_batch()

# ### With the data block API

# We can use the data block API with NLP and have a lot more flexibility than what the default factory methods offer. In the previous example for instance, the data was randomly split between train and validation instead of reading the third column of the csv.
#
# With the data block API though, we have to manually call the tokenize and numericalize steps. This allows more flexibility, and if you're not using the defaults from fastai, the various arguments to pass will appear in the step they're revelant, so it'll be more readable.

# +
imdb_lm = DataBlock(blocks=(TextBlock.from_df('text', is_lm=True),),
                    get_x=ColReader('text'),
                    splitter=RandomSplitter())

dbunch_lm = imdb_lm.dataloaders(df)
# -

# ## Language model

# Note that language models can use a lot of GPU, so you may need to decrease batchsize here.

bs = 128

# Now let's grab the full dataset for what follows.

path = untar_data(URLs.IMDB)
path.ls()

(path / 'train').ls()

# The reviews are in a training and test set following an imagenet structure. The only difference is that there is an `unsup` folder on top of `train` and `test` that contains the unlabelled data.
#
# We're not going to train a model that classifies the reviews from scratch. Like in computer vision, we'll use a model pretrained on a bigger dataset (a cleaned subset of wikipedia called [wikitext-103](https://einstein.ai/research/blog/the-wikitext-long-term-dependency-language-modeling-dataset)). That model has been trained to guess what the next word is, its input being all the previous words. It has a recurrent structure and a hidden state that is updated each time it sees a new word. This hidden state thus contains information about the sentence up to that point.
#
# We are going to use that 'knowledge' of the English language to build our classifier, but first, like for computer vision, we need to fine-tune the pretrained model to our particular dataset. Because the English of the reviews left by people on IMDB isn't the same as the English of wikipedia, we'll need to adjust the parameters of our model by a little bit. Plus there might be some words that would be extremely common in the reviews dataset but would be barely present in wikipedia, and therefore might not be part of the vocabulary the model was trained on.

# This is where the unlabelled data is going to be useful to us, as we can use it to fine-tune our model. Let's create our data object with the data block API (next line takes a few minutes the first minute you run it).

# +
imdb_lm = DataBlock(blocks=(TextBlock.from_folder(path, is_lm=True),),
                    get_items=partial(get_text_files, folders=['train', 'test', 'unsup']),
                    splitter=RandomSplitter(0.1))

dbunch_lm = imdb_lm.dataloaders(path, path=path, bs=bs, seq_len=80)
# -

# We have to use a special kind of `TextDataLoaders` for the language model, that ignores the labels (that's why we put 0 everywhere), will shuffle the texts at each epoch before concatenating them all together (only for training, we don't shuffle for the validation set) and will send batches that read that text in order with targets that are the next word in the sentence.
#
# The line before being a bit long, we want to load quickly the final ids by using the following cell.

dbunch_lm.show_batch()

# We can then put this in a learner object very easily with a model loaded with the pretrained weights. They'll be downloaded the first time you'll execute the following line and stored in `~/.fastai/models/` (or elsewhere if you specified different paths in your config file).

len(dbunch_lm.vocab)

learn = language_model_learner(dbunch_lm, AWD_LSTM, drop_mult=0.3, metrics=[accuracy, Perplexity()]).to_fp16()

learn.lr_find()

learn.recorder.plot_lr_find(skip_end=15)

learn.fit_one_cycle(1, 2e-2, moms=(0.8, 0.7, 0.8))

learn.save('fit_head')

learn.load('fit_head')

# To complete the fine-tuning, we can then unfeeze and launch a new training.

learn.unfreeze()

learn.fit_one_cycle(10, 2e-3, moms=(0.8, 0.7, 0.8))

learn.save('fine_tuned')

# How good is our model? Well let's try to see what it predicts after a few given words.

learn.load('fine_tuned')

TEXT = "I liked this movie because"
N_WORDS = 40
N_SENTENCES = 2

print("\n".join(learn.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))

# We have to save not only the model, but also its encoder, the part that's responsible for creating and updating the hidden state. For the next part, we don't care about the part that tries to guess the next word.

learn.save_encoder('fine_tuned_enc')


# ## Classifier

# Now, we'll create a new data object that only grabs the labelled data and keeps those labels. Again, this line takes a bit of time.

def read_tokenized_file(f): return L(f.read_text().split(' '))


# +
imdb_clas = DataBlock(blocks=(TextBlock.from_folder(path, vocab=dbunch_lm.vocab), CategoryBlock),
                      get_x=read_tokenized_file,
                      get_y=parent_label,
                      get_items=partial(get_text_files, folders=['train', 'test']),
                      splitter=GrandparentSplitter(valid_name='test'))

dbunch_clas = imdb_clas.dataloaders(path, path=path, bs=bs, seq_len=80)
# -

dbunch_clas.show_batch()

# We can then create a model to classify those reviews and load the encoder we saved before.

learn = text_classifier_learner(dbunch_clas, AWD_LSTM, drop_mult=0.5, metrics=accuracy).to_fp16()
learn.load_encoder('fine_tuned_enc')

learn.lr_find()

learn.fit_one_cycle(1, 2e-2, moms=(0.8, 0.7, 0.8))

learn.save('first')

learn.load('first')

learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(1e-2 / (2.6**4), 1e-2), moms=(0.8, 0.7, 0.8))

learn.save('second')

learn.load('second')

learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(5e-3 / (2.6**4), 5e-3), moms=(0.8, 0.7, 0.8))

learn.save('third')

learn.load('third')

learn.unfreeze()
learn.fit_one_cycle(2, slice(1e-3 / (2.6**4), 1e-3), moms=(0.8, 0.7, 0.8))

learn.predict("I really loved that movie , it was awesome !")
