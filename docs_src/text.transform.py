
# coding: utf-8

# # NLP Preprocessing

from fastai.gen_doc.nbdoc import *
from fastai.text import *
from fastai import *


# `text.tranform` contains the functions that deal behind the scenes with the two main tasks when preparing texts for modelling: *tokenization* and *numericalization*.
#
# *Tokenization* splits the raw texts into tokens (wich can be words, or punctuation signs...). The most basic way to do this would be to separate according to spaces, but it's possible to be more subtle; for instance, the contractions like "isn't" or "don't" should be split in \["is","n't"\] or \["do","n't"\]. By default fastai will use the powerful [spacy tokenizer](https://spacy.io/api/tokenizer).
#
# *Numericalization* is easier as it just consists in attributing a unique id to each token and mapping each of those tokens to their respective ids.

# ## Tokenization

# ### Introduction

# This step is actually divided in two phases: first, we apply a certain list of `rules` to the raw texts as preprocessing, then we use the tokenizer to split them in lists of tokens. Combining together those `rules`, the `tok_func`and the `lang` to process the texts is the role of the [`Tokenizer`](/text.transform.html#Tokenizer) class.

show_doc(Tokenizer, doc_string=False)


# This class will process texts by appling them the `rules` then tokenizing them with `tok_func(lang)`. `special_cases` are a list of tokens passed as special to the tokenizer and `n_cpus` is the number of cpus to use for multi-processing (by default, half the cpus available). We don't directly pass a tokenizer for multi-processing purposes: each process needs to initiate a tokenizer of its own. The rules and special_cases default to
#
# `default_rules = [fix_html, replace_rep, replace_wrep, deal_caps, spec_add_spaces, rm_useless_spaces]` <div style="text-align: right"><a href="https://github.com/fastai/fastai/blob/master/fastai/text/transform.py#L78">[source]</a></div>
#
# and
#
# `default_spec_tok = [BOS, FLD, UNK, PAD]` <div style="text-align: right"><a href="https://github.com/fastai/fastai/blob/master/fastai/text/transform.py#L79">[source]</a></div>

show_doc(Tokenizer.process_text)


show_doc(Tokenizer.process_all)


# For an example, we're going to grab some IMDB reviews.

path = untar_data(URLs.IMDB_SAMPLE)
path


df = pd.read_csv(path / 'texts.csv', header=None)
example_text = df.iloc[2][1]; example_text


tokenizer = Tokenizer()
tok = SpacyTokenizer('en')
' '.join(tokenizer.process_text(example_text, tok))


# As explained before, the tokenizer split the text according to words/punctuations signs but in a smart manner. The rules (see below) also have modified the text a little bit. We can tokenize a list of texts directly at the same time:

df = pd.read_csv(path / 'texts.csv', header=None)
texts = df[1].values
tokenizer = Tokenizer()
tokens = tokenizer.process_all(texts)
' '.join(tokens[2])


# ### Customize the tokenizer

# The `tok_func` must return an instance of [`BaseTokenizer`](/text.transform.html#BaseTokenizer):

show_doc(BaseTokenizer)


show_doc(BaseTokenizer.tokenizer)


# Take a text `t` and returns the list of its tokens.

show_doc(BaseTokenizer.add_special_cases)


# Record a list of special tokens `toks`.

# The fastai library uses [spacy](https://spacy.io/) tokenizers as its default. The following class wraps it as [`BaseTokenizer`](/text.transform.html#BaseTokenizer).

show_doc(SpacyTokenizer)


# If you want to use your custom tokenizer, just subclass the [`BaseTokenizer`](/text.transform.html#BaseTokenizer) and override its `tokenizer` and `add_spec_cases` functions.

# ### Rules

# Rules are just functions that take a string and return the modified string. This allows you to customize the list of `default_rules` as you please. Those `default_rules` are:

show_doc(deal_caps, doc_string=False)


# In `t`, every word is lower-casse. If a word begins with a capital, we put a token `TK_MAJ` in front of it.

show_doc(fix_html, doc_string=False)


# This rules replaces a bunch of HTML characters or norms in plain text ones. For instance `<br />` are replaced by `\n`, `&nbsp;` by spaces etc...

fix_html("Some HTML&nbsp;text<br />")


show_doc(replace_all_caps)


show_doc(replace_rep, doc_string=False)


# Whenever a character is repeated more than three times in `t`, we replace the whole thing by 'TK_REP n char' where n is the number of occurences and char the character.

replace_rep("I'm so excited!!!!!!!!")


show_doc(replace_wrep, doc_string=False)


# Whenever a word is repeated more than four times in `t`, we replace the whole thing by 'TK_WREP n w' where n is the number of occurences and w the word repeated.

replace_wrep("I've never ever ever ever ever ever ever ever done this.")


show_doc(rm_useless_spaces)


rm_useless_spaces("Inconsistent   use  of     spaces.")


show_doc(spec_add_spaces)


spec_add_spaces('I #like to #put #hashtags #everywhere!')


# ## Numericalization

# To convert our set of tokens to unique ids (and be able to have them go through embeddings), we use the following class:

show_doc(Vocab, doc_string=False)


# Contain the correspondance between numbers and tokens and numericalize. `itos` contains the id to token correspondance.

show_doc(Vocab.create, doc_string=False)


# Create a [`Vocab`](/text.transform.html#Vocab) dictionary from a set of `tokens`. Only keeps `max_vocab` tokens, and only if they appear at least `min_freq` times, set the rest to `UNK`.

show_doc(Vocab.numericalize)


show_doc(Vocab.textify)


vocab = Vocab.create(tokens, max_vocab=1000, min_freq=2)
vocab.numericalize(tokens[2])[:10]


# ## Undocumented Methods - Methods moved below this line will intentionally be hidden

show_doc(SpacyTokenizer.tokenizer)


show_doc(SpacyTokenizer.add_special_cases)


# ## New Methods - Please document or move to the undocumented section
