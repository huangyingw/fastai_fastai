
# coding: utf-8

# # NLP datasets

from fastai.gen_doc.nbdoc import *
from fastai.text import *
from fastai.gen_doc.nbdoc import *
from fastai import *
os.chdir(os.path.dirname(os.path.realpath(__file__)))


# This module contains the [`TextDataset`](/text.data.html#TextDataset) class, which is the main dataset you should use for your NLP tasks. It automatically does the preprocessing steps described in [`text.transform`](/text.transform.html#text.transform). It also contains all the functions to quickly get a [`TextDataBunch`](/text.data.html#TextDataBunch) ready.

# ## Quickly assemble your data

# You should get your data in one of the following formats to make the most of the fastai library and use one of the factory methods of one of the [`TextDataBunch`](/text.data.html#TextDataBunch) classes:
# - raw text files in folders train, valid, test in an ImageNet style,
# - a csv where some column(s) gives the label(s) and the folowwing one the associated text,
# - a dataframe structured the same way,
# - tokens and labels arrays,
# - ids, vocabulary (correspondance id to word) and labels.
#
# If you are assembling the data for a language model, you should define your labels as always 0 to respect those formats. The first time you create a [`DataBunch`](/basic_data.html#DataBunch) with one of those functions, your data will be preprocessed automatically. You can save it, so that the next time you call it is almost instantaneous.
#
# Below are the classes that help assembling the raw data in a [`DataBunch`](/basic_data.html#DataBunch) suitable for NLP.

show_doc(TextLMDataBunch, title_level=3, doc_string=False)


# Create a [`DataBunch`](/basic_data.html#DataBunch) suitable for language modeling: all the texts in the [`datasets`](/datasets.html#datasets) are concatenated and the labels are ignored. Instead, the target is the next word in the sentence.

show_doc(TextLMDataBunch.show_batch)


show_doc(TextClasDataBunch, title_level=3, doc_string=False)


# Create a [`DataBunch`](/basic_data.html#DataBunch) suitable for a text classifier: all the texts are grouped by length (with a bit of randomness for the training set) then padded.

show_doc(TextClasDataBunch.show_batch)


show_doc(TextDataBunch, title_level=3, doc_string=False)


# Create a [`DataBunch`](/basic_data.html#DataBunch) with the raw texts. This is only going to work if they all have the same lengths.

# ### Factory methods (TextDataBunch)

# All those classes have the following factory methods.

show_doc(TextDataBunch.from_folder, doc_string=False)


# This function will create a [`DataBunch`](/basic_data.html#DataBunch) from texts placed in `path` in a [`train`](/train.html#train), `valid` and maybe `test` folders. Text files in the [`train`](/train.html#train) and `valid` folders should be places in subdirectories according to their classes (always the same for a language model) and the ones for the `test` folder should all be placed there directly. `tokenizer` will be used to parse those texts into tokens. The `shuffle` flag will optionally shuffle the texts found.
#
# You can pass a specific `vocab` for the numericalization step (if you are building a classifier from a language model you fine-tuned for instance). kwargs will be split between the [`TextDataset`](/text.data.html#TextDataset) function and to the class initialization, you can precise there parameters such as `max_vocab`, `chunksize`, `min_freq`, `n_labels` (see the [`TextDataset`](/text.data.html#TextDataset) documentation) or `bs`, `bptt` and `pad_idx` (see the sections LM data and classifier data).

show_doc(TextDataBunch.from_csv, doc_string=False)


# This function will create a [`DataBunch`](/basic_data.html#DataBunch) from texts placed in `path` in a csv file and maybe `test` csv file opened with `header`. You can specify `txt_cols` and `lbl_cols` or just an integer `n_labels` in which case the label(s) should be the first column(s). `tokenizer` will be used to parse those texts into tokens.
#
# You can pass a specific `vocab` for the numericalization step (if you are building a classifier from a language model you fine-tuned for instance). kwargs will be split between the [`TextDataset`](/text.data.html#TextDataset) function and to the class initialization, you can precise there parameters such as `max_vocab`, `chunksize`, `min_freq`, `n_labels` (see the [`TextDataset`](/text.data.html#TextDataset) documentation) or `bs`, `bptt` and `pad_idx` (see the sections LM data and classifier data).

show_doc(TextDataBunch.from_df, doc_string=False)


# This function will create a [`DataBunch`](/basic_data.html#DataBunch) in `path` from texts in `train_df`, `valid_df` and maybe `test_df`. By default, those are opened with `header=infer` but you can specify another value in the kwargs. You can specify `txt_cols` and `lbl_cols` or just an integer `n_labels` in which case the label(s) should be the first column(s). `tokenizer` will be used to parse those texts into tokens.
#
# You can pass a specific `vocab` for the numericalization step (if you are building a classifier from a language model you fine-tuned for instance). kwargs will be split between the [`TextDataset`](/text.data.html#TextDataset) function and to the class initialization, you can precise there parameters such as `max_vocab`, `chunksize`, `min_freq`, `n_labels` (see the [`TextDataset`](/text.data.html#TextDataset) documentation) or `bs`, `bptt` and `pad_idx` (see the sections LM data and classifier data).

show_doc(TextDataBunch.from_tokens, doc_string=False)


# This function will create a [`DataBunch`](/basic_data.html#DataBunch) from `trn_tok`, `trn_lbls`, `val_tok`, `val_lbls` and maybe `tst_tok`.
#
# You can pass a specific `vocab` for the numericalization step (if you are building a classifier from a language model you fine-tuned for instance). kwargs will be split between the [`TextDataset`](/text.data.html#TextDataset) function and to the class initialization, you can precise there parameters such as `max_vocab`, `chunksize`, `min_freq`, `n_labels`, `tok_suff` and `lbl_suff` (see the [`TextDataset`](/text.data.html#TextDataset) documentation) or `bs`, `bptt` and `pad_idx` (see the sections LM data and classifier data).

show_doc(TextDataBunch.from_ids, doc_string=False)


# This function will create a [`DataBunch`](/basic_data.html#DataBunch) in `path` from texts already processed into `trn_ids`, `trn_lbls`, `val_ids`, `val_lbls` and maybe `tst_ids`. You can specify the corresponding `classes` if applciable. You must specify the `vocab` so that the [`RNNLearner`](/text.learner.html#RNNLearner) class can later infer the corresponding sizes in the model it will create. kwargs will be passed to the class initialization.

# ### Load and save

# To avoid losing time preprocessing the text data more than once, you should save/load your [`TextDataBunch`](/text.data.html#TextDataBunch) using thse methods.

show_doc(TextDataBunch.load)


show_doc(TextDataBunch.save)


# ### Example

# Untar the IMDB sample dataset if not already done:

path = untar_data(URLs.IMDB_SAMPLE)
path


# Since it comes in the form of csv files, we will use the corresponding `text_data` method. Here is an overview of what your file you should look like:

pd.read_csv(path / 'texts.csv').head()


# And here is a simple way of creating your [`DataBunch`](/basic_data.html#DataBunch) for language modelling or classification.

data_lm = TextLMDataBunch.from_csv(Path(path), 'texts.csv')
data_clas = TextClasDataBunch.from_csv(Path(path), 'texts.csv')


# ## The TextList input classes

# Behind the scenes, the previous functions will create a training, validation and maybe test [`TextList`](/text.data.html#TextList) that will be tokenized and numericalized (if needed) using [`PreProcessor`](/data_block.html#PreProcessor).

show_doc(Text, doc_string=False, title_level=3)


# Basic item for text data, contains the numericalized `ids` and the corresponding [`text`](/text.html#text).

show_doc(TextList, title_level=3)


# The basic [`ItemList`](/data_block.html#ItemList) for text data in `items` with the corresponding `vocab`.

show_doc(TextList.label_for_lm)


show_doc(TextList.from_folder)


show_doc(TextList.show_xys)


show_doc(TextList.show_xyzs)


show_doc(OpenFileProcessor, title_level=3)


# Simple `Preprocessor` that opens the files in items and reads the texts inside them.

show_doc(open_text)


show_doc(TokenizeProcessor, title_level=3)


# Simple [`PreProcessor`](/data_block.html#PreProcessor) that tokenizes the texts in `items` using `tokenizer` by bits of `chunsize`. If `mark_fields` is `True`, add field tokens.

show_doc(NumericalizeProcessor, title_level=3, doc_string=False)


# Numericalize the tokens with `vocab` (if not None) otherwise create one with `max_vocab` and `min_freq` from tokens.

# ## Language Model data

# A language model is trained to guess what the next word is inside a flow of words. We don't feed it the different texts separately but concatenate them all together in a big array. To create the batches, we split this array into `bs` chuncks of continuous texts. Note that in all NLP tasks, we use the pytoch convention of sequence length being the first dimension (and batch size being the second one) so we transpose that array so that we can read the chunks of texts in columns. Here is an example of batch from our imdb sample dataset.

path = untar_data(URLs.IMDB_SAMPLE)
data = TextLMDataBunch.from_csv(path, 'texts.csv')
x, y = next(iter(data.train_dl))
example = x[:20, :10].cpu()
texts = pd.DataFrame([data.train_ds.vocab.textify(l).split(' ') for l in example])
texts


# Then, as suggested in [this article](https://arxiv.org/abs/1708.02182) from Stephen Merity et al., we don't use a fixed `bptt` through the different batches but slightly change it from batch to batch.

iter_dl = iter(data.train_dl)
for _ in range(5):
    x, y = next(iter_dl)
    print(x.size())


# This is all done internally when we use [`TextLMDataBunch`](/text.data.html#TextLMDataBunch), by creating [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) using the following class:

show_doc(LanguageModelLoader, doc_string=False)


# Takes the texts from `dataset` and concatenate them all, then create a big array with `bs` columns (transposed from the data source so that we read the texts in the columns). Spits batches with a size approximately equal to `bptt` but changing at every batch. If `backwards` is True, reverses the original text. If `shuffle` is True, we shuffle the texts before concatenating them together at the start of each epoch. `max_len` is the maximum amount we add to `bptt`.

show_doc(LanguageModelLoader.batchify, doc_string=False)


# Called at the inialization to create the big array of text ids from the [`data`](/text.data.html#text.data) array.

show_doc(LanguageModelLoader.get_batch)


# ## Classifier data

# When preparing the data for a classifier, we keep the different texts separate, which poses another challenge for the creation of batches: since they don't all have the same length, we can't easily collate them together in batches. To help with this we use two different techniques:
# - padding: each text is padded with the `PAD` token to get all the ones we picked to the same size
# - sorting the texts (ish): to avoid having together a very long text with a very short one (which would then have a lot of `PAD` tokens), we regroup the texts by order of length. For the training set, we still add some randomness to avoid showing the same batches at every step of the training.
#
# Here is an example of batch with padding (the padding index is 1, and the padding is applied before the sentences start).

path = untar_data(URLs.IMDB_SAMPLE)
data = TextClasDataBunch.from_csv(path, 'texts.csv')
iter_dl = iter(data.train_dl)
_ = next(iter_dl)
x, y = next(iter_dl)
x[:20, -10:]


# This is all done internally when we use [`TextClasDataBunch`](/text.data.html#TextClasDataBunch), by using the following classes:

show_doc(SortSampler, doc_string=False)


# pytorch [`Sampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler) to batchify the `data_source` by order of length of the texts. Used for the validation and (if applicable) the test set.

show_doc(SortishSampler, doc_string=False)


# pytorch [`Sampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler) to batchify with size `bs` the `data_source` by order of length of the texts with a bit of randomness. Used for the training set.

show_doc(pad_collate, doc_string=False)


# Function used by the pytorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) to collate the `samples` in batches while adding padding with `pad_idx`. If `pad_first` is True, padding is applied at the beginning (before the sentence starts) otherwise it's applied at the end.

# ## Undocumented Methods - Methods moved below this line will intentionally be hidden

show_doc(TextLMDataBunch.create)


show_doc(TextClasDataBunch.create)


show_doc(TextList.new)


show_doc(TextList.get)


show_doc(TokenizeProcessor.process_one)


show_doc(TokenizeProcessor.process)


show_doc(OpenFileProcessor.process_one)


show_doc(NumericalizeProcessor.process)


show_doc(NumericalizeProcessor.process_one)


show_doc(TextList.reconstruct)


# ## New Methods - Please document or move to the undocumented section
