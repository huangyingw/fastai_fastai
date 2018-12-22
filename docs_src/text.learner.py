# coding: utf-8
# # NLP model creation and training
from fastai.gen_doc.nbdoc import *
from fastai.text import *
# The main thing here is [`RNNLearner`](/text.learner.html#RNNLearner). There are also some utility functions to help create and update text models.
# ## Quickly get a learner
show_doc(language_model_learner)
# `bptt` (for backprop trough time) is the number of words we will store the gradient for, and use for the optimization step.
#
# The model used is an [AWD-LSTM](https://arxiv.org/abs/1708.02182) that is built with embeddings of size `emb_sz`, a hidden size of `nh`, and `nl` layers (the `vocab_size` is inferred from the [`data`](/text.data.html#text.data)). All the dropouts are put to values that we found worked pretty well and you can control their strength by adjusting `drop_mult`. If <code>qrnn</code> is True, the model uses [QRNN cells](https://arxiv.org/abs/1611.01576) instead of LSTMs. The flag `tied_weights` control if we should use the same weights for the encoder and the decoder, the flag `bias` controls if the last linear layer (the decoder) has bias or not.
#
# You can specify `pretrained_model` if you want to use the weights of a pretrained model. If you have your own set of weights and the corrsesponding dictionary, you can pass them in `pretrained_fnames`. This should be a list of the name of the weight file and the name of the corresponding dictionary. The dictionary is needed because the function will internally convert the embeddings of the pretrained models to match the dictionary of the [`data`](/text.data.html#text.data) passed (a word may have a different id for the pretrained model). Those two files should be in the models directory of `data.path`.
path = untar_data(URLs.IMDB_SAMPLE)
data = TextLMDataBunch.from_csv(path, 'texts.csv')
learn = language_model_learner(data, pretrained_model=URLs.WT103, drop_mult=0.5)
show_doc(text_classifier_learner)
# `bptt` (for backprop trough time) is the number of words we will store the gradient for, and use for the optimization step.
#
# The model used is the encoder of an [AWD-LSTM](https://arxiv.org/abs/1708.02182) that is built with embeddings of size `emb_sz`, a hidden size of `nh`, and `nl` layers (the `vocab_size` is inferred from the [`data`](/text.data.html#text.data)). All the dropouts are put to values that we found worked pretty well and you can control their strength by adjusting `drop_mult`. If <code>qrnn</code> is True, the model uses [QRNN cells](https://arxiv.org/abs/1611.01576) instead of LSTMs.
#
# The input texts are fed into that model by bunch of `bptt` and only the last `max_len` activations are considerated. This gives us the backbone of our model. The head then consists of:
# - a layer that concatenates the final outputs of the RNN with the maximum and average of all the intermediate outputs (on the sequence length dimension),
# - blocks of ([`nn.BatchNorm1d`](https://pytorch.org/docs/stable/nn.html#torch.nn.BatchNorm1d), [`nn.Dropout`](https://pytorch.org/docs/stable/nn.html#torch.nn.Dropout), [`nn.Linear`](https://pytorch.org/docs/stable/nn.html#torch.nn.Linear), [`nn.ReLU`](https://pytorch.org/docs/stable/nn.html#torch.nn.ReLU)) layers.
#
# The blocks are defined by the `lin_ftrs` and `drops` arguments. Specifically, the first block will have a number of inputs inferred from the backbone arch and the last one will have a number of outputs equal to data.c (which contains the number of classes of the data) and the intermediate blocks have a number of inputs/outputs determined by `lin_ftrs` (of course a block has a number of inputs equal to the number of outputs of the previous block). The dropouts all have a the same value ps if you pass a float, or the corresponding values if you pass a list. Default is to have an intermediate hidden size of 50 (which makes two blocks model_activation -> 50 -> n_classes) with a dropout of 0.1.
jekyll_note("Using QRNN require to have cuda installed (same version as pytorhc is using).")
path = untar_data(URLs.IMDB_SAMPLE)
data = TextClasDataBunch.from_csv(path, 'texts.csv')
learn = text_classifier_learner(data, drop_mult=0.5)
show_doc(RNNLearner)
# Handles the whole creation from <code>data</code> and a `model` with a text data using a certain `bptt`. The `split_func` is used to properly split the model in different groups for gradual unfreezing and differential learning rates. Gradient clipping of `clip` is optionally applied. `adjust`, `alpha` and `beta` are all passed to create an instance of [`RNNTrainer`](/callbacks.rnn.html#RNNTrainer). Can be used for a language model or an RNN classifier. It also handles the conversion of weights from a pretrained model as well as saving or loading the encoder.
show_doc(RNNLearner.get_preds)
# If `ordered=True`, returns the predictions in the order of the dataset, otherwise they will be ordered by the sampler (from the longest text to the shortest). The other arguments are passed [`Learner.get_preds`](/basic_train.html#Learner.get_preds).
# ### Loading and saving
show_doc(RNNLearner.load_encoder)
show_doc(RNNLearner.save_encoder)
show_doc(RNNLearner.load_pretrained)
# Opens the weights in the `wgts_fname` of `self.model_dir` and the dictionary in `itos_fname` then adapts the pretrained weights to the vocabulary of the <code>data</code>. The two files should be in the models directory of the `learner.path`.
# ## Utility functions
show_doc(lm_split)
show_doc(rnn_classifier_split)
show_doc(convert_weights)
# Uses the dictionary `stoi_wgts` (mapping of word to id) of the weights to map them to a new dictionary `itos_new` (mapping id to word).
# ## Get predictions
show_doc(LanguageLearner, title_level=3)
show_doc(LanguageLearner.predict)
# If `no_unk=True` the unknown token is never picked. Words are taken randomly with the distribution of probabilities returned by the model. If `min_p` is not `None`, that value is the minimum probability to be considered in the pool of words. Lowering `temperature` will make the texts less randomized.
# ## Undocumented Methods - Methods moved below this line will intentionally be hidden
show_doc(RNNLearner.get_preds)
show_doc(LanguageLearner.show_results)
# ## New Methods - Please document or move to the undocumented section
