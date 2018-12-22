# coding: utf-8
# # Implementation of the language models
from fastai.gen_doc.nbdoc import *
from fastai.text import *
from fastai.text.models import *
# [`text.models`](/text.models.html#text.models) module fully implements the [AWD-LSTM](https://arxiv.org/pdf/1708.02182.pdf) from Stephen Merity et al. The main idea of the article is to use a [RNN](http://www.pnas.org/content/79/8/2554) with dropout everywhere, but in an intelligent way. There is a difference with the usual dropout, which is why you’ll see a [`RNNDropout`](/text.models.html#RNNDropout) module: we zero things, as is usual in dropout, but we always zero the same thing according to the sequence dimension (which is the first dimension in pytorch). This ensures consistency when updating the hidden state through the whole sentences/articles.
#
# This being given, there are five different dropouts in the AWD-LSTM:
#
# - the first one, embedding dropout, is applied when we look the ids of our tokens inside the embedding matrix (to transform them from numbers to a vector of float). We zero some lines of it, so random ids are sent to a vector of zeros instead of being sent to their embedding vector.
# - the second one, input dropout, is applied to the result of the embedding with dropout. We forget random pieces of the embedding matrix (but as stated in the last paragraph, the same ones in the sequence dimension).
# - the third one is the weight dropout. It’s the trickiest to implement as we randomly replace by 0s some weights of the hidden-to-hidden matrix inside the RNN: this needs to be done in a way that ensure the gradients are still computed and the initial weights still updated.
# - the fourth one is the hidden dropout. It’s applied to the output of one of the layers of the RNN before it’s used as input of the next layer (again same coordinates are zeroed in the sequence dimension). This one isn’t applied to the last output, but rather…
# - the fifth one is the output dropout, it’s applied to the last output of the model (and like the others, it’s applied the same way through the first dimension).
# ## Basic functions to get a model
show_doc(get_language_model)
# The first embedding of `vocab_sz` by `emb_sz`, a hidden size of `n_hid`, RNNs with `n_layers` that can be bidirectional if `bidir` is True. The last RNN as an output size of `emb_sz` so that we can use the same decoder as the encoder if `tie_weights` is True. The decoder is a `Linear` layer with or without `bias`. If `qrnn` is set to True, we use [QRNN cells] instead of LSTMS. `pad_token` is the token used for padding.
#
# `embed_p` is used for the embedding dropout, `input_p` is used for the input dropout, `weight_p` is used for the weight dropout, `hidden_p` is used for the hidden dropout and `output_p` is used for the output dropout.
#
# Note that the model returns a list of three things, the actual output being the first, the two others being the intermediate hidden states before and after dropout (used by the [`RNNTrainer`](/callbacks.rnn.html#RNNTrainer)). Most loss functions expect one output, so you should use a Callback to remove the other two if you're not using [`RNNTrainer`](/callbacks.rnn.html#RNNTrainer).
show_doc(get_rnn_classifier)
# This model uses an encoder taken from an AWD-LSTM with arguments `vocab_sz`, `emb_sz`, `n_hid`, `n_layers`, `bias`, `bidir`, `qrnn`, `pad_token` and the dropouts parameters. This encoder is fed the sequence by successive bits of size `bptt` and we only keep the last `max_seq` outputs for the pooling layers.
#
# The decoder use a concatenation of the last outputs, a `MaxPooling` of all the ouputs and an `AveragePooling` of all the outputs. It then uses a list of `BatchNorm`, `Dropout`, `Linear`, `ReLU` blocks (with no `ReLU` in the last one), using a first layer size of `3*emb_sz` then follwoing the numbers in `n_layers` to stop at `n_class`. The dropouts probabilities are read in `drops`.
#
# Note that the model returns a list of three things, the actual output being the first, the two others being the intermediate hidden states before and after dropout (used by the [`RNNTrainer`](/callbacks.rnn.html#RNNTrainer)). Most loss functions expect one output, so you should use a Callback to remove the other two if you're not using [`RNNTrainer`](/callbacks.rnn.html#RNNTrainer).
# ## Basic NLP modules
# On top of the pytorch or the fastai [`layers`](/layers.html#layers), the language models use some custom layers specific to NLP.
show_doc(EmbeddingDropout, title_level=3)
# Each row of the embedding matrix has a probability `embed_p` of being replaced by zeros while the others are rescaled accordingly.
enc = nn.Embedding(100, 7, padding_idx=1)
enc_dp = EmbeddingDropout(enc, 0.5)
tst_input = torch.randint(0, 100, (8,))
enc_dp(tst_input)
show_doc(RNNDropout, title_level=3)
dp = RNNDropout(0.3)
tst_input = torch.randn(3, 3, 7)
tst_input, dp(tst_input)
show_doc(WeightDropout, title_level=3)
# Applies dropout of probability `weight_p` to the layers in `layer_names` of `module` in training mode. A copy of those weights is kept so that the dropout mask can change at every batch.
module = nn.LSTM(5, 2)
dp_module = WeightDropout(module, 0.4)
getattr(dp_module.module, 'weight_hh_l0')
# It's at the beginning of a forward pass that the dropout is applied to the weights.
tst_input = torch.randn(4, 20, 5)
h = (torch.zeros(1, 20, 2), torch.zeros(1, 20, 2))
x, h = dp_module(tst_input, h)
getattr(dp_module.module, 'weight_hh_l0')
show_doc(SequentialRNN, title_level=3)
show_doc(SequentialRNN.reset)
# Call the `reset` function of [`self.children`](/torch_core.html#children) (if they have one).
show_doc(dropout_mask)
tst_input = torch.randn(3, 3, 7)
dropout_mask(tst_input, (3, 7), 0.3)
# Such a mask is then expanded in the sequence length dimension and multiplied by the input to do an [`RNNDropout`](/text.models.html#RNNDropout).
# ## Language model modules
show_doc(RNNCore, title_level=3)
# This is the encoder of the model with an embedding layer of `vocab_sz` by `emb_sz`, a hidden size of `n_hid`, `n_layers` layers. `pad_token` is passed to the `Embedding`, if `bidir` is True, the model is bidirectional. If `qrnn` is True, we use QRNN cells instead of LSTMs. Dropouts are `embed_p`, `input_p`, `weight_p` and `hidden_p`.
show_doc(RNNCore.reset)
show_doc(LinearDecoder, title_level=3)
# Create a the decoder to go on top of an [`RNNCore`](/text.models.html#RNNCore) encoder and create a language model. `n_hid` is the dimension of the last hidden state of the encoder, `n_out` the size of the output. Dropout of `output_p` is applied. If a `tie_encoder` is passed, it will be used for the weights of the linear layer, that will have `bias` or not.
# ## Classifier modules
show_doc(MultiBatchRNNCore, title_level=3)
# Text is passed by chunks of sequence length `bptt` and only the last `max_seq` outputs are kept for the next layer. `args` and `kwargs` are passed to the [`RNNCore`](/text.models.html#RNNCore).
show_doc(MultiBatchRNNCore.concat)
show_doc(PoolingLinearClassifier, title_level=3)
# The last output, `MaxPooling` of all the outputs and `AvgPooling` of all the outputs are concatenated, then blocks of [`bn_drop_lin`](/layers.html#bn_drop_lin) are stacked, according to the values in [`layers`](/layers.html#layers) and `drops`.
show_doc(PoolingLinearClassifier.pool)
# The input tensor `x` (of batch size `bs`) is pooled along the batch dimension. `is_max` decides if we do an `AvgPooling` or a `MaxPooling`.
# ## Undocumented Methods - Methods moved below this line will intentionally be hidden
show_doc(WeightDropout.forward)
show_doc(RNNCore.forward)
show_doc(EmbeddingDropout.forward)
show_doc(RNNDropout.forward)
show_doc(WeightDropout.reset)
show_doc(PoolingLinearClassifier.forward)
show_doc(MultiBatchRNNCore.forward)
show_doc(LinearDecoder.forward)
# ## New Methods - Please document or move to the undocumented section
