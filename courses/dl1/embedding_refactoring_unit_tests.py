# coding: utf-8
import torch
from torch.autograd import Variable
from fastai.rnn_reg import *
import numpy as np
def embedded_dropout(embed, words, dropout=0.1, scale=None):
    """ Applies dropout in the embedding layer by zeroing out some elements of the embedding vector.
    Uses the dropout_mask custom layer to achieve this.
    Args:
        embed (torch.nn.Embedding): An embedding torch layer
        words (torch.nn.Variable): A torch variable
        dropout (float): dropout fraction to apply to the embedding weights
        scale (float): additional scaling to apply to the modified embedding weights
    Returns:
        tensor of size: (batch_size x seq_length x embedding_size)
    Example:
    >> embed = torch.nn.Embedding(10,3)
    >> words = Variable(torch.LongTensor([[1,2,4,5] ,[4,3,2,9]]))
    >> words.size()
        (2,4)
    >> dropout_out_ = embedded_dropout(embed, words, dropout=0.40)
    >> dropout_out_
        Variable containing:
        (0 ,.,.) =
          1.2549  1.8230  1.9367
          0.0000 -0.0000  0.0000
          2.2540 -0.1299  1.5448
          0.0000 -0.0000 -0.0000
        (1 ,.,.) =
          2.2540 -0.1299  1.5448
         -4.0457  2.4815 -0.2897
          0.0000 -0.0000  0.0000
          1.8796 -0.4022  3.8773
        [torch.FloatTensor of size 2x4x3]
    """
    if dropout:
        mask = Variable(dropout_mask(embed.weight.data, (embed.weight.size(0), 1), dropout))
        masked_embed_weight = mask * embed.weight
    else: masked_embed_weight = embed.weight
    if scale: masked_embed_weight = scale * masked_embed_weight
    padding_idx = embed.padding_idx
    if padding_idx is None: padding_idx = -1
    X = embed._backend.Embedding.apply(words, masked_embed_weight,
        padding_idx, embed.max_norm, embed.norm_type,
        embed.scale_grad_by_freq, embed.sparse
    )
    return X
# ## Test 1
# #### Initialize embedding matrix and input
embed = torch.nn.Embedding(10, 3)
words = torch.autograd.Variable(torch.LongTensor([[1, 2, 4, 5], [4, 3, 2, 9]]))
# #### propagate the input via the old method (embedded_dropout)
torch.manual_seed(88123)
dropout_out_old = embedded_dropout(embed, words, dropout=0.40)
dropout_out_old
# #### propagate the input via the forward method in the new layer (EmbeddingDropout)
torch.manual_seed(88123)
embed_dropout_layer = EmbeddingDropout(embed)
dropout_out_new = embed_dropout_layer(words, dropout=0.4)
dropout_out_new
print(np.testing.assert_array_equal(to_np(dropout_out_old), to_np(dropout_out_new)))
# ## Test 2
# #### Initialize embedding and matrix
embed = torch.nn.Embedding(10, 7)
words = torch.autograd.Variable(torch.LongTensor([[1, 2, 4, 5, 2, 8], [4, 3, 2, 9, 7, 6]]))
# #### get the input by propagating via the old method
torch.manual_seed(7123)
dropout_out_old = embedded_dropout(embed, words, dropout=0.64)
dropout_out_old
# #### get the input by propagating input via the embedding layer
torch.manual_seed(7123)
embed_dropout_layer = EmbeddingDropout(embed)
dropout_out_new = embed_dropout_layer(words, dropout=0.64)
dropout_out_new
print(np.testing.assert_array_equal(to_np(dropout_out_old), to_np(dropout_out_new)))
