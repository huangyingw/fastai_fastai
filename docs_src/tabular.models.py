# coding: utf-8
# # Simple model for tabular data
from fastai.gen_doc.nbdoc import *
from fastai.tabular.models import TabularModel
show_doc(TabularModel)
# `emb_szs` match each categorical variable size with an embedding size, `n_cont` is the number of continuous variables. The model consists of `Embedding` layers for the categorical variables, followed by a `Dropout` of `emb_drop`, and a `BatchNorm` for the continuous variables. The results are concatenated and followed by blocks of `BatchNorm`, `Dropout`, `Linear` and `ReLU` (the first block skips `BatchNorm` and `Dropout`, the last block skips the `ReLU`).
#
# The sizes of the blocks are given in [`layers`](/layers.html#layers) and the probabilities of the `Dropout` in `ps`. The last size is `out_sz`, and we add a last activation that is a sigmoid rescaled to cover `y_range` (if it's not `None`). Lastly, if `use_bn` is set to False, all `BatchNorm` layers are skipped except the one applied to the continuous variables.
#
# Generally it's easiest to just create a learner with [`tabular_learner`](/tabular.data.html#tabular_learner), which will automatically create a [`TabularModel`](/tabular.models.html#TabularModel) for you.
# ## Undocumented Methods - Methods moved below this line will intentionally be hidden
show_doc(TabularModel.forward)
show_doc(TabularModel.get_sizes)
# ## New Methods - Please document or move to the undocumented section
