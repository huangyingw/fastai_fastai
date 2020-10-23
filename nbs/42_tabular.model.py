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

# hide
# skip
from nbdev.export import notebook2script
from nbdev.showdoc import *
from fastai.tabular.core import *
from fastai.torch_basics import *
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# export

# hide


# +
# default_exp tabular.model
# -

# # Tabular model
#
# > A basic model that can be used on tabular data

# ## Embeddings

# export
def emb_sz_rule(n_cat):
    "Rule of thumb to pick embedding size corresponding to `n_cat`"
    return min(600, round(1.6 * n_cat**0.56))


# export
def _one_emb_sz(classes, n, sz_dict=None):
    "Pick an embedding size for `n` depending on `classes` if not given in `sz_dict`."
    sz_dict = ifnone(sz_dict, {})
    n_cat = len(classes[n])
    sz = sz_dict.get(n, int(emb_sz_rule(n_cat)))  # rule of thumb
    return n_cat, sz


# Through trial and error, this general rule takes the lower of two values:
# * A dimension space of 600
# * A dimension space equal to 1.6 times the cardinality of the variable to 0.56.
#
# This provides a good starter for a good embedding space for your variables. For more advanced users who wish to lean into this practice, you can tweak these values to your discretion. It is not uncommon for slight adjustments to this general formula to provide more success.

# export
def get_emb_sz(to, sz_dict=None):
    "Get default embedding size from `TabularPreprocessor` `proc` or the ones in `sz_dict`"
    return [_one_emb_sz(to.classes, n, sz_dict) for n in to.cat_names]


# export
class TabularModel(Module):
    "Basic model for tabular data."

    def __init__(self, emb_szs, n_cont, out_sz, layers, ps=None, embed_p=0.,
                 y_range=None, use_bn=True, bn_final=False, bn_cont=True, act_cls=nn.ReLU(inplace=True)):
        ps = ifnone(ps, [0] * len(layers))
        if not is_listy(ps):
            ps = [ps] * len(layers)
        self.embeds = nn.ModuleList([Embedding(ni, nf) for ni, nf in emb_szs])
        self.emb_drop = nn.Dropout(embed_p)
        self.bn_cont = nn.BatchNorm1d(n_cont) if bn_cont else None
        n_emb = sum(e.embedding_dim for e in self.embeds)
        self.n_emb, self.n_cont = n_emb, n_cont
        sizes = [n_emb + n_cont] + layers + [out_sz]
        actns = [act_cls for _ in range(len(sizes) - 2)] + [None]
        _layers = [LinBnDrop(sizes[i], sizes[i + 1], bn=use_bn and (i != len(actns) - 1 or bn_final), p=p, act=a)
                   for i, (p, a) in enumerate(zip(ps + [0.], actns))]
        if y_range is not None:
            _layers.append(SigmoidRange(*y_range))
        self.layers = nn.Sequential(*_layers)

    def forward(self, x_cat, x_cont=None):
        if self.n_emb != 0:
            x = [e(x_cat[:, i]) for i, e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            if self.bn_cont is not None:
                x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        return self.layers(x)


# This model expects your `cat` and `cont` variables seperated. `cat` is passed through an `Embedding` layer and potential `Dropout`, while `cont` is passed though potential `BatchNorm1d`. Afterwards both are concatenated and passed through a series of `LinBnDrop`, before a final `Linear` layer corresponding to the expected outputs.

emb_szs = [(4, 2), (17, 8)]
m = TabularModel(emb_szs, n_cont=2, out_sz=2, layers=[200, 100]).eval()
x_cat = torch.tensor([[2, 12]]).long()
x_cont = torch.tensor([[0.7633, -0.1887]]).float()
out = m(x_cat, x_cont)


# export
@delegates(TabularModel.__init__)
def tabular_config(**kwargs):
    "Convenience function to easily create a config for `TabularModel`"
    return kwargs


# Any direct setup of `TabularModel`'s internals should be passed through here:

config = tabular_config(embed_p=0.6, use_bn=False)
config

# ## Export -

# hide
notebook2script()
