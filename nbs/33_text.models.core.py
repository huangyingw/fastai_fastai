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
from fastai.text.models.awdlstm import *
from fastai.text.core import *
from fastai.data.all import *
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# export

# hide

# +
# default_exp text.models.core
# default_cls_lvl 3
# -

# # Core text modules
#
# > Contain the modules common between different architectures and the generic functions to get models

# export
_model_meta = {AWD_LSTM: {'hid_name': 'emb_sz', 'url': URLs.WT103_FWD, 'url_bwd': URLs.WT103_BWD,
                          'config_lm': awd_lstm_lm_config, 'split_lm': awd_lstm_lm_split,
                          'config_clas': awd_lstm_clas_config, 'split_clas': awd_lstm_clas_split},
               AWD_QRNN: {'hid_name': 'emb_sz',
                          'config_lm': awd_qrnn_lm_config, 'split_lm': awd_lstm_lm_split,
                          'config_clas': awd_qrnn_clas_config, 'split_clas': awd_lstm_clas_split}, }
# Transformer: {'hid_name':'d_model', 'url':URLs.OPENAI_TRANSFORMER,
#               'config_lm':tfmer_lm_config, 'split_lm': tfmer_lm_split,
#               'config_clas':tfmer_clas_config, 'split_clas': tfmer_clas_split},
# TransformerXL: {'hid_name':'d_model',
#                'config_lm':tfmerXL_lm_config, 'split_lm': tfmerXL_lm_split,
#                'config_clas':tfmerXL_clas_config, 'split_clas': tfmerXL_clas_split}}


# ## Language models

# export
class LinearDecoder(Module):
    "To go on top of a RNNCore module and create a Language Model."
    initrange = 0.1

    def __init__(self, n_out, n_hid, output_p=0.1, tie_encoder=None, bias=True):
        self.decoder = nn.Linear(n_hid, n_out, bias=bias)
        self.decoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.output_dp = RNNDropout(output_p)
        if bias:
            self.decoder.bias.data.zero_()
        if tie_encoder:
            self.decoder.weight = tie_encoder.weight

    def forward(self, input):
        dp_inp = self.output_dp(input)
        return self.decoder(dp_inp), input, dp_inp



# +
enc = AWD_LSTM(100, 20, 10, 2)
x = torch.randint(0, 100, (10, 5))
r = enc(x)

tst = LinearDecoder(100, 20, 0.1)
y = tst(r)
test_eq(y[1], r)
test_eq(y[2].shape, r.shape)
test_eq(y[0].shape, [10, 5, 100])

tst = LinearDecoder(100, 20, 0.1, tie_encoder=enc.encoder)
test_eq(tst.decoder.weight, enc.encoder.weight)


# -

# export
class SequentialRNN(nn.Sequential):
    "A sequential module that passes the reset call to its children."
    def reset(self):
        for c in self.children():
            getattr(c, 'reset', noop)()


# +
class _TstMod(Module):
    def reset(self): print('reset')

tst = SequentialRNN(_TstMod(), _TstMod())
test_stdout(tst.reset, 'reset\nreset')


# -

# export
def get_language_model(arch, vocab_sz, config=None, drop_mult=1.):
    "Create a language model from `arch` and its `config`."
    meta = _model_meta[arch]
    config = ifnone(config, meta['config_lm']).copy()
    for k in config.keys():
        if k.endswith('_p'):
            config[k] *= drop_mult
    tie_weights, output_p, out_bias = map(config.pop, ['tie_weights', 'output_p', 'out_bias'])
    init = config.pop('init') if 'init' in config else None
    encoder = arch(vocab_sz, **config)
    enc = encoder.encoder if tie_weights else None
    decoder = LinearDecoder(vocab_sz, config[meta['hid_name']], output_p, tie_encoder=enc, bias=out_bias)
    model = SequentialRNN(encoder, decoder)
    return model if init is None else model.apply(init)


# The default `config` used can be found in `_model_meta[arch]['config_lm']`. `drop_mult` is applied to all the probabilities of dropout in that config.

# +
config = awd_lstm_lm_config.copy()
config.update({'n_hid': 10, 'emb_sz': 20})

tst = get_language_model(AWD_LSTM, 100, config=config)
x = torch.randint(0, 100, (10, 5))
y = tst(x)
test_eq(y[0].shape, [10, 5, 100])
test_eq(y[1].shape, [10, 5, 20])
test_eq(y[2].shape, [10, 5, 20])
test_eq(tst[1].decoder.weight, tst[0].encoder.weight)
# -

# test drop_mult
tst = get_language_model(AWD_LSTM, 100, config=config, drop_mult=0.5)
test_eq(tst[1].output_dp.p, config['output_p'] * 0.5)
for rnn in tst[0].rnns:
    test_eq(rnn.weight_p, config['weight_p'] * 0.5)
for dp in tst[0].hidden_dps:
    test_eq(dp.p, config['hidden_p'] * 0.5)
test_eq(tst[0].encoder_dp.embed_p, config['embed_p'] * 0.5)
test_eq(tst[0].input_dp.p, config['input_p'] * 0.5)


# ## Classification models

# export
def _pad_tensor(t, bs):
    if t.size(0) < bs:
        return torch.cat([t, t.new_zeros(bs - t.size(0), *t.shape[1:])])
    return t


# export
class SentenceEncoder(Module):
    "Create an encoder over `module` that can process a full sentence."
    def __init__(self, bptt, module, pad_idx=1, max_len=None): store_attr('bptt,module,pad_idx,max_len')
    def reset(self): getattr(self.module, 'reset', noop)()

    def forward(self, input):
        bs, sl = input.size()
        self.reset()
        mask = input == self.pad_idx
        outs, masks = [], []
        for i in range(0, sl, self.bptt):
            # Note: this expects that sequence really begins on a round multiple of bptt
            real_bs = (input[:, i] != self.pad_idx).long().sum()
            o = self.module(input[:real_bs, i: min(i + self.bptt, sl)])
            if self.max_len is None or sl - i <= self.max_len:
                outs.append(o)
                masks.append(mask[:, i: min(i + self.bptt, sl)])
        outs = torch.cat([_pad_tensor(o, bs) for o in outs], dim=1)
        mask = torch.cat(masks, dim=1)
        return outs, mask


# > Warning: This module expects the inputs padded with most of the padding first, with the sequence beginning at a round multiple of `bptt` (and the rest of the padding at the end). Use `pad_input_chunk` to get your data in a suitable format.

# +
mod = nn.Embedding(5, 10)
tst = SentenceEncoder(5, mod, pad_idx=0)
x = torch.randint(1, 5, (3, 15))
x[2, :5] = 0
out, mask = tst(x)

test_eq(out[:1], mod(x)[:1])
test_eq(out[2, 5:], mod(x)[2, 5:])
test_eq(mask, x == 0)


# -

# export
def masked_concat_pool(output, mask, bptt):
    "Pool `MultiBatchEncoder` outputs into one vector [last_hidden, max_pool, avg_pool]"
    lens = output.shape[1] - mask.long().sum(dim=1)
    last_lens = mask[:, -bptt:].long().sum(dim=1)
    avg_pool = output.masked_fill(mask[:, :, None], 0).sum(dim=1)
    avg_pool.div_(lens.type(avg_pool.dtype)[:, None])
    max_pool = output.masked_fill(mask[:, :, None], -float('inf')).max(dim=1)[0]
    x = torch.cat([output[torch.arange(0, output.size(0)), -last_lens - 1], max_pool, avg_pool], 1)  # Concat pooling.
    return x


# +
out = torch.randn(2, 4, 5)
mask = tensor([[True, True, False, False], [False, False, False, True]])
x = masked_concat_pool(out, mask, 2)

test_close(x[0, :5], out[0, -1])
test_close(x[1, :5], out[1, -2])
test_close(x[0, 5:10], out[0, 2:].max(dim=0)[0])
test_close(x[1, 5:10], out[1, :3].max(dim=0)[0])
test_close(x[0, 10:], out[0, 2:].mean(dim=0))
test_close(x[1, 10:], out[1, :3].mean(dim=0))
# -

# Test the result is independent of padding by replacing the padded part by some random content
out1 = torch.randn(2, 4, 5)
out1[0, 2:] = out[0, 2:].clone()
out1[1, :3] = out[1, :3].clone()
x1 = masked_concat_pool(out1, mask, 2)
test_eq(x, x1)


# export
class PoolingLinearClassifier(Module):
    "Create a linear classifier with pooling"
    def __init__(self, dims, ps, bptt, y_range=None):
        if len(ps) != len(dims) - 1:
            raise ValueError("Number of layers and dropout values do not match.")
        acts = [nn.ReLU(inplace=True)] * (len(dims) - 2) + [None]
        layers = [LinBnDrop(i, o, p=p, act=a) for i, o, p, a in zip(dims[:-1], dims[1:], ps, acts)]
        if y_range is not None:
            layers.append(SigmoidRange(*y_range))
        self.layers = nn.Sequential(*layers)
        self.bptt = bptt

    def forward(self, input):
        out, mask = input
        x = masked_concat_pool(out, mask, self.bptt)
        x = self.layers(x)
        return x, out, out


# +
mod = nn.Embedding(5, 10)
tst = SentenceEncoder(5, mod, pad_idx=0)
x = torch.randint(1, 5, (3, 15))
x[2, :5] = 0
out, mask = tst(x)

test_eq(out[:1], mod(x)[:1])
test_eq(out[2, 5:], mod(x)[2, 5:])
test_eq(mask, x == 0)

# +
# hide
mod = nn.Embedding(5, 10)
tst = nn.Sequential(SentenceEncoder(5, mod, pad_idx=0), PoolingLinearClassifier([10 * 3, 4], [0.], 5))

x = torch.randint(1, 5, (3, 14))
x[2, :5] = 0
res, raw, out = tst(x)

test_eq(raw[:1], mod(x)[:1])
test_eq(raw[2, 5:], mod(x)[2, 5:])
test_eq(out[:1], mod(x)[:1])
test_eq(out[2, 5:], mod(x)[2, 5:])
test_eq(res.shape, [3, 4])

x1 = torch.cat([x, tensor([0, 0, 0])[:, None]], dim=1)
res1, raw1, out1 = tst(x1)
test_eq(res, res1)


# -

# export
def get_text_classifier(arch, vocab_sz, n_class, seq_len=72, config=None, drop_mult=1., lin_ftrs=None,
                        ps=None, pad_idx=1, max_len=72 * 20, y_range=None):
    "Create a text classifier from `arch` and its `config`, maybe `pretrained`"
    meta = _model_meta[arch]
    config = ifnone(config, meta['config_clas']).copy()
    for k in config.keys():
        if k.endswith('_p'):
            config[k] *= drop_mult
    if lin_ftrs is None:
        lin_ftrs = [50]
    if ps is None:
        ps = [0.1] * len(lin_ftrs)
    layers = [config[meta['hid_name']] * 3] + lin_ftrs + [n_class]
    ps = [config.pop('output_p')] + ps
    init = config.pop('init') if 'init' in config else None
    encoder = SentenceEncoder(seq_len, arch(vocab_sz, **config), pad_idx=pad_idx, max_len=max_len)
    model = SequentialRNN(encoder, PoolingLinearClassifier(layers, ps, bptt=seq_len, y_range=y_range))
    return model if init is None else model.apply(init)


# +
config = awd_lstm_clas_config.copy()
config.update({'n_hid': 10, 'emb_sz': 20})

tst = get_text_classifier(AWD_LSTM, 100, 3, config=config)
x = torch.randint(2, 100, (10, 5))
y = tst(x)
test_eq(y[0].shape, [10, 3])
test_eq(y[1].shape, [10, 5, 20])
test_eq(y[2].shape, [10, 5, 20])
# -

# test padding gives same results
tst.eval()
y = tst(x)
x1 = torch.cat([x, tensor([2, 1, 1, 1, 1, 1, 1, 1, 1, 1])[:, None]], dim=1)
y1 = tst(x1)
test_close(y[0][1:], y1[0][1:])

# test drop_mult
tst = get_text_classifier(AWD_LSTM, 100, 3, config=config, drop_mult=0.5)
test_eq(tst[1].layers[1][1].p, 0.1)
test_eq(tst[1].layers[0][1].p, config['output_p'] * 0.5)
for rnn in tst[0].module.rnns:
    test_eq(rnn.weight_p, config['weight_p'] * 0.5)
for dp in tst[0].module.hidden_dps:
    test_eq(dp.p, config['hidden_p'] * 0.5)
test_eq(tst[0].module.encoder_dp.embed_p, config['embed_p'] * 0.5)
test_eq(tst[0].module.input_dp.p, config['input_p'] * 0.5)

# ## Export -

# hide
notebook2script()
