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
from fastai.text.models.core import _model_meta
from nbdev.showdoc import *
from fastai.callback.progress import *
from fastai.callback.rnn import *
from fastai.text.models.awdlstm import *
from fastai.text.models.core import *
from fastai.text.data import *
from fastai.text.core import *
from fastai.basics import *
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# export

# hide


# +
# default_exp text.learner
# -

# # Learner for the text application
#
# > All the functions necessary to build `Learner` suitable for transfer learning in NLP

# The most important functions of this module are `language_model_learner` and `text_classifier_learner`. They will help you define a `Learner` using a pretrained model. See the [text tutorial](http://docs.fast.ai/tutorial.text) for exmaples of use.

# ## Loading a pretrained model

# In text, to load a pretrained model, we need to adapt the embeddings of the vocabulary used for the pre-training to the vocabulary of our current corpus.

# export
def match_embeds(old_wgts, old_vocab, new_vocab):
    "Convert the embedding in `old_wgts` to go from `old_vocab` to `new_vocab`."
    bias, wgts = old_wgts.get('1.decoder.bias', None), old_wgts['0.encoder.weight']
    wgts_m = wgts.mean(0)
    new_wgts = wgts.new_zeros((len(new_vocab), wgts.size(1)))
    if bias is not None:
        bias_m = bias.mean(0)
        new_bias = bias.new_zeros((len(new_vocab),))
    old_o2i = old_vocab.o2i if hasattr(old_vocab, 'o2i') else {w: i for i, w in enumerate(old_vocab)}
    for i, w in enumerate(new_vocab):
        idx = old_o2i.get(w, -1)
        new_wgts[i] = wgts[idx] if idx >= 0 else wgts_m
        if bias is not None:
            new_bias[i] = bias[idx] if idx >= 0 else bias_m
    old_wgts['0.encoder.weight'] = new_wgts
    if '0.encoder_dp.emb.weight' in old_wgts:
        old_wgts['0.encoder_dp.emb.weight'] = new_wgts.clone()
    old_wgts['1.decoder.weight'] = new_wgts.clone()
    if bias is not None:
        old_wgts['1.decoder.bias'] = new_bias
    return old_wgts


# For words in `new_vocab` that don't have a corresponding match in `old_vocab`, we use the mean of all pretrained embeddings.

wgts = {'0.encoder.weight': torch.randn(5, 3)}
new_wgts = match_embeds(wgts.copy(), ['a', 'b', 'c'], ['a', 'c', 'd', 'b'])
old, new = wgts['0.encoder.weight'], new_wgts['0.encoder.weight']
test_eq(new[0], old[0])
test_eq(new[1], old[2])
test_eq(new[2], old.mean(0))
test_eq(new[3], old[1])

# hide
# With bias
wgts = {'0.encoder.weight': torch.randn(5, 3), '1.decoder.bias': torch.randn(5)}
new_wgts = match_embeds(wgts.copy(), ['a', 'b', 'c'], ['a', 'c', 'd', 'b'])
old_w, new_w = wgts['0.encoder.weight'], new_wgts['0.encoder.weight']
old_b, new_b = wgts['1.decoder.bias'], new_wgts['1.decoder.bias']
test_eq(new_w[0], old_w[0])
test_eq(new_w[1], old_w[2])
test_eq(new_w[2], old_w.mean(0))
test_eq(new_w[3], old_w[1])
test_eq(new_b[0], old_b[0])
test_eq(new_b[1], old_b[2])
test_eq(new_b[2], old_b.mean(0))
test_eq(new_b[3], old_b[1])


# export
def _get_text_vocab(dls):
    vocab = dls.vocab
    if isinstance(vocab, L):
        vocab = vocab[0]
    return vocab


# export
def load_ignore_keys(model, wgts):
    "Load `wgts` in `model` ignoring the names of the keys, just taking parameters in order"
    sd = model.state_dict()
    for k1, k2 in zip(sd.keys(), wgts.keys()):
        sd[k1].data = wgts[k2].data.clone()
    return model.load_state_dict(sd)


# export
def _rm_module(n):
    t = n.split('.')
    for i in range(len(t) - 1, -1, -1):
        if t[i] == 'module':
            t.pop(i)
            break
    return '.'.join(t)


# export
# For previous versions compatibility, remove for release
def clean_raw_keys(wgts):
    keys = list(wgts.keys())
    for k in keys:
        t = k.split('.module')
        if f'{_rm_module(k)}_raw' in keys:
            del wgts[k]
    return wgts


# export
# For previous versions compatibility, remove for release
def load_model_text(file, model, opt, with_opt=None, device=None, strict=True):
    "Load `model` from `file` along with `opt` (if available, and if `with_opt`)"
    distrib_barrier()
    if isinstance(device, int):
        device = torch.device('cuda', device)
    elif device is None:
        device = 'cpu'
    state = torch.load(file, map_location=device)
    hasopt = set(state) == {'model', 'opt'}
    model_state = state['model'] if hasopt else state
    get_model(model).load_state_dict(clean_raw_keys(model_state), strict=strict)
    if hasopt and ifnone(with_opt, True):
        try:
            opt.load_state_dict(state['opt'])
        except:
            if with_opt:
                warn("Could not load the optimizer state.")
    elif with_opt:
        warn("Saved filed doesn't contain an optimizer state.")


# export
@log_args(but_as=Learner.__init__)
@delegates(Learner.__init__)
class TextLearner(Learner):
    "Basic class for a `Learner` in NLP."

    def __init__(self, dls, model, alpha=2., beta=1., moms=(0.8, 0.7, 0.8), **kwargs):
        super().__init__(dls, model, moms=moms, **kwargs)
        self.add_cbs([ModelResetter(), RNNRegularizer(alpha=alpha, beta=beta)])

    def save_encoder(self, file):
        "Save the encoder to `file` in the model directory"
        if rank_distrib():
            return  # don't save if child proc
        encoder = get_model(self.model)[0]
        if hasattr(encoder, 'module'):
            encoder = encoder.module
        torch.save(encoder.state_dict(), join_path_file(file, self.path / self.model_dir, ext='.pth'))

    def load_encoder(self, file, device=None):
        "Load the encoder `file` from the model directory, optionally ensuring it's on `device`"
        encoder = get_model(self.model)[0]
        if device is None:
            device = self.dls.device
        if hasattr(encoder, 'module'):
            encoder = encoder.module
        distrib_barrier()
        wgts = torch.load(join_path_file(file, self.path / self.model_dir, ext='.pth'), map_location=device)
        encoder.load_state_dict(clean_raw_keys(wgts))
        self.freeze()
        return self

    def load_pretrained(self, wgts_fname, vocab_fname, model=None):
        "Load a pretrained model and adapt it to the data vocabulary."
        old_vocab = load_pickle(vocab_fname)
        new_vocab = _get_text_vocab(self.dls)
        distrib_barrier()
        wgts = torch.load(wgts_fname, map_location=lambda storage, loc: storage)
        if 'model' in wgts:
            wgts = wgts['model']  # Just in case the pretrained model was saved with an optimizer
        wgts = match_embeds(wgts, old_vocab, new_vocab)
        load_ignore_keys(self.model if model is None else model, clean_raw_keys(wgts))
        self.freeze()
        return self

    # For previous versions compatibility. Remove at release
    @delegates(load_model_text)
    def load(self, file, with_opt=None, device=None, **kwargs):
        if device is None:
            device = self.dls.device
        if self.opt is None:
            self.create_opt()
        file = join_path_file(file, self.path / self.model_dir, ext='.pth')
        load_model_text(file, self.model, self.opt, device=device, **kwargs)
        return self


# Adds a `ModelResetter` and an `RNNRegularizer` with `alpha` and `beta` to the callbacks, the rest is the same as `Learner` init.
#
# This `Learner` adds functionality to the base class:

show_doc(TextLearner.load_pretrained)

# `wgts_fname` should point to the weights of the pretrained model and `vocab_fname` to the vocabulary used to pretrain it.

show_doc(TextLearner.save_encoder)

# The model directory is `Learner.path/Learner.model_dir`.

show_doc(TextLearner.load_encoder)


# ## Language modeling predictions

# For language modeling, the predict method is quite different form the other applications, which is why it needs its own subclass.

# export
def decode_spec_tokens(tokens):
    "Decode the special tokens in `tokens`"
    new_toks, rule, arg = [], None, None
    for t in tokens:
        if t in [TK_MAJ, TK_UP, TK_REP, TK_WREP]:
            rule = t
        elif rule is None:
            new_toks.append(t)
        elif rule == TK_MAJ:
            new_toks.append(t[:1].upper() + t[1:].lower())
            rule = None
        elif rule == TK_UP:
            new_toks.append(t.upper())
            rule = None
        elif arg is None:
            try:
                arg = int(t)
            except:
                rule = None
        else:
            if rule == TK_REP:
                new_toks.append(t * arg)
            else:
                new_toks += [t] * arg
    return new_toks


test_eq(decode_spec_tokens(['xxmaj', 'text']), ['Text'])
test_eq(decode_spec_tokens(['xxup', 'text']), ['TEXT'])
test_eq(decode_spec_tokens(['xxrep', '3', 'a']), ['aaa'])
test_eq(decode_spec_tokens(['xxwrep', '3', 'word']), ['word', 'word', 'word'])


# export
@log_args(but_as=TextLearner.__init__)
class LMLearner(TextLearner):
    "Add functionality to `TextLearner` when dealing with a language model"

    def predict(self, text, n_words=1, no_unk=True, temperature=1., min_p=None, no_bar=False,
                decoder=decode_spec_tokens, only_last_word=False):
        "Return `text` and the `n_words` that come after"
        self.model.reset()
        idxs = idxs_all = self.dls.test_dl([text]).items[0].to(self.dls.device)
        if no_unk:
            unk_idx = self.dls.vocab.index(UNK)
        for _ in (range(n_words) if no_bar else progress_bar(range(n_words), leave=False)):
            with self.no_bar():
                preds, _ = self.get_preds(dl=[(idxs[None],)])
            res = preds[0][-1]
            if no_unk:
                res[unk_idx] = 0.
            if min_p is not None:
                if (res >= min_p).float().sum() == 0:
                    warn(f"There is no item with probability >= {min_p}, try a lower value.")
                else:
                    res[res < min_p] = 0.
            if temperature != 1.:
                res.pow_(1 / temperature)
            idx = torch.multinomial(res, 1).item()
            idxs = idxs_all = torch.cat([idxs_all, idxs.new([idx])])
            if only_last_word:
                idxs = idxs[-1][None]

        num = self.dls.train_ds.numericalize
        tokens = [num.vocab[i] for i in idxs_all if num.vocab[i] not in [BOS, PAD]]
        sep = self.dls.train_ds.tokenizer.sep
        return sep.join(decoder(tokens))

    @delegates(Learner.get_preds)
    def get_preds(self, concat_dim=1, **kwargs): return super().get_preds(concat_dim=1, **kwargs)


show_doc(LMLearner, title_level=3)

show_doc(LMLearner.predict)

# The words are picked randomly among the predictions, depending on the probability of each index. `no_unk` means we never pick the `UNK` token, `temperature` is applied to the predictions, if `min_p` is passed, we don't consider the indices with a probability lower than it. Set `no_bar` to `True` if you don't want any progress bar, and you can pass a long a custom `decoder` to process the predicted tokens.

# ## `Learner` convenience functions

# export


# export
def _get_text_vocab(dls):
    vocab = dls.vocab
    if isinstance(vocab, L):
        vocab = vocab[0]
    return vocab


# export
@log_args(to_return=True, but_as=Learner.__init__)
@delegates(Learner.__init__)
def language_model_learner(dls, arch, config=None, drop_mult=1., backwards=False, pretrained=True, pretrained_fnames=None, **kwargs):
    "Create a `Learner` with a language model from `dls` and `arch`."
    vocab = _get_text_vocab(dls)
    model = get_language_model(arch, len(vocab), config=config, drop_mult=drop_mult)
    meta = _model_meta[arch]
    learn = LMLearner(dls, model, loss_func=CrossEntropyLossFlat(), splitter=meta['split_lm'], **kwargs)
    url = 'url_bwd' if backwards else 'url'
    if pretrained or pretrained_fnames:
        if pretrained_fnames is not None:
            fnames = [learn.path / learn.model_dir / f'{fn}.{ext}' for fn, ext in zip(pretrained_fnames, ['pth', 'pkl'])]
        else:
            if url not in meta:
                warn("There are no pretrained weights for that architecture yet!")
                return learn
            model_path = untar_data(meta[url], c_key='model')
            try:
                fnames = [list(model_path.glob(f'*.{ext}'))[0] for ext in ['pth', 'pkl']]
            except IndexError:
                print(f'The model in {model_path} is incomplete, download again')
                raise
        learn = learn.load_pretrained(*fnames)
    return learn


# You can use the `config` to customize the architecture used (change the values from `awd_lstm_lm_config` for this), `pretrained` will use fastai's pretrained model for this `arch` (if available) or you can pass specific `pretrained_fnames` containing your own pretrained model and the corresponding vocabulary. All other arguments are passed to `Learner`.

path = untar_data(URLs.IMDB_SAMPLE)
df = pd.read_csv(path / 'texts.csv')
dls = TextDataLoaders.from_df(df, path=path, text_col='text', is_lm=True, valid_col='is_valid')
learn = language_model_learner(dls, AWD_LSTM)

# You can then use the `.predict` method to generate new text.

learn.predict('This movie is about', n_words=20)

# By default the entire sentence is feed again to the model after each predicted word, this little trick shows an improvement on the quality of the generated text. If you want to feed only the last word, specify argument `only_last_word`.

learn.predict('This movie is about', n_words=20, only_last_word=True)


# export
@log_args(to_return=True, but_as=Learner.__init__)
@delegates(Learner.__init__)
def text_classifier_learner(dls, arch, seq_len=72, config=None, backwards=False, pretrained=True, drop_mult=0.5, n_out=None,
                            lin_ftrs=None, ps=None, max_len=72 * 20, y_range=None, **kwargs):
    "Create a `Learner` with a text classifier from `dls` and `arch`."
    vocab = _get_text_vocab(dls)
    if n_out is None:
        n_out = get_c(dls)
    assert n_out, "`n_out` is not defined, and could not be inferred from data, set `dls.c` or pass `n_out`"
    model = get_text_classifier(arch, len(vocab), n_out, seq_len=seq_len, config=config, y_range=y_range,
                                drop_mult=drop_mult, lin_ftrs=lin_ftrs, ps=ps, max_len=max_len)
    meta = _model_meta[arch]
    learn = TextLearner(dls, model, splitter=meta['split_clas'], **kwargs)
    url = 'url_bwd' if backwards else 'url'
    if pretrained:
        if url not in meta:
            warn("There are no pretrained weights for that architecture yet!")
            return learn
        model_path = untar_data(meta[url], c_key='model')
        try:
            fnames = [list(model_path.glob(f'*.{ext}'))[0] for ext in ['pth', 'pkl']]
        except IndexError:
            print(f'The model in {model_path} is incomplete, download again')
            raise
        learn = learn.load_pretrained(*fnames, model=learn.model[0])
        learn.freeze()
    return learn


# You can use the `config` to customize the architecture used (change the values from `awd_lstm_clas_config` for this), `pretrained` will use fastai's pretrained model for this `arch` (if available). `drop_mult` is a global multiplier applied to control all dropouts. `n_out` is usually inferred from the `dls` but you may pass it.
#
# The model uses a `SentenceEncoder`, which means the texts are passed `seq_len` tokens at a time, and will only compute the gradients on the last `max_len` steps. `lin_ftrs` and `ps` are passed to `get_text_classifier`.
#
# All other arguments are passed to `Learner`.

path = untar_data(URLs.IMDB_SAMPLE)
df = pd.read_csv(path / 'texts.csv')
dls = TextDataLoaders.from_df(df, path=path, text_col='text', label_col='label', valid_col='is_valid')
learn = text_classifier_learner(dls, AWD_LSTM)


# ## Show methods -

# export
@typedispatch
def show_results(x: LMTensorText, y, samples, outs, ctxs=None, max_n=10, **kwargs):
    if ctxs is None:
        ctxs = get_empty_df(min(len(samples), max_n))
    for i, l in enumerate(['input', 'target']):
        ctxs = [b.show(ctx=c, label=l, **kwargs) for b, c, _ in zip(samples.itemgot(i), ctxs, range(max_n))]
    ctxs = [b.show(ctx=c, label='pred', **kwargs) for b, c, _ in zip(outs.itemgot(0), ctxs, range(max_n))]
    display_df(pd.DataFrame(ctxs))
    return ctxs


# export
@typedispatch
def show_results(x: TensorText, y, samples, outs, ctxs=None, max_n=10, trunc_at=150, **kwargs):
    if ctxs is None:
        ctxs = get_empty_df(min(len(samples), max_n))
    samples = L((s[0].truncate(trunc_at), *s[1:]) for s in samples)
    ctxs = show_results[object](x, y, samples, outs, ctxs=ctxs, max_n=max_n, **kwargs)
    display_df(pd.DataFrame(ctxs))
    return ctxs


# export
@typedispatch
def plot_top_losses(x: TensorText, y: TensorCategory, samples, outs, raws, losses, trunc_at=150, **kwargs):
    rows = get_empty_df(len(samples))
    samples = L((s[0].truncate(trunc_at), *s[1:]) for s in samples)
    for i, l in enumerate(['input', 'target']):
        rows = [b.show(ctx=c, label=l, **kwargs) for b, c in zip(samples.itemgot(i), rows)]
    outs = L(o + (TitledFloat(r.max().item()), TitledFloat(l.item())) for o, r, l in zip(outs, raws, losses))
    for i, l in enumerate(['predicted', 'probability', 'loss']):
        rows = [b.show(ctx=c, label=l, **kwargs) for b, c in zip(outs.itemgot(i), rows)]
    display_df(pd.DataFrame(rows))


# ## Export -

# hide
notebook2script()
