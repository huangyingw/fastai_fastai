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
from fastai.text.core import *
from fastai.data.all import *
from fastai.torch_basics import *
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# export

# hide


# +
# default_exp text.data
# default_cls_lvl 3
# -

# # Text data
#
# > Functions and transforms to help gather text data in a `Datasets`

# ## Backwards
#
# Reversing the text can provide higher accuracy with an ensemble with a forward model. All that is needed is a `type_tfm` that will reverse the text as it is brought in:

# export
def reverse_text(x): return x.flip(0)


t = tensor([0, 1, 2])
r = reverse_text(t)
test_eq(r, tensor([2, 1, 0]))


# ## Numericalizing

# Numericalization is the step in which we convert tokens to integers. The first step is to build a correspondence token to index that is called a vocab.

# export
def make_vocab(count, min_freq=3, max_vocab=60000, special_toks=None):
    "Create a vocab of `max_vocab` size from `Counter` `count` with items present more than `min_freq`"
    vocab = [o for o, c in count.most_common(max_vocab) if c >= min_freq]
    special_toks = ifnone(special_toks, defaults.text_spec_tok)
    for o in reversed(special_toks):  # Make sure all special tokens are in the vocab
        if o in vocab:
            vocab.remove(o)
        vocab.insert(0, o)
    vocab = vocab[:max_vocab]
    return vocab + [f'xxfake' for i in range(0, 8 - len(vocab) % 8)]


# If there are more than `max_vocab` tokens, the ones kept are the most frequent.
#
# > Note: For performance when using mixed precision, the vocabulary is always made of size a multiple of 8, potentially by adding `xxfake` tokens.

count = Counter(['a', 'a', 'a', 'a', 'b', 'b', 'c', 'c', 'd'])
test_eq(set([x for x in make_vocab(count) if not x.startswith('xxfake')]),
        set(defaults.text_spec_tok + 'a'.split()))
test_eq(len(make_vocab(count)) % 8, 0)
test_eq(set([x for x in make_vocab(count, min_freq=1) if not x.startswith('xxfake')]),
        set(defaults.text_spec_tok + 'a b c d'.split()))
test_eq(set([x for x in make_vocab(count, max_vocab=12, min_freq=1) if not x.startswith('xxfake')]),
        set(defaults.text_spec_tok + 'a b c'.split()))


# +
# export
class TensorText(TensorBase):
    pass
class LMTensorText(TensorText):
    pass

TensorText.__doc__ = "Semantic type for a tensor representing text"
LMTensorText.__doc__ = "Semantic type for a tensor representing text in language modeling"


# -

# export
class Numericalize(Transform):
    "Reversible transform of tokenized texts to numericalized ids"
    def __init__(self, vocab=None, min_freq=3, max_vocab=60000, special_toks=None, pad_tok=None):
        store_attr('vocab,min_freq,max_vocab,special_toks,pad_tok')
        self.o2i = None if vocab is None else defaultdict(int, {v: k for k, v in enumerate(vocab)})

    def setups(self, dsets):
        if dsets is None:
            return
        if self.vocab is None:
            count = dsets.counter if getattr(dsets, 'counter', None) is not None else Counter(p for o in dsets for p in o)
            if self.special_toks is None and hasattr(dsets, 'special_toks'):
                self.special_toks = dsets.special_toks
            self.vocab = make_vocab(count, min_freq=self.min_freq, max_vocab=self.max_vocab, special_toks=self.special_toks)
            self.o2i = defaultdict(int, {v: k for k, v in enumerate(self.vocab) if v != 'xxfake'})

    def encodes(self, o): return TensorText(tensor([self.o2i[o_] for o_ in o]))
    def decodes(self, o): return L(self.vocab[o_] for o_ in o if self.vocab[o_] != self.pad_tok)


# If no `vocab` is passed, one is created at setup from the data, using `make_vocab` with `min_freq` and `max_vocab`.

# +
start = 'This is an example of text'
num = Numericalize(min_freq=1)
num.setup(L(start.split(), 'this is another text'.split()))
test_eq(set([x for x in num.vocab if not x.startswith('xxfake')]),
        set(defaults.text_spec_tok + 'This is an example of text this another'.split()))
test_eq(len(num.vocab) % 8, 0)
t = num(start.split())

test_eq(t, tensor([11, 9, 12, 13, 14, 10]))
test_eq(num.decode(t), start.split())
# -

num = Numericalize(min_freq=2)
num.setup(L('This is an example of text'.split(), 'this is another text'.split()))
test_eq(set([x for x in num.vocab if not x.startswith('xxfake')]),
        set(defaults.text_spec_tok + 'is text'.split()))
test_eq(len(num.vocab) % 8, 0)
t = num(start.split())
test_eq(t, tensor([0, 9, 0, 0, 0, 10]))
test_eq(num.decode(t), f'{UNK} is {UNK} {UNK} {UNK} text'.split())

# hide
df = pd.DataFrame({'texts': ['This is an example of text', 'this is another text']})
tl = TfmdLists(df, [attrgetter('text'), Tokenizer.from_df('texts'), Numericalize(min_freq=2)])
test_eq(tl, [tensor([2, 8, 9, 10, 0, 0, 0, 11]), tensor([2, 9, 10, 0, 11])])


# ## LM_DataLoader -

# export
def _maybe_first(o): return o[0] if isinstance(o, tuple) else o


# export
def _get_tokenizer(ds):
    tok = getattr(ds, 'tokenizer', None)
    if isinstance(tok, Tokenizer):
        return tok
    if isinstance(tok, (list, L)):
        for t in tok:
            if isinstance(t, Tokenizer):
                return t


# export
def _get_lengths(ds):
    tok = _get_tokenizer(ds)
    if tok is None:
        return
    return tok.get_lengths(ds.items)


# export
# TODO: add backward
@log_args(but_as=TfmdDL.__init__)
@delegates()
class LMDataLoader(TfmdDL):
    "A `DataLoader` suitable for language modeling"
    def __init__(self, dataset, lens=None, cache=2, bs=64, seq_len=72, num_workers=0, **kwargs):
        self.items = ReindexCollection(dataset, cache=cache, tfm=_maybe_first)
        self.seq_len = seq_len
        if lens is None:
            lens = _get_lengths(dataset)
        if lens is None:
            lens = [len(o) for o in self.items]
        self.lens = ReindexCollection(lens, idxs=self.items.idxs)
        # The "-1" is to allow for final label, we throw away the end that's less than bs
        corpus = round_multiple(sum(lens) - 1, bs, round_down=True)
        self.bl = corpus // bs  # bl stands for batch length
        self.n_batches = self.bl // (seq_len) + int(self.bl % seq_len != 0)
        self.last_len = self.bl - (self.n_batches - 1) * seq_len
        self.make_chunks()
        super().__init__(dataset=dataset, bs=bs, num_workers=num_workers, **kwargs)
        self.n = self.n_batches * bs

    def make_chunks(self): self.chunks = Chunks(self.items, self.lens)
    def shuffle_fn(self, idxs):
        self.items.shuffle()
        self.make_chunks()
        return idxs

    def create_item(self, seq):
        if seq >= self.n:
            raise IndexError
        sl = self.last_len if seq // self.bs == self.n_batches - 1 else self.seq_len
        st = (seq % self.bs) * self.bl + (seq // self.bs) * self.seq_len
        txt = self.chunks[st: st + sl + 1]
        return LMTensorText(txt[:-1]), txt[1:]

    @delegates(TfmdDL.new)
    def new(self, dataset=None, seq_len=None, **kwargs):
        lens = self.lens.coll if dataset is None else None
        seq_len = self.seq_len if seq_len is None else seq_len
        return super().new(dataset=dataset, lens=lens, seq_len=seq_len, **kwargs)


show_doc(LMDataLoader, title_level=2)

# `dataset` should be a collection of numericalized texts for this to work. `lens` can be passed for optimizing the creation, otherwise, the `LMDataLoader` will do a full pass of the `dataset` to compute them. `cache` is used to avoid reloading items unnecessarily.
#
# The `LMDataLoader` will concatenate all texts (maybe `shuffle`d) in one big stream, split it in `bs` contiguous sentences, then go through those `seq_len` at a time.

# hide
bs, sl = 4, 3
ints = L([0, 1, 2, 3, 4], [5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18], [19, 20], [21, 22]).map(tensor)
dl = LMDataLoader(ints, bs=bs, seq_len=sl)
list(dl)
test_eq(list(dl),
        [[tensor([[0, 1, 2], [5, 6, 7], [10, 11, 12], [15, 16, 17]]),
          tensor([[1, 2, 3], [6, 7, 8], [11, 12, 13], [16, 17, 18]])],
         [tensor([[3, 4], [8, 9], [13, 14], [18, 19]]),
            tensor([[4, 5], [9, 10], [14, 15], [19, 20]])]])

bs, sl = 4, 3
ints = L([0, 1, 2, 3, 4], [5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18], [19, 20], [21, 22, 23], [24]).map(tensor)

dl = LMDataLoader(ints, bs=bs, seq_len=sl)
test_eq(list(dl),
        [[tensor([[0, 1, 2], [6, 7, 8], [12, 13, 14], [18, 19, 20]]),
          tensor([[1, 2, 3], [7, 8, 9], [13, 14, 15], [19, 20, 21]])],
         [tensor([[3, 4, 5], [9, 10, 11], [15, 16, 17], [21, 22, 23]]),
            tensor([[4, 5, 6], [10, 11, 12], [16, 17, 18], [22, 23, 24]])]])

# hide
# Check lens work
dl = LMDataLoader(ints, lens=ints.map(len), bs=bs, seq_len=sl)
test_eq(list(dl),
        [[tensor([[0, 1, 2], [6, 7, 8], [12, 13, 14], [18, 19, 20]]),
          tensor([[1, 2, 3], [7, 8, 9], [13, 14, 15], [19, 20, 21]])],
         [tensor([[3, 4, 5], [9, 10, 11], [15, 16, 17], [21, 22, 23]]),
            tensor([[4, 5, 6], [10, 11, 12], [16, 17, 18], [22, 23, 24]])]])

dl = LMDataLoader(ints, bs=bs, seq_len=sl, shuffle=True)
for x, y in dl:
    test_eq(x[:, 1:], y[:, :-1])
((x0, y0), (x1, y1)) = tuple(dl)
# Second batch begins where first batch ended
test_eq(y0[:, -1], x1[:, 0])
test_eq(type(x0), LMTensorText)

# hide
# test new works
dl = LMDataLoader(ints, bs=bs, seq_len=sl, shuffle=True)
dl1 = dl.new()
test_eq(dl1.seq_len, sl)
dl2 = dl.new(seq_len=2)
test_eq(dl2.seq_len, 2)


# ### Showing -

# export
@typedispatch
def show_batch(x: TensorText, y, samples, ctxs=None, max_n=10, trunc_at=150, **kwargs):
    if ctxs is None:
        ctxs = get_empty_df(min(len(samples), max_n))
    if trunc_at is not None:
        samples = L((s[0].truncate(trunc_at), *s[1:]) for s in samples)
    ctxs = show_batch[object](x, y, samples, max_n=max_n, ctxs=ctxs, **kwargs)
    display_df(pd.DataFrame(ctxs))
    return ctxs


# export
@typedispatch
def show_batch(x: LMTensorText, y, samples, ctxs=None, max_n=10, trunc_at=150, **kwargs):
    samples = L((s[0].truncate(trunc_at), s[1].truncate(trunc_at)) for s in samples)
    return show_batch[TensorText](x, None, samples, ctxs=ctxs, max_n=max_n, trunc_at=None, **kwargs)


# ## Classification

# For classification, we deal with the fact that texts don't all have the same length by using padding.

# export
def pad_input(samples, pad_idx=1, pad_fields=0, pad_first=False, backwards=False):
    "Function that collect `samples` and adds padding"
    pad_fields = L(pad_fields)
    max_len_l = pad_fields.map(lambda f: max([len(s[f]) for s in samples]))
    if backwards:
        pad_first = not pad_first
    def _f(field_idx, x):
        if field_idx not in pad_fields:
            return x
        idx = pad_fields.items.index(field_idx)  # TODO: remove items if L.index is fixed
        sl = slice(-len(x), sys.maxsize) if pad_first else slice(0, len(x))
        pad = x.new_zeros(max_len_l[idx] - x.shape[0]) + pad_idx
        x1 = torch.cat([pad, x] if pad_first else [x, pad])
        if backwards:
            x1 = x1.flip(0)
        return retain_type(x1, x)
    return [tuple(map(lambda idxx: _f(*idxx), enumerate(s))) for s in samples]


# `pad_idx` is used for the padding, and the padding is applied to the `pad_fields` of the samples. The padding is applied at the beginning if `pad_first` is `True`, and if `backwards` is added, the tensors are flipped.

test_eq(pad_input([(tensor([1, 2, 3]), 1), (tensor([4, 5]), 2), (tensor([6]), 3)], pad_idx=0),
        [(tensor([1, 2, 3]), 1), (tensor([4, 5, 0]), 2), (tensor([6, 0, 0]), 3)])
test_eq(pad_input([(tensor([1, 2, 3]), (tensor([6]))), (tensor([4, 5]), tensor([4, 5])), (tensor([6]), (tensor([1, 2, 3])))], pad_idx=0, pad_fields=1),
        [(tensor([1, 2, 3]), (tensor([6, 0, 0]))), (tensor([4, 5]), tensor([4, 5, 0])), ((tensor([6]), tensor([1, 2, 3])))])
test_eq(pad_input([(tensor([1, 2, 3]), 1), (tensor([4, 5]), 2), (tensor([6]), 3)], pad_idx=0, pad_first=True),
        [(tensor([1, 2, 3]), 1), (tensor([0, 4, 5]), 2), (tensor([0, 0, 6]), 3)])
test_eq(pad_input([(tensor([1, 2, 3]), 1), (tensor([4, 5]), 2), (tensor([6]), 3)], pad_idx=0, backwards=True),
        [(tensor([3, 2, 1]), 1), (tensor([5, 4, 0]), 2), (tensor([6, 0, 0]), 3)])
x = test_eq(pad_input([(tensor([1, 2, 3]), 1), (tensor([4, 5]), 2), (tensor([6]), 3)], pad_idx=0, backwards=True),
            [(tensor([3, 2, 1]), 1), (tensor([5, 4, 0]), 2), (tensor([6, 0, 0]), 3)])

# hide
# Check retain type
x = [(TensorText([1, 2, 3]), 1), (TensorText([4, 5]), 2), (TensorText([6]), 3)]
y = pad_input(x, pad_idx=0)
for s in y:
    test_eq(type(s[0]), TensorText)


# export
def pad_input_chunk(samples, pad_idx=1, pad_first=True, seq_len=72):
    "Pad `samples` by adding padding by chunks of size `seq_len`"
    max_len = max([len(s[0]) for s in samples])
    def _f(x):
        l = max_len - x.shape[0]
        pad_chunk = x.new_zeros((l // seq_len) * seq_len) + pad_idx
        pad_res = x.new_zeros(l % seq_len) + pad_idx
        x1 = torch.cat([pad_chunk, x, pad_res]) if pad_first else torch.cat([x, pad_res, pad_chunk])
        return retain_type(x1, x)
    return [(_f(s[0]), *s[1:]) for s in samples]


# The difference with the base `pad_input` is that most of the padding is applied first (if `pad_first=True`) or at the end (if `pad_first=False`) but only by a round multiple of `seq_len`. The rest of the padding is applied to the end (or the beginning if `pad_first=False`). This is to work with `SequenceEncoder` with recurrent models.

test_eq(pad_input_chunk([(tensor([1, 2, 3, 4, 5, 6]), 1), (tensor([1, 2, 3]), 2), (tensor([1, 2]), 3)], pad_idx=0, seq_len=2),
        [(tensor([1, 2, 3, 4, 5, 6]), 1), (tensor([0, 0, 1, 2, 3, 0]), 2), (tensor([0, 0, 0, 0, 1, 2]), 3)])
test_eq(pad_input_chunk([(tensor([1, 2, 3, 4, 5, 6]),), (tensor([1, 2, 3]),), (tensor([1, 2]),)], pad_idx=0, seq_len=2),
        [(tensor([1, 2, 3, 4, 5, 6]),), (tensor([0, 0, 1, 2, 3, 0]),), (tensor([0, 0, 0, 0, 1, 2]),)])
test_eq(pad_input_chunk([(tensor([1, 2, 3, 4, 5, 6]),), (tensor([1, 2, 3]),), (tensor([1, 2]),)], pad_idx=0, seq_len=2, pad_first=False),
        [(tensor([1, 2, 3, 4, 5, 6]),), (tensor([1, 2, 3, 0, 0, 0]),), (tensor([1, 2, 0, 0, 0, 0]),)])


# +
# export
def _default_sort(x): return len(x[0])

@delegates(TfmdDL)
class SortedDL(TfmdDL):
    "A `DataLoader` that goes throught the item in the order given by `sort_func`"
    def __init__(self, dataset, sort_func=None, res=None, **kwargs):
        super().__init__(dataset, **kwargs)
        self.sort_func = _default_sort if sort_func is None else sort_func
        if res is None and self.sort_func == _default_sort:
            res = _get_lengths(dataset)
        self.res = [self.sort_func(self.do_item(i)) for i in range_of(self.dataset)] if res is None else res
        if len(self.res) > 0:
            self.idx_max = np.argmax(self.res)

    def get_idxs(self):
        idxs = super().get_idxs()
        if self.shuffle:
            return idxs
        return sorted(idxs, key=lambda i: self.res[i], reverse=True)

    def shuffle_fn(self, idxs):
        idxs = np.random.permutation(len(self.dataset))
        idx_max = np.where(idxs == self.idx_max)[0][0]
        idxs[0], idxs[idx_max] = idxs[idx_max], idxs[0]
        sz = self.bs * 50
        chunks = [idxs[i:i + sz] for i in range(0, len(idxs), sz)]
        chunks = [sorted(s, key=lambda i: self.res[i], reverse=True) for s in chunks]
        sort_idx = np.concatenate(chunks)

        sz = self.bs
        batches = [sort_idx[i:i + sz] for i in range(0, len(sort_idx), sz)]
        sort_idx = np.concatenate(np.random.permutation(batches[1:-1])) if len(batches) > 2 else np.array([], dtype=np.int)
        sort_idx = np.concatenate((batches[0], sort_idx) if len(batches) == 1 else (batches[0], sort_idx, batches[-1]))
        return iter(sort_idx)

    @delegates(TfmdDL.new)
    def new(self, dataset=None, **kwargs):
        if 'val_res' in kwargs and kwargs['val_res'] is not None:
            res = kwargs['val_res']
        else:
            res = self.res if dataset is None else None
        return super().new(dataset=dataset, res=res, **kwargs)


# -

# `res` is the result of `sort_func` applied on all elements of the `dataset`. You can pass it if available to make the init much faster by avoiding an initial pass over the whole dataset. For example if sorting by text length (as in the default `sort_func`, called `_default_sort`) you should pass a list with the length of each element in `dataset` to `res` to take advantage of this speed-up.
#
# To get the same init speed-up for the validation set, `val_res` (a list of text lengths for your validation set) can be passed to the `kwargs` argument of `SortedDL`. Below is an example to reduce the init time by passing a list of text lengths for both the training set and the validation set:
#
# ```
# # Pass the training dataset text lengths to SortedDL
# srtd_dl=partial(SortedDL, res = train_text_lens)
#
# # Pass the validation dataset text lengths
# dl_kwargs = [{},{'val_res': val_text_lens}]
#
# # init our Datasets
# dsets = Datasets(...)
#
# # init our Dataloaders
# dls = dsets.dataloaders(...,dl_type = srtd_dl, dl_kwargs = dl_kwargs)
# ```
#
# If `shuffle` is `True`, this will shuffle a bit the results of the sort to have items of roughly the same size in batches, but not in the exact sorted order.

ds = [(tensor([1, 2]), 1), (tensor([3, 4, 5, 6]), 2), (tensor([7]), 3), (tensor([8, 9, 10]), 4)]
dl = SortedDL(ds, bs=2, before_batch=partial(pad_input, pad_idx=0))
test_eq(list(dl), [(tensor([[3, 4, 5, 6], [8, 9, 10, 0]]), tensor([2, 4])),
                   (tensor([[1, 2], [7, 0]]), tensor([1, 3]))])

ds = [(tensor(range(random.randint(1, 10))), i) for i in range(101)]
dl = SortedDL(ds, bs=2, create_batch=partial(pad_input, pad_idx=-1), shuffle=True, num_workers=0)
batches = list(dl)
max_len = len(batches[0][0])
for b in batches:
    assert(len(b[0])) <= max_len
    test_ne(b[0][-1], -1)


# ## TransformBlock for text

# To use the data block API, you will need this build block for texts.

# export
class TextBlock(TransformBlock):
    "A `TransformBlock` for texts"
    @delegates(Numericalize.__init__)
    def __init__(self, tok_tfm, vocab=None, is_lm=False, seq_len=72, backwards=False, **kwargs):
        type_tfms = [tok_tfm, Numericalize(vocab, **kwargs)]
        if backwards:
            type_tfms += [reverse_text]
        return super().__init__(type_tfms=type_tfms,
                                dl_type=LMDataLoader if is_lm else SortedDL,
                                dls_kwargs={'seq_len': seq_len} if is_lm else {'before_batch': partial(pad_input_chunk, seq_len=seq_len)})

    @classmethod
    @delegates(Tokenizer.from_df, keep=True)
    def from_df(cls, text_cols, vocab=None, is_lm=False, seq_len=72, backwards=False, min_freq=3, max_vocab=60000, **kwargs):
        "Build a `TextBlock` from a dataframe using `text_cols`"
        return cls(Tokenizer.from_df(text_cols, **kwargs), vocab=vocab, is_lm=is_lm, seq_len=seq_len,
                   backwards=backwards, min_freq=min_freq, max_vocab=max_vocab)

    @classmethod
    @delegates(Tokenizer.from_folder, keep=True)
    def from_folder(cls, path, vocab=None, is_lm=False, seq_len=72, backwards=False, min_freq=3, max_vocab=60000, **kwargs):
        "Build a `TextBlock` from a `path`"
        return cls(Tokenizer.from_folder(path, **kwargs), vocab=vocab, is_lm=is_lm, seq_len=seq_len,
                   backwards=backwards, min_freq=min_freq, max_vocab=max_vocab)


# For efficient tokenization, you probably want to use one of the factory methods. Otherwise, you can pass your custom `tok_tfm` that will deal with tokenization (if your texts are already tokenized, you can pass `noop`), a `vocab`, or leave it to be inferred on the texts using `min_freq` and `max_vocab`.
#
# `is_lm` indicates if we want to use texts for language modeling or another task, `seq_len` is only necessary to tune if `is_lm=False`, and is passed along to `pad_input_chunk`.

show_doc(TextBlock.from_df)

# Here is an example using a sample of IMDB stored as a CSV file:

# +
path = untar_data(URLs.IMDB_SAMPLE)
df = pd.read_csv(path / 'texts.csv')

imdb_clas = DataBlock(
    blocks=(TextBlock.from_df('text', seq_len=72), CategoryBlock),
    get_x=ColReader('text'), get_y=ColReader('label'), splitter=ColSplitter())

dls = imdb_clas.dataloaders(df, bs=64)
dls.show_batch(max_n=2)
# -

# `vocab`,  `is_lm`, `seq_len`, `min_freq` and `max_vocab` are passed to the main init, the other argument to `Tokenizer.from_df`.

show_doc(TextBlock.from_folder)


# `vocab`, `is_lm`, `seq_len`, `min_freq` and `max_vocab` are passed to the main init, the other argument to `Tokenizer.from_folder`.

# ## TextDataLoaders -

# +
# export
class TextDataLoaders(DataLoaders):
    "Basic wrapper around several `DataLoader`s with factory methods for NLP problems"
    @classmethod
    @delegates(DataLoaders.from_dblock)
    def from_folder(cls, path, train='train', valid='valid', valid_pct=None, seed=None, vocab=None, text_vocab=None, is_lm=False,
                    tok_tfm=None, seq_len=72, backwards=False, **kwargs):
        "Create from imagenet style dataset in `path` with `train` and `valid` subfolders (or provide `valid_pct`)"
        splitter = GrandparentSplitter(train_name=train, valid_name=valid) if valid_pct is None else RandomSplitter(valid_pct, seed=seed)
        blocks = [TextBlock.from_folder(path, text_vocab, is_lm, seq_len, backwards) if tok_tfm is None else TextBlock(tok_tfm, text_vocab, is_lm, seq_len, backwards)]
        if not is_lm:
            blocks.append(CategoryBlock(vocab=vocab))
        get_items = partial(get_text_files, folders=[train, valid]) if valid_pct is None else get_text_files
        dblock = DataBlock(blocks=blocks,
                           get_items=get_items,
                           splitter=splitter,
                           get_y=None if is_lm else parent_label)
        return cls.from_dblock(dblock, path, path=path, seq_len=seq_len, **kwargs)

    @classmethod
    @delegates(DataLoaders.from_dblock)
    def from_df(cls, df, path='.', valid_pct=0.2, seed=None, text_col=0, label_col=1, label_delim=None, y_block=None,
                text_vocab=None, is_lm=False, valid_col=None, tok_tfm=None, seq_len=72, backwards=False, **kwargs):
        "Create from `df` in `path` with `valid_pct`"
        blocks = [TextBlock.from_df(text_col, text_vocab, is_lm, seq_len, backwards) if tok_tfm is None else TextBlock(tok_tfm, text_vocab, is_lm, seq_len, backwards)]
        if y_block is None and not is_lm:
            blocks.append(MultiCategoryBlock if is_listy(label_col) and len(label_col) > 1 else CategoryBlock)
        if y_block is not None and not is_lm:
            blocks += (y_block if is_listy(y_block) else [y_block])
        splitter = RandomSplitter(valid_pct, seed=seed) if valid_col is None else ColSplitter(valid_col)
        dblock = DataBlock(blocks=blocks,
                           get_x=ColReader("text"),
                           get_y=None if is_lm else ColReader(label_col, label_delim=label_delim),
                           splitter=splitter)
        return cls.from_dblock(dblock, df, path=path, seq_len=seq_len, **kwargs)

    @classmethod
    def from_csv(cls, path, csv_fname='labels.csv', header='infer', delimiter=None, **kwargs):
        "Create from `csv` file in `path/csv_fname`"
        df = pd.read_csv(Path(path) / csv_fname, header=header, delimiter=delimiter)
        return cls.from_df(df, path=path, **kwargs)

TextDataLoaders.from_csv = delegates(to=TextDataLoaders.from_df)(TextDataLoaders.from_csv)
# -

show_doc(TextDataLoaders, title_level=2)

# You should not use the init directly but one of the following factory methods. All those factory methods accept as arguments:
#
# - `text_vocab`: the vocabulary used for numericalizing texts (if not passed, it's inferred from the data)
# - `tok_tfm`: if passed, uses this `tok_tfm` instead of the default
# - `seq_len`: the sequence length used for batch
# - `bs`: the batch size
# - `val_bs`: the batch size for the validation `DataLoader` (defaults to `bs`)
# - `shuffle_train`: if we shuffle the training `DataLoader` or not
# - `device`: the PyTorch device to use (defaults to `default_device()`)

show_doc(TextDataLoaders.from_folder)

# If `valid_pct` is provided, a random split is performed (with an optional `seed`) by setting aside that percentage of the data for the validation set (instead of looking at the grandparents folder). If a `vocab` is passed, only the folders with names in `vocab` are kept.
#
# Here is an example on a sample of the IMDB movie review dataset:

# slow
path = untar_data(URLs.IMDB)
dls = TextDataLoaders.from_folder(path)
dls.show_batch(max_n=3)

show_doc(TextDataLoaders.from_df)

# `seed` can optionally be passed for reproducibility. `text_col`, `label_col` and optionally `valid_col` are indices or names of columns for texts/labels and the validation flag. `label_delim` can be passed for a multi-label problem if your labels are in one column, separated by a particular char. `y_block` should be passed to indicate your type of targets, in case the library did no infer it properly.
#
# Here are examples on subsets of IMDB:

dls = TextDataLoaders.from_df(df, path=path, text_col='text', label_col='label', valid_col='is_valid')
dls.show_batch(max_n=3)

dls = TextDataLoaders.from_df(df, path=path, text_col='text', is_lm=True, valid_col='is_valid')
dls.show_batch(max_n=3)

show_doc(TextDataLoaders.from_csv)

# Opens the csv file with `header` and `delimiter`, then pass all the other arguments to `TextDataLoaders.from_df`.

dls = TextDataLoaders.from_csv(path=path, csv_fname='texts.csv', text_col='text', label_col='label', valid_col='is_valid')
dls.show_batch(max_n=3)

# ## Export -

# hide
notebook2script()
