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
from fastai.data.load import *
from fastai.torch_basics import *
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# +
# default_exp data.core
# -

# export

# hide


# # Data core
#
# > Core functionality for gathering data

# The classes here provide functionality for applying a list of transforms to a set of items (`TfmdLists`, `Datasets`) or a `DataLoader` (`TfmdDl`) as well as the base class used to gather the data for model training: `DataLoaders`.

# ## TfmdDL -

# export
@typedispatch
def show_batch(x, y, samples, ctxs=None, max_n=9, **kwargs):
    if ctxs is None:
        ctxs = Inf.nones
    if hasattr(samples[0], 'show'):
        ctxs = [s.show(ctx=c, **kwargs) for s, c, _ in zip(samples, ctxs, range(max_n))]
    else:
        for i in range_of(samples[0]):
            ctxs = [b.show(ctx=c, **kwargs) for b, c, _ in zip(samples.itemgot(i), ctxs, range(max_n))]
    return ctxs


# `show_batch` is a type-dispatched function that is responsible for showing decoded `samples`. `x` and `y` are the input and the target in the batch to be shown, and are passed along to dispatch on their types. There is a different implementation of `show_batch` if `x` is a `TensorImage` or a `TensorText` for instance (see vision.core or text.data for more details). `ctxs` can be passed but the function is responsible to create them if necessary. `kwargs` depend on the specific implementation.

# export
@typedispatch
def show_results(x, y, samples, outs, ctxs=None, max_n=9, **kwargs):
    if ctxs is None:
        ctxs = Inf.nones
    for i in range(len(samples[0])):
        ctxs = [b.show(ctx=c, **kwargs) for b, c, _ in zip(samples.itemgot(i), ctxs, range(max_n))]
    for i in range(len(outs[0])):
        ctxs = [b.show(ctx=c, **kwargs) for b, c, _ in zip(outs.itemgot(i), ctxs, range(max_n))]
    return ctxs


# `show_results` is a type-dispatched function that is responsible for showing decoded `samples` and their corresponding `outs`. Like in `show_batch`, `x` and `y` are the input and the target in the batch to be shown, and are passed along to dispatch on their types. `ctxs` can be passed but the function is responsible to create them if necessary. `kwargs` depend on the specific implementation.

# export
_all_ = ["show_batch", "show_results"]

# export
_batch_tfms = ('after_item', 'before_batch', 'after_batch')


# export
@log_args(but_as=DataLoader.__init__)
@delegates()
class TfmdDL(DataLoader):
    "Transformed `DataLoader`"

    def __init__(self, dataset, bs=64, shuffle=False, num_workers=None, verbose=False, do_setup=True, **kwargs):
        if num_workers is None:
            num_workers = min(16, defaults.cpus)
        for nm in _batch_tfms:
            kwargs[nm] = Pipeline(kwargs.get(nm, None))
        super().__init__(dataset, bs=bs, shuffle=shuffle, num_workers=num_workers, **kwargs)
        if do_setup:
            for nm in _batch_tfms:
                pv(f"Setting up {nm}: {kwargs[nm]}", verbose)
                kwargs[nm].setup(self)

    def _one_pass(self):
        b = self.do_batch([self.do_item(0)])
        if self.device is not None:
            b = to_device(b, self.device)
        its = self.after_batch(b)
        self._n_inp = 1 if not isinstance(its, (list, tuple)) or len(its) == 1 else len(its) - 1
        self._types = explode_types(its)

    def _retain_dl(self, b):
        if not getattr(self, '_types', None):
            self._one_pass()
        return retain_types(b, typs=self._types)

    @delegates(DataLoader.new)
    def new(self, dataset=None, cls=None, **kwargs):
        res = super().new(dataset, cls, do_setup=False, **kwargs)
        if not hasattr(self, '_n_inp') or not hasattr(self, '_types'):
            try:
                self._one_pass()
                res._n_inp, res._types = self._n_inp, self._types
            except:
                print("Could not do one pass in your dataloader, there is something wrong in it")
        else:
            res._n_inp, res._types = self._n_inp, self._types
        return res

    def before_iter(self):
        super().before_iter()
        split_idx = getattr(self.dataset, 'split_idx', None)
        for nm in _batch_tfms:
            f = getattr(self, nm)
            if isinstance(f, Pipeline):
                f.split_idx = split_idx

    def decode(self, b): return self.before_batch.decode(to_cpu(self.after_batch.decode(self._retain_dl(b))))
    def decode_batch(self, b, max_n=9, full=True): return self._decode_batch(self.decode(b), max_n, full)

    def _decode_batch(self, b, max_n=9, full=True):
        f = self.after_item.decode
        f = compose(f, partial(getattr(self.dataset, 'decode', noop), full=full))
        return L(batch_to_samples(b, max_n=max_n)).map(f)

    def _pre_show_batch(self, b, max_n=9):
        "Decode `b` to be ready for `show_batch`"
        b = self.decode(b)
        if hasattr(b, 'show'):
            return b, None, None
        its = self._decode_batch(b, max_n, full=False)
        if not is_listy(b):
            b, its = [b], L((o,) for o in its)
        return detuplify(b[:self.n_inp]), detuplify(b[self.n_inp:]), its

    def show_batch(self, b=None, max_n=9, ctxs=None, show=True, unique=False, **kwargs):
        if unique:
            old_get_idxs = self.get_idxs
            self.get_idxs = lambda: Inf.zeros
        if b is None:
            b = self.one_batch()
        if not show:
            return self._pre_show_batch(b, max_n=max_n)
        show_batch(*self._pre_show_batch(b, max_n=max_n), ctxs=ctxs, max_n=max_n, **kwargs)
        if unique:
            self.get_idxs = old_get_idxs

    def show_results(self, b, out, max_n=9, ctxs=None, show=True, **kwargs):
        x, y, its = self.show_batch(b, max_n=max_n, show=False)
        b_out = type(b)(b[:self.n_inp] + (tuple(out) if is_listy(out) else (out,)))
        x1, y1, outs = self.show_batch(b_out, max_n=max_n, show=False)
        res = (x, x1, None, None) if its is None else (x, y, its, outs.itemgot(slice(self.n_inp, None)))
        if not show:
            return res
        show_results(*res, ctxs=ctxs, max_n=max_n, **kwargs)

    @property
    def n_inp(self):
        if hasattr(self.dataset, 'n_inp'):
            return self.dataset.n_inp
        if not hasattr(self, '_n_inp'):
            self._one_pass()
        return self._n_inp

    def to(self, device):
        self.device = device
        for tfm in self.after_batch.fs:
            for a in L(getattr(tfm, 'parameters', None)):
                setattr(tfm, a, getattr(tfm, a).to(device))
        return self


# A `TfmdDL` is a `DataLoader` that creates `Pipeline` from a list of `Transform`s for the callbacks `after_item`, `before_batch` and `after_batch`. As a result, it can decode or show a processed `batch`.

# export
add_docs(TfmdDL,
         decode="Decode `b` using `tfms`",
         decode_batch="Decode `b` entirely",
         new="Create a new version of self with a few changed attributes",
         show_batch="Show `b` (defaults to `one_batch`), a list of lists of pipeline outputs (i.e. output of a `DataLoader`)",
         show_results="Show each item of `b` and `out`",
         before_iter="override",
         to="Put self and its transforms state on `device`")


class _Category(int, ShowTitle):
    pass


# +
# Test retain type
class NegTfm(Transform):
    def encodes(self, x): return torch.neg(x)
    def decodes(self, x): return torch.neg(x)


tdl = TfmdDL([(TensorImage([1]),)] * 4, after_batch=NegTfm(), bs=4, num_workers=4)
b = tdl.one_batch()
test_eq(type(b[0]), TensorImage)
b = (tensor([1., 1., 1., 1.]),)
test_eq(type(tdl.decode_batch(b)[0][0]), TensorImage)


# +
class A(Transform):
    def encodes(self, x): return x
    def decodes(self, x): return TitledInt(x)


@Transform
def f(x) -> None: return fastuple((x, x))


start = torch.arange(50)
test_eq_type(f(2), fastuple((2, 2)))

# +
a = A()
tdl = TfmdDL(start, after_item=lambda x: (a(x), f(x)), bs=4)
x, y = tdl.one_batch()
test_eq(type(y), fastuple)

s = tdl.decode_batch((x, y))
test_eq(type(s[0][1]), fastuple)
# -

tdl = TfmdDL(torch.arange(0, 50), after_item=A(), after_batch=NegTfm(), bs=4)
test_eq(tdl.dataset[0], start[0])
test_eq(len(tdl), (50 - 1) // 4 + 1)
test_eq(tdl.bs, 4)
test_stdout(tdl.show_batch, '0\n1\n2\n3')
test_stdout(partial(tdl.show_batch, unique=True), '0\n0\n0\n0')


# +
class B(Transform):
    parameters = 'a'
    def __init__(self): self.a = torch.tensor(0.)
    def encodes(self, x): x


tdl = TfmdDL([(TensorImage([1]),)] * 4, after_batch=B(), bs=4)
test_eq(tdl.after_batch.fs[0].a.device, torch.device('cpu'))
tdl.to(default_device())
test_eq(tdl.after_batch.fs[0].a.device, default_device())
# -

# ### Methods

show_doc(TfmdDL.one_batch)

tfm = NegTfm()
tdl = TfmdDL(start, after_batch=tfm, bs=4)

b = tdl.one_batch()
test_eq(tensor([0, -1, -2, -3]), b)

show_doc(TfmdDL.decode)

test_eq(tdl.decode(b), tensor(0, 1, 2, 3))

show_doc(TfmdDL.decode_batch)

test_eq(tdl.decode_batch(b), [0, 1, 2, 3])

show_doc(TfmdDL.show_batch)

show_doc(TfmdDL.to)


# ## DataLoaders -

# export
@docs
class DataLoaders(GetAttr):
    "Basic wrapper around several `DataLoader`s."
    _default = 'train'

    def __init__(self, *loaders, path='.', device=None):
        self.loaders, self.path = list(loaders), Path(path)
        if device is not None or hasattr(loaders[0], 'to'):
            self.device = device

    def __getitem__(self, i): return self.loaders[i]

    def new_empty(self):
        loaders = [dl.new(dl.dataset.new_empty()) for dl in self.loaders]
        return type(self)(*loaders, path=self.path, device=self.device)

    def _set(i, self, v): self.loaders[i] = v
    train, valid = add_props(lambda i, x: x[i], _set)
    train_ds, valid_ds = add_props(lambda i, x: x[i].dataset)

    @property
    def device(self): return self._device

    @device.setter
    def device(self, d):
        for dl in self.loaders:
            dl.to(d)
        self._device = d

    def to(self, device):
        self.device = device
        return self

    def cuda(self): return self.to(device=default_device())
    def cpu(self): return self.to(device=torch.device('cpu'))

    @classmethod
    def from_dsets(cls, *ds, path='.', bs=64, device=None, dl_type=TfmdDL, **kwargs):
        default = (True,) + (False,) * (len(ds) - 1)
        defaults = {'shuffle': default, 'drop_last': default}
        for nm in _batch_tfms:
            if nm in kwargs:
                kwargs[nm] = Pipeline(kwargs[nm])
        kwargs = merge(defaults, {k: tuplify(v, match=ds) for k, v in kwargs.items()})
        kwargs = [{k: v[i] for k, v in kwargs.items()} for i in range_of(ds)]
        return cls(*[dl_type(d, bs=bs, **k) for d, k in zip(ds, kwargs)], path=path, device=device)

    @classmethod
    def from_dblock(cls, dblock, source, path='.', bs=64, val_bs=None, shuffle_train=True, device=None, **kwargs):
        return dblock.dataloaders(source, path=path, bs=bs, val_bs=val_bs, shuffle_train=shuffle_train, device=device, **kwargs)

    _docs = dict(__getitem__="Retrieve `DataLoader` at `i` (`0` is training, `1` is validation)",
                 train="Training `DataLoader`",
                 valid="Validation `DataLoader`",
                 train_ds="Training `Dataset`",
                 valid_ds="Validation `Dataset`",
                 to="Use `device`",
                 cuda="Use the gpu if available",
                 cpu="Use the cpu",
                 new_empty="Create a new empty version of `self` with the same transforms",
                 from_dblock="Create a dataloaders from a given `dblock`")


dls = DataLoaders(tdl, tdl)
x = dls.train.one_batch()
x2 = first(tdl)
test_eq(x, x2)
x2 = dls.one_batch()
test_eq(x, x2)

# hide
# test assignment works
dls.train = dls.train.new(bs=4)

# ### Methods

show_doc(DataLoaders.__getitem__)

x2 = dls[0].one_batch()
test_eq(x, x2)

show_doc(DataLoaders.train, name="DataLoaders.train")

show_doc(DataLoaders.valid, name="DataLoaders.valid")

show_doc(DataLoaders.train_ds, name="DataLoaders.train_ds")

show_doc(DataLoaders.valid_ds, name="DataLoaders.valid_ds")


# ## TfmdLists -

# +
# export
class FilteredBase:
    "Base class for lists with subsets"
    _dl_type, _dbunch_type = TfmdDL, DataLoaders

    def __init__(self, *args, dl_type=None, **kwargs):
        if dl_type is not None:
            self._dl_type = dl_type
        self.dataloaders = delegates(self._dl_type.__init__)(self.dataloaders)
        super().__init__(*args, **kwargs)

    @property
    def n_subsets(self): return len(self.splits)
    def _new(self, items, **kwargs): return super()._new(items, splits=self.splits, **kwargs)
    def subset(self): raise NotImplemented

    def dataloaders(self, bs=64, val_bs=None, shuffle_train=True, n=None, path='.', dl_type=None, dl_kwargs=None,
                    device=None, **kwargs):
        if device is None:
            device = default_device()
        if dl_kwargs is None:
            dl_kwargs = [{}] * self.n_subsets
        if dl_type is None:
            dl_type = self._dl_type
        drop_last = kwargs.pop('drop_last', shuffle_train)
        dl = dl_type(self.subset(0), bs=bs, shuffle=shuffle_train, drop_last=drop_last, n=n, device=device,
                     **merge(kwargs, dl_kwargs[0]))
        dls = [dl] + [dl.new(self.subset(i), bs=(bs if val_bs is None else val_bs), shuffle=False, drop_last=False,
                             n=None, **dl_kwargs[i]) for i in range(1, self.n_subsets)]
        return self._dbunch_type(*dls, path=path, device=device)


FilteredBase.train, FilteredBase.valid = add_props(lambda i, x: x.subset(i))


# -

# export
class TfmdLists(FilteredBase, L, GetAttr):
    "A `Pipeline` of `tfms` applied to a collection of `items`"
    _default = 'tfms'

    def __init__(self, items, tfms, use_list=None, do_setup=True, split_idx=None, train_setup=True,
                 splits=None, types=None, verbose=False, dl_type=None):
        super().__init__(items, use_list=use_list)
        if dl_type is not None:
            self._dl_type = dl_type
        self.splits = L([slice(None), []] if splits is None else splits).map(mask2idxs)
        if isinstance(tfms, TfmdLists):
            tfms = tfms.tfms
        if isinstance(tfms, Pipeline):
            do_setup = False
        self.tfms = Pipeline(tfms, split_idx=split_idx)
        store_attr('types,split_idx')
        if do_setup:
            pv(f"Setting up {self.tfms}", verbose)
            self.setup(train_setup=train_setup)

    def _new(self, items, split_idx=None, **kwargs):
        split_idx = ifnone(split_idx, self.split_idx)
        return super()._new(items, tfms=self.tfms, do_setup=False, types=self.types, split_idx=split_idx, **kwargs)

    def subset(self, i): return self._new(self._get(self.splits[i]), split_idx=i)
    def _after_item(self, o): return self.tfms(o)
    def __repr__(self): return f"{self.__class__.__name__}: {self.items}\ntfms - {self.tfms.fs}"
    def __iter__(self): return (self[i] for i in range(len(self)))
    def show(self, o, **kwargs): return self.tfms.show(o, **kwargs)
    def decode(self, o, **kwargs): return self.tfms.decode(o, **kwargs)
    def __call__(self, o, **kwargs): return self.tfms.__call__(o, **kwargs)
    def overlapping_splits(self): return L(Counter(self.splits.concat()).values()).filter(gt(1))
    def new_empty(self): return self._new([])

    def setup(self, train_setup=True):
        self.tfms.setup(self, train_setup)
        if len(self) != 0:
            x = super().__getitem__(0) if self.splits is None else super().__getitem__(self.splits[0])[0]
            self.types = []
            for f in self.tfms.fs:
                self.types.append(getattr(f, 'input_types', type(x)))
                x = f(x)
            self.types.append(type(x))
        types = L(t if is_listy(t) else [t] for t in self.types).concat().unique()
        self.pretty_types = '\n'.join([f'  - {t}' for t in types])

    def infer_idx(self, x):
        # TODO: check if we really need this, or can simplify
        idx = 0
        for t in self.types:
            if isinstance(x, t):
                break
            idx += 1
        types = L(t if is_listy(t) else [t] for t in self.types).concat().unique()
        pretty_types = '\n'.join([f'  - {t}' for t in types])
        assert idx < len(self.types), f"Expected an input of type in \n{pretty_types}\n but got {type(x)}"
        return idx

    def infer(self, x):
        return compose_tfms(x, tfms=self.tfms.fs[self.infer_idx(x):], split_idx=self.split_idx)

    def __getitem__(self, idx):
        res = super().__getitem__(idx)
        if self._after_item is None:
            return res
        return self._after_item(res) if is_indexer(idx) else res.map(self._after_item)


# export
add_docs(TfmdLists,
         setup="Transform setup with self",
         decode="From `Pipeline`",
         show="From `Pipeline`",
         overlapping_splits="All splits that are in more than one split",
         subset="New `TfmdLists` with same tfms that only includes items in `i`th split",
         infer_idx="Finds the index where `self.tfms` can be applied to `x`, depending on the type of `x`",
         infer="Apply `self.tfms` to `x` starting at the right tfm depending on the type of `x`",
         new_empty="A new version of `self` but with no items")


# exports
def decode_at(o, idx):
    "Decoded item at `idx`"
    return o.decode(o[idx])


# exports
def show_at(o, idx, **kwargs):
    "Show item at `idx`",
    return o.show(o[idx], **kwargs)


# A `TfmdLists` combines a collection of object with a `Pipeline`. `tfms` can either be a `Pipeline` or a list of transforms, in which case, it will wrap them in a `Pipeline`. `use_list` is passed along to `L` with the `items` and `split_idx` are passed to each transform of the `Pipeline`. `do_setup` indicates if the `Pipeline.setup` method should be called during initialization.

# +
class _IntFloatTfm(Transform):
    def encodes(self, o): return TitledInt(o)
    def decodes(self, o): return TitledFloat(o)


int2f_tfm = _IntFloatTfm()


def _neg(o): return -o


neg_tfm = Transform(_neg, _neg)
# -

items = L([1., 2., 3.])
tfms = [neg_tfm, int2f_tfm]
tl = TfmdLists(items, tfms=tfms)
test_eq_type(tl[0], TitledInt(-1))
test_eq_type(tl[1], TitledInt(-2))
test_eq_type(tl.decode(tl[2]), TitledFloat(3.))
test_stdout(lambda: show_at(tl, 2), '-3')
test_eq(tl.types, [float, float, TitledInt])
tl

# add splits to TfmdLists
splits = [[0, 2], [1]]
tl = TfmdLists(items, tfms=tfms, splits=splits)
test_eq(tl.n_subsets, 2)
test_eq(tl.train, tl.subset(0))
test_eq(tl.valid, tl.subset(1))
test_eq(tl.train.items, items[splits[0]])
test_eq(tl.valid.items, items[splits[1]])
test_eq(tl.train.tfms.split_idx, 0)
test_eq(tl.valid.tfms.split_idx, 1)
test_eq(tl.train.new_empty().split_idx, 0)
test_eq(tl.valid.new_empty().split_idx, 1)
test_eq_type(tl.splits, L(splits))
assert not tl.overlapping_splits()

df = pd.DataFrame(dict(a=[1, 2, 3], b=[2, 3, 4]))
tl = TfmdLists(df, lambda o: o.a + 1, splits=[[0], [1, 2]])
test_eq(tl[1, 2], [3, 4])
tr = tl.subset(0)
test_eq(tr[:], [2])
val = tl.subset(1)
test_eq(val[:], [3, 4])


# +
class _B(Transform):
    def __init__(self): self.m = 0
    def encodes(self, o): return o + self.m
    def decodes(self, o): return o - self.m

    def setups(self, items):
        print(items)
        self.m = tensor(items).float().mean().item()


# test for setup, which updates `self.m`
tl = TfmdLists(items, _B())
test_eq(tl.m, 2)


# -

# Here's how we can use `TfmdLists.setup` to implement a simple category list, getting labels from a mock file list:

# +
class _Cat(Transform):
    order = 1
    def encodes(self, o): return int(self.o2i[o])
    def decodes(self, o): return TitledStr(self.vocab[o])
    def setups(self, items): self.vocab, self.o2i = uniqueify(L(items), sort=True, bidir=True)


tcat = _Cat()


def _lbl(o): return TitledStr(o.split('_')[0])


# Check that tfms are sorted by `order` & `_lbl` is called first
fns = ['dog_0.jpg', 'cat_0.jpg', 'cat_2.jpg', 'cat_1.jpg', 'dog_1.jpg']
tl = TfmdLists(fns, [tcat, _lbl])
exp_voc = ['cat', 'dog']
test_eq(tcat.vocab, exp_voc)
test_eq(tl.tfms.vocab, exp_voc)
test_eq(tl.vocab, exp_voc)
test_eq(tl, (1, 0, 0, 0, 1))
test_eq([tl.decode(o) for o in tl], ('dog', 'cat', 'cat', 'cat', 'dog'))
# -

# Check only the training set is taken into account for setup
tl = TfmdLists(fns, [tcat, _lbl], splits=[[0, 4], [1, 2, 3]])
test_eq(tcat.vocab, ['dog'])

tfm = NegTfm(split_idx=1)
tds = TfmdLists(start, A())
tdl = TfmdDL(tds, after_batch=tfm, bs=4)
x = tdl.one_batch()
test_eq(x, torch.arange(4))
tds.split_idx = 1
x = tdl.one_batch()
test_eq(x, -torch.arange(4))
tds.split_idx = 0
x = tdl.one_batch()
test_eq(x, torch.arange(4))

tds = TfmdLists(start, A())
tdl = TfmdDL(tds, after_batch=NegTfm(), bs=4)
test_eq(tdl.dataset[0], start[0])
test_eq(len(tdl), (len(tds) - 1) // 4 + 1)
test_eq(tdl.bs, 4)
test_stdout(tdl.show_batch, '0\n1\n2\n3')

show_doc(TfmdLists.subset)

show_doc(TfmdLists.infer_idx)

show_doc(TfmdLists.infer)


# +
def mult(x): return x * 2


mult.order = 2

fns = ['dog_0.jpg', 'cat_0.jpg', 'cat_2.jpg', 'cat_1.jpg', 'dog_1.jpg']
tl = TfmdLists(fns, [_lbl, _Cat(), mult])

test_eq(tl.infer_idx('dog_45.jpg'), 0)
test_eq(tl.infer('dog_45.jpg'), 2)

test_eq(tl.infer_idx(4), 2)
test_eq(tl.infer(4), 8)

test_fail(lambda: tl.infer_idx(2.0))
test_fail(lambda: tl.infer(2.0))

# +
# hide
# Test input_types works on a Transform
cat = _Cat()
cat.input_types = (str, float)
tl = TfmdLists(fns, [_lbl, cat, mult])
test_eq(tl.infer_idx(2.0), 1)

# Test type annotations work on a function


def mult(x: (int, float)): return x * 2


mult.order = 2
tl = TfmdLists(fns, [_lbl, _Cat(), mult])
test_eq(tl.infer_idx(2.0), 2)


# -

# ## Datasets -

# export
@docs
@delegates(TfmdLists)
class Datasets(FilteredBase):
    "A dataset that creates a tuple from each `tfms`, passed through `item_tfms`"

    def __init__(self, items=None, tfms=None, tls=None, n_inp=None, dl_type=None, **kwargs):
        super().__init__(dl_type=dl_type)
        self.tls = L(tls if tls else [TfmdLists(items, t, **kwargs) for t in L(ifnone(tfms, [None]))])
        self.n_inp = ifnone(n_inp, max(1, len(self.tls) - 1))

    def __getitem__(self, it):
        res = tuple([tl[it] for tl in self.tls])
        return res if is_indexer(it) else list(zip(*res))

    def __getattr__(self, k): return gather_attrs(self, k, 'tls')
    def __dir__(self): return super().__dir__() + gather_attr_names(self, 'tls')
    def __len__(self): return len(self.tls[0])
    def __iter__(self): return (self[i] for i in range(len(self)))
    def __repr__(self): return coll_repr(self)
    def decode(self, o, full=True): return tuple(tl.decode(o_, full=full) for o_, tl in zip(o, tuplify(self.tls, match=o)))
    def subset(self, i): return type(self)(tls=L(tl.subset(i) for tl in self.tls), n_inp=self.n_inp)
    def _new(self, items, *args, **kwargs): return super()._new(items, tfms=self.tfms, do_setup=False, **kwargs)
    def overlapping_splits(self): return self.tls[0].overlapping_splits()
    def new_empty(self): return type(self)(tls=[tl.new_empty() for tl in self.tls], n_inp=self.n_inp)
    @property
    def splits(self): return self.tls[0].splits
    @property
    def split_idx(self): return self.tls[0].tfms.split_idx
    @property
    def items(self): return self.tls[0].items

    @items.setter
    def items(self, v):
        for tl in self.tls:
            tl.items = v

    def show(self, o, ctx=None, **kwargs):
        for o_, tl in zip(o, self.tls):
            ctx = tl.show(o_, ctx=ctx, **kwargs)
        return ctx

    @contextmanager
    def set_split_idx(self, i):
        old_split_idx = self.split_idx
        for tl in self.tls:
            tl.tfms.split_idx = i
        try:
            yield self
        finally:
            for tl in self.tls:
                tl.tfms.split_idx = old_split_idx

    _docs = dict(
        decode="Compose `decode` of all `tuple_tfms` then all `tfms` on `i`",
        show="Show item `o` in `ctx`",
        dataloaders="Get a `DataLoaders`",
        overlapping_splits="All splits that are in more than one split",
        subset="New `Datasets` that only includes subset `i`",
        new_empty="Create a new empty version of the `self`, keeping only the transforms",
        set_split_idx="Contextmanager to use the same `Datasets` with another `split_idx`"
    )


# A `Datasets` creates a tuple from `items` (typically input,target) by applying to them each list of `Transform` (or `Pipeline`) in `tfms`. Note that if `tfms` contains only one list of `tfms`, the items given by `Datasets` will be tuples of one element.
#
# `n_inp` is the number of elements in the tuples that should be considered part of the input and will default to 1 if `tfms` consists of one set of transforms, `len(tfms)-1` otherwise. In most cases, the number of elements in the tuples spit out by `Datasets` will be 2 (for input,target) but it can happen that there is 3 (Siamese networks or tabular data) in which case we need to be able to determine when the inputs end and the targets begin.

items = [1, 2, 3, 4]
dsets = Datasets(items, [[neg_tfm, int2f_tfm], [add(1)]])
t = dsets[0]
test_eq(t, (-1, 2))
test_eq(dsets[0, 1, 2], [(-1, 2), (-2, 3), (-3, 4)])
test_eq(dsets.n_inp, 1)
dsets.decode(t)


class Norm(Transform):
    def encodes(self, o): return (o - self.m) / self.s
    def decodes(self, o): return (o * self.s) + self.m

    def setups(self, items):
        its = tensor(items).float()
        self.m, self.s = its.mean(), its.std()


# +
items = [1, 2, 3, 4]
nrm = Norm()
dsets = Datasets(items, [[neg_tfm, int2f_tfm], [neg_tfm, nrm]])

x, y = zip(*dsets)
test_close(tensor(y).mean(), 0)
test_close(tensor(y).std(), 1)
test_eq(x, (-1, -2, -3, -4,))
test_eq(nrm.m, -2.5)
test_stdout(lambda: show_at(dsets, 1), '-2')

test_eq(dsets.m, nrm.m)
test_eq(dsets.norm.m, nrm.m)
test_eq(dsets.train.norm.m, nrm.m)


# +
# hide
# Check filtering is properly applied
class B(Transform):
    def encodes(self, x) -> None: return int(x + 1)
    def decodes(self, x): return TitledInt(x - 1)


add1 = B(split_idx=1)

dsets = Datasets(items, [neg_tfm, [neg_tfm, int2f_tfm, add1]], splits=[[3], [0, 1, 2]])
test_eq(dsets[1], [-2, -2])
test_eq(dsets.valid[1], [-2, -1])
test_eq(dsets.valid[[1, 1]], [[-2, -1], [-2, -1]])
test_eq(dsets.train[0], [-4, -4])
# -

test_fns = ['dog_0.jpg', 'cat_0.jpg', 'cat_2.jpg', 'cat_1.jpg', 'kid_1.jpg']
tcat = _Cat()
dsets = Datasets(test_fns, [[tcat, _lbl]], splits=[[0, 1, 2], [3, 4]])
test_eq(tcat.vocab, ['cat', 'dog'])
test_eq(dsets.train, [(1,), (0,), (0,)])
test_eq(dsets.valid[0], (0,))
test_stdout(lambda: show_at(dsets.train, 0), "dog")

# +
inp = [0, 1, 2, 3, 4]
dsets = Datasets(inp, tfms=[None])

test_eq(*dsets[2], 2)          # Retrieve one item (subset 0 is the default)
test_eq(dsets[1, 2], [(1,), (2,)])    # Retrieve two items by index
mask = [True, False, False, True, False]
test_eq(dsets[mask], [(0,), (3,)])   # Retrieve two items by mask
# -

inp = pd.DataFrame(dict(a=[5, 1, 2, 3, 4]))
dsets = Datasets(inp, tfms=attrgetter('a')).subset(0)
test_eq(*dsets[2], 2)          # Retrieve one item (subset 0 is the default)
test_eq(dsets[1, 2], [(1,), (2,)])    # Retrieve two items by index
mask = [True, False, False, True, False]
test_eq(dsets[mask], [(5,), (3,)])   # Retrieve two items by mask

# test n_inp
inp = [0, 1, 2, 3, 4]
dsets = Datasets(inp, tfms=[None])
test_eq(dsets.n_inp, 1)
dsets = Datasets(inp, tfms=[[None], [None], [None]])
test_eq(dsets.n_inp, 2)
dsets = Datasets(inp, tfms=[[None], [None], [None]], n_inp=1)
test_eq(dsets.n_inp, 1)

# +
# splits can be indices
dsets = Datasets(range(5), tfms=[None], splits=[tensor([0, 2]), [1, 3, 4]])

test_eq(dsets.subset(0), [(0,), (2,)])
test_eq(dsets.train, [(0,), (2,)])       # Subset 0 is aliased to `train`
test_eq(dsets.subset(1), [(1,), (3,), (4,)])
test_eq(dsets.valid, [(1,), (3,), (4,)])     # Subset 1 is aliased to `valid`
test_eq(*dsets.valid[2], 4)
#assert '[(1,),(3,),(4,)]' in str(dsets) and '[(0,),(2,)]' in str(dsets)
dsets

# +
# splits can be boolean masks (they don't have to cover all items, but must be disjoint)
splits = [[False, True, True, False, True], [True, False, False, False, False]]
dsets = Datasets(range(5), tfms=[None], splits=splits)

test_eq(dsets.train, [(1,), (2,), (4,)])
test_eq(dsets.valid, [(0,)])
# -

# apply transforms to all items
tfm = [[lambda x: x * 2, lambda x: x + 1]]
splits = [[1, 2], [0, 3, 4]]
dsets = Datasets(range(5), tfm, splits=splits)
test_eq(dsets.train, [(3,), (5,)])
test_eq(dsets.valid, [(1,), (7,), (9,)])
test_eq(dsets.train[False, True], [(5,)])


# only transform subset 1
class _Tfm(Transform):
    split_idx = 1
    def encodes(self, x): return x * 2
    def decodes(self, x): return TitledStr(x // 2)


dsets = Datasets(range(5), [_Tfm()], splits=[[1, 2], [0, 3, 4]])
test_eq(dsets.train, [(1,), (2,)])
test_eq(dsets.valid, [(0,), (6,), (8,)])
test_eq(dsets.train[False, True], [(2,)])
dsets

# A context manager to change the split_idx and apply the validation transform on the training set
ds = dsets.train
with ds.set_split_idx(1):
    test_eq(ds, [(2,), (4,)])
test_eq(dsets.train, [(1,), (2,)])

# hide
# Test Datasets pickles
dsrc1 = pickle.loads(pickle.dumps(dsets))
test_eq(dsets.train, dsrc1.train)
test_eq(dsets.valid, dsrc1.valid)

dsets = Datasets(range(5), [_Tfm(), noop], splits=[[1, 2], [0, 3, 4]])
test_eq(dsets.train, [(1, 1), (2, 2)])
test_eq(dsets.valid, [(0, 0), (6, 3), (8, 4)])

start = torch.arange(0, 50)
tds = Datasets(start, [A()])
tdl = TfmdDL(tds, after_item=NegTfm(), bs=4)
b = tdl.one_batch()
test_eq(tdl.decode_batch(b), ((0,), (1,), (2,), (3,)))
test_stdout(tdl.show_batch, "0\n1\n2\n3")


# +
# only transform subset 1
class _Tfm(Transform):
    split_idx = 1
    def encodes(self, x): return x * 2


dsets = Datasets(range(8), [None], splits=[[1, 2, 5, 7], [0, 3, 4, 6]])


# +
# only transform subset 1
class _Tfm(Transform):
    split_idx = 1
    def encodes(self, x): return x * 2


dsets = Datasets(range(8), [None], splits=[[1, 2, 5, 7], [0, 3, 4, 6]])
dls = dsets.dataloaders(bs=4, after_batch=_Tfm(), shuffle_train=False, device=torch.device('cpu'))
test_eq(dls.train, [(tensor([1, 2, 5, 7]),)])
test_eq(dls.valid, [(tensor([0, 6, 8, 12]),)])
test_eq(dls.n_inp, 1)
# -

# ### Methods

items = [1, 2, 3, 4]
dsets = Datasets(items, [[neg_tfm, int2f_tfm]])

# hide_input
_dsrc = Datasets([1, 2])
show_doc(_dsrc.dataloaders, name="Datasets.dataloaders")

show_doc(Datasets.decode)

test_eq(*dsets[0], -1)
test_eq(*dsets.decode((-1,)), 1)

show_doc(Datasets.show)

test_stdout(lambda: dsets.show(dsets[1]), '-2')

show_doc(Datasets.new_empty)

items = [1, 2, 3, 4]
nrm = Norm()
dsets = Datasets(items, [[neg_tfm, int2f_tfm], [neg_tfm]])
empty = dsets.new_empty()
test_eq(empty.items, [])

# hide
# test it works for dataframes too
df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [6, 7, 8, 9, 10]})
dsets = Datasets(df, [[attrgetter('a')], [attrgetter('b')]])
empty = dsets.new_empty()


# ## Add test set for inference

# +
# only transform subset 1
class _Tfm1(Transform):
    split_idx = 0
    def encodes(self, x): return x * 3


dsets = Datasets(range(8), [[_Tfm(), _Tfm1()]], splits=[[1, 2, 5, 7], [0, 3, 4, 6]])
test_eq(dsets.train, [(3,), (6,), (15,), (21,)])
test_eq(dsets.valid, [(0,), (6,), (8,), (12,)])


# -

# export
def test_set(dsets, test_items, rm_tfms=None, with_labels=False):
    "Create a test set from `test_items` using validation transforms of `dsets`"
    if isinstance(dsets, Datasets):
        tls = dsets.tls if with_labels else dsets.tls[:dsets.n_inp]
        test_tls = [tl._new(test_items, split_idx=1) for tl in tls]
        if rm_tfms is None:
            rm_tfms = [tl.infer_idx(get_first(test_items)) for tl in test_tls]
        else:
            rm_tfms = tuplify(rm_tfms, match=test_tls)
        for i, j in enumerate(rm_tfms):
            test_tls[i].tfms.fs = test_tls[i].tfms.fs[j:]
        return Datasets(tls=test_tls)
    elif isinstance(dsets, TfmdLists):
        test_tl = dsets._new(test_items, split_idx=1)
        if rm_tfms is None:
            rm_tfms = dsets.infer_idx(get_first(test_items))
        test_tl.tfms.fs = test_tl.tfms.fs[rm_tfms:]
        return test_tl
    else:
        raise Exception(f"This method requires using the fastai library to assemble your data. Expected a `Datasets` or a `TfmdLists` but got {dsets.__class__.__name__}")


# +
class _Tfm1(Transform):
    split_idx = 0
    def encodes(self, x): return x * 3


dsets = Datasets(range(8), [[_Tfm(), _Tfm1()]], splits=[[1, 2, 5, 7], [0, 3, 4, 6]])
test_eq(dsets.train, [(3,), (6,), (15,), (21,)])
test_eq(dsets.valid, [(0,), (6,), (8,), (12,)])

# Tranform of the validation set are applied
tst = test_set(dsets, [1, 2, 3])
test_eq(tst, [(2,), (4,), (6,)])

# +
# hide
# Test with different types
tfm = _Tfm1()
tfm.split_idx, tfm.order = None, 2
dsets = Datasets(['dog', 'cat', 'cat', 'dog'], [[_Cat(), tfm]])

# With strings
test_eq(test_set(dsets, ['dog', 'cat', 'cat']), [(3,), (0,), (0,)])
# With ints
test_eq(test_set(dsets, [1, 2]), [(3,), (6,)])

# +
# hide
# Test with various input lengths
dsets = Datasets(range(8), [[_Tfm(), _Tfm1()], [_Tfm(), _Tfm1()], [_Tfm(), _Tfm1()]], splits=[[1, 2, 5, 7], [0, 3, 4, 6]])
tst = test_set(dsets, [1, 2, 3])
test_eq(tst, [(2, 2), (4, 4), (6, 6)])

dsets = Datasets(range(8), [[_Tfm(), _Tfm1()], [_Tfm(), _Tfm1()], [_Tfm(), _Tfm1()]], splits=[[1, 2, 5, 7], [0, 3, 4, 6]], n_inp=1)
tst = test_set(dsets, [1, 2, 3])
test_eq(tst, [(2,), (4,), (6,)])

# +
# hide
# Test with rm_tfms
dsets = Datasets(range(8), [[_Tfm(), _Tfm()]], splits=[[1, 2, 5, 7], [0, 3, 4, 6]])
tst = test_set(dsets, [1, 2, 3])
test_eq(tst, [(4,), (8,), (12,)])

dsets = Datasets(range(8), [[_Tfm(), _Tfm()]], splits=[[1, 2, 5, 7], [0, 3, 4, 6]])
tst = test_set(dsets, [1, 2, 3], rm_tfms=1)
test_eq(tst, [(2,), (4,), (6,)])

dsets = Datasets(range(8), [[_Tfm(), _Tfm()], [_Tfm(), _Tfm()]], splits=[[1, 2, 5, 7], [0, 3, 4, 6]], n_inp=2)
tst = test_set(dsets, [1, 2, 3], rm_tfms=(1, 0))
test_eq(tst, [(2, 4), (4, 8), (6, 12)])


# -

# export
@delegates(TfmdDL.__init__)
@patch
def test_dl(self: DataLoaders, test_items, rm_type_tfms=None, with_labels=False, **kwargs):
    "Create a test dataloader from `test_items` using validation transforms of `dls`"
    test_ds = test_set(self.valid_ds, test_items, rm_tfms=rm_type_tfms, with_labels=with_labels
                       ) if isinstance(self.valid_ds, (Datasets, TfmdLists)) else test_items
    return self.valid.new(test_ds, **kwargs)


dsets = Datasets(range(8), [[_Tfm(), _Tfm1()]], splits=[[1, 2, 5, 7], [0, 3, 4, 6]])
dls = dsets.dataloaders(bs=4, device=torch.device('cpu'))

dsets = Datasets(range(8), [[_Tfm(), _Tfm1()]], splits=[[1, 2, 5, 7], [0, 3, 4, 6]])
dls = dsets.dataloaders(bs=4, device=torch.device('cpu'))
tst_dl = dls.test_dl([2, 3, 4, 5])
test_eq(tst_dl._n_inp, 1)
test_eq(list(tst_dl), [(tensor([4, 6, 8, 10]),)])
# Test you can change transforms
tst_dl = dls.test_dl([2, 3, 4, 5], after_item=add1)
test_eq(list(tst_dl), [(tensor([5, 7, 9, 11]),)])

# ## Export -

# hide
notebook2script()
