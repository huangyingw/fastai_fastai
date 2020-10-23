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
from inspect import isfunction, ismethod
from nbdev.export import notebook2script
from fastai.data.load import _collate_types
from fastai.vision.data import *
from fastai.vision.core import *
from nbdev.showdoc import *
from fastai.data.transforms import *
from fastai.data.external import *
from fastai.data.load import *
from fastai.data.core import *
from fastai.torch_basics import *
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# +
# default_exp data.block
# -

# export

# hide


# # Data block
#
# > High level API to quickly get your data in a `DataLoaders`

# ## TransformBlock -

# export
class TransformBlock():
    "A basic wrapper that links defaults transforms for the data block API"

    def __init__(self, type_tfms=None, item_tfms=None, batch_tfms=None, dl_type=None, dls_kwargs=None):
        self.type_tfms = L(type_tfms)
        self.item_tfms = ToTensor + L(item_tfms)
        self.batch_tfms = L(batch_tfms)
        self.dl_type, self.dls_kwargs = dl_type, ({} if dls_kwargs is None else dls_kwargs)


# export
def CategoryBlock(vocab=None, sort=True, add_na=False):
    "`TransformBlock` for single-label categorical targets"
    return TransformBlock(type_tfms=Categorize(vocab=vocab, sort=sort, add_na=add_na))


# export
def MultiCategoryBlock(encoded=False, vocab=None, add_na=False):
    "`TransformBlock` for multi-label categorical targets"
    tfm = EncodedMultiCategorize(vocab=vocab) if encoded else [MultiCategorize(vocab=vocab, add_na=add_na), OneHotEncode]
    return TransformBlock(type_tfms=tfm)


# export
def RegressionBlock(n_out=None):
    "`TransformBlock` for float targets"
    return TransformBlock(type_tfms=RegressionSetup(c=n_out))


# ## General API

# export


# export
def _merge_grouper(o):
    if isinstance(o, LambdaType):
        return id(o)
    elif isinstance(o, type):
        return o
    elif (isfunction(o) or ismethod(o)):
        return o.__qualname__
    return o.__class__


# +
# export
def _merge_tfms(*tfms):
    "Group the `tfms` in a single list, removing duplicates (from the same class) and instantiating"
    g = groupby(concat(*tfms), _merge_grouper)
    return L(v[-1] for k, v in g.items()).map(instantiate)


def _zip(x): return L(x).zip()


# -

# For example, so not exported

# +
# hide
tfms = _merge_tfms([Categorize, MultiCategorize, Categorize(['dog', 'cat'])], Categorize(['a', 'b']))
# If there are several instantiated versions, the last one is kept.
test_eq(len(tfms), 2)
test_eq(tfms[1].__class__, MultiCategorize)
test_eq(tfms[0].__class__, Categorize)
test_eq(tfms[0].vocab, ['a', 'b'])

tfms = _merge_tfms([PILImage.create, PILImage.show])
# Check methods are properly separated
test_eq(len(tfms), 2)
tfms = _merge_tfms([show_image, set_trace])
# Check functions are properly separated
test_eq(len(tfms), 2)


def _f(x): return 0


test_eq(len(_merge_tfms([_f, lambda x: 1])), 2)
test_eq(len(_merge_tfms([_f, _f])), 1)


# -

# export
@docs
@funcs_kwargs
class DataBlock():
    "Generic container to quickly build `Datasets` and `DataLoaders`"
    get_x = get_items = splitter = get_y = None
    blocks, dl_type = (TransformBlock, TransformBlock), TfmdDL
    _methods = 'get_items splitter get_y get_x'.split()
    _msg = "If you wanted to compose several transforms in your getter don't forget to wrap them in a `Pipeline`."

    def __init__(self, blocks=None, dl_type=None, getters=None, n_inp=None, item_tfms=None, batch_tfms=None, **kwargs):
        blocks = L(self.blocks if blocks is None else blocks)
        blocks = L(b() if callable(b) else b for b in blocks)
        self.type_tfms = blocks.attrgot('type_tfms', L())
        self.default_item_tfms = _merge_tfms(*blocks.attrgot('item_tfms', L()))
        self.default_batch_tfms = _merge_tfms(*blocks.attrgot('batch_tfms', L()))
        for b in blocks:
            if getattr(b, 'dl_type', None) is not None:
                self.dl_type = b.dl_type
        if dl_type is not None:
            self.dl_type = dl_type
        self.dataloaders = delegates(self.dl_type.__init__)(self.dataloaders)
        self.dls_kwargs = merge(*blocks.attrgot('dls_kwargs', {}))

        self.n_inp = ifnone(n_inp, max(1, len(blocks) - 1))
        self.getters = ifnone(getters, [noop] * len(self.type_tfms))
        if self.get_x:
            if len(L(self.get_x)) != self.n_inp:
                raise ValueError(f'get_x contains {len(L(self.get_x))} functions, but must contain {self.n_inp} (one for each input)\n{self._msg}')
            self.getters[:self.n_inp] = L(self.get_x)
        if self.get_y:
            n_targs = len(self.getters) - self.n_inp
            if len(L(self.get_y)) != n_targs:
                raise ValueError(f'get_y contains {len(L(self.get_y))} functions, but must contain {n_targs} (one for each target)\n{self._msg}')
            self.getters[self.n_inp:] = L(self.get_y)

        if kwargs:
            raise TypeError(f'invalid keyword arguments: {", ".join(kwargs.keys())}')
        self.new(item_tfms, batch_tfms)

    def _combine_type_tfms(self): return L([self.getters, self.type_tfms]).map_zip(
        lambda g, tt: (g.fs if isinstance(g, Pipeline) else L(g)) + tt)

    def new(self, item_tfms=None, batch_tfms=None):
        self.item_tfms = _merge_tfms(self.default_item_tfms, item_tfms)
        self.batch_tfms = _merge_tfms(self.default_batch_tfms, batch_tfms)
        return self

    @classmethod
    def from_columns(cls, blocks=None, getters=None, get_items=None, **kwargs):
        if getters is None:
            getters = L(ItemGetter(i) for i in range(2 if blocks is None else len(L(blocks))))
        get_items = _zip if get_items is None else compose(get_items, _zip)
        return cls(blocks=blocks, getters=getters, get_items=get_items, **kwargs)

    def datasets(self, source, verbose=False):
        self.source = source
        pv(f"Collecting items from {source}", verbose)
        items = (self.get_items or noop)(source)
        pv(f"Found {len(items)} items", verbose)
        splits = (self.splitter or RandomSplitter())(items)
        pv(f"{len(splits)} datasets of sizes {','.join([str(len(s)) for s in splits])}", verbose)
        return Datasets(items, tfms=self._combine_type_tfms(), splits=splits, dl_type=self.dl_type, n_inp=self.n_inp, verbose=verbose)

    def dataloaders(self, source, path='.', verbose=False, **kwargs):
        dsets = self.datasets(source)
        kwargs = {**self.dls_kwargs, **kwargs, 'verbose': verbose}
        return dsets.dataloaders(path=path, after_item=self.item_tfms, after_batch=self.batch_tfms, **kwargs)

    _docs = dict(new="Create a new `DataBlock` with other `item_tfms` and `batch_tfms`",
                 datasets="Create a `Datasets` object from `source`",
                 dataloaders="Create a `DataLoaders` object from `source`")


# To build a `DataBlock` you need to give the library four things: the types of your input/labels, and at least two functions: `get_items` and `splitter`. You may also need to include `get_x` and `get_y` or a more generic list of `getters` that are applied to the results of `get_items`.
#
# Once those are provided, you automatically get a `Datasets` or a `DataLoaders`:

show_doc(DataBlock.datasets)

# hide_input
dblock = DataBlock()
show_doc(dblock.dataloaders, name="DataBlock.dataloaders")

# You can create a `DataBlock` by passing functions:

mnist = DataBlock(blocks=(ImageBlock(cls=PILImageBW), CategoryBlock),
                  get_items=get_image_files,
                  splitter=GrandparentSplitter(),
                  get_y=parent_label)

# Each type comes with default transforms that will be applied
# - at the base level to create items in a tuple (usually input,target) from the base elements (like filenames)
# - at the item level of the datasets
# - at the batch level
#
# They are called respectively type transforms, item transforms, batch transforms. In the case of MNIST, the type transforms are the method to create a `PILImageBW` (for the input) and the `Categorize` transform (for the target), the item transform is `ToTensor` and the batch transforms are `Cuda` and `IntToFloatTensor`. You can add any other transforms by passing them in `DataBlock.datasets` or `DataBlock.dataloaders`.

test_eq(mnist.type_tfms[0], [PILImageBW.create])
test_eq(mnist.type_tfms[1].map(type), [Categorize])
test_eq(mnist.default_item_tfms.map(type), [ToTensor])
test_eq(mnist.default_batch_tfms.map(type), [IntToFloatTensor])

dsets = mnist.datasets(untar_data(URLs.MNIST_TINY))
test_eq(dsets.vocab, ['3', '7'])
x, y = dsets.train[0]
test_eq(x.size, (28, 28))
show_at(dsets.train, 0, cmap='Greys', figsize=(2, 2))

test_fail(lambda: DataBlock(wrong_kwarg=42, wrong_kwarg2='foo'))

# We can pass any number of blocks to `DataBlock`, we can then define what are the input and target blocks by changing `n_inp`. For example, defining `n_inp=2` will consider the first two blocks passed as inputs and the others as targets.

mnist = DataBlock((ImageBlock, ImageBlock, CategoryBlock), get_items=get_image_files, splitter=GrandparentSplitter(),
                  get_y=parent_label)
dsets = mnist.datasets(untar_data(URLs.MNIST_TINY))
test_eq(mnist.n_inp, 2)
test_eq(len(dsets.train[0]), 3)

test_fail(lambda: DataBlock((ImageBlock, ImageBlock, CategoryBlock), get_items=get_image_files, splitter=GrandparentSplitter(),
                            get_y=[parent_label, noop],
                            n_inp=2), msg='get_y contains 2 functions, but must contain 1 (one for each output)')

mnist = DataBlock((ImageBlock, ImageBlock, CategoryBlock), get_items=get_image_files, splitter=GrandparentSplitter(),
                  n_inp=1,
                  get_y=[noop, Pipeline([noop, parent_label])])
dsets = mnist.datasets(untar_data(URLs.MNIST_TINY))
test_eq(len(dsets.train[0]), 3)


# ## Debugging

# export
def _short_repr(x):
    if isinstance(x, tuple):
        return f'({", ".join([_short_repr(y) for y in x])})'
    if isinstance(x, list):
        return f'[{", ".join([_short_repr(y) for y in x])}]'
    if not isinstance(x, Tensor):
        return str(x)
    if x.numel() <= 20 and x.ndim <= 1:
        return str(x)
    return f'{x.__class__.__name__} of size {"x".join([str(d) for d in x.shape])}'


# hide
test_eq(_short_repr(TensorImage(torch.randn(40, 56))), 'TensorImage of size 40x56')
test_eq(_short_repr(TensorCategory([1, 2, 3])), 'TensorCategory([1, 2, 3])')
test_eq(_short_repr((TensorImage(torch.randn(40, 56)), TensorImage(torch.randn(32, 20)))),
        '(TensorImage of size 40x56, TensorImage of size 32x20)')


# export
def _apply_pipeline(p, x):
    print(f"  {p}\n    starting from\n      {_short_repr(x)}")
    for f in p.fs:
        name = f.name
        try:
            x = f(x)
            if name != "noop":
                print(f"    applying {name} gives\n      {_short_repr(x)}")
        except Exception as e:
            print(f"    applying {name} failed.")
            raise e
    return x


# +
# export


def _find_fail_collate(s):
    s = L(*s)
    for x in s[0]:
        if not isinstance(x, _collate_types):
            return f"{type(x).__name__} is not collatable"
    for i in range_of(s[0]):
        try:
            _ = default_collate(s.itemgot(i))
        except:
            shapes = [getattr(o[i], 'shape', None) for o in s]
            return f"Could not collate the {i}-th members of your tuples because got the following shapes\n{','.join([str(s) for s in shapes])}"


# -

# export
@patch
def summary(self: DataBlock, source, bs=4, show_batch=False, **kwargs):
    "Steps through the transform pipeline for one batch, and optionally calls `show_batch(**kwargs)` on the transient `Dataloaders`."
    print(f"Setting-up type transforms pipelines")
    dsets = self.datasets(source, verbose=True)
    print("\nBuilding one sample")
    for tl in dsets.train.tls:
        _apply_pipeline(tl.tfms, get_first(dsets.train.items))
    print(f"\nFinal sample: {dsets.train[0]}\n\n")

    dls = self.dataloaders(source, bs=bs, verbose=True)
    print("\nBuilding one batch")
    if len([f for f in dls.train.after_item.fs if f.name != 'noop']) != 0:
        print("Applying item_tfms to the first sample:")
        s = [_apply_pipeline(dls.train.after_item, dsets.train[0])]
        print(f"\nAdding the next {bs-1} samples")
        s += [dls.train.after_item(dsets.train[i]) for i in range(1, bs)]
    else:
        print("No item_tfms to apply")
        s = [dls.train.after_item(dsets.train[i]) for i in range(bs)]

    if len([f for f in dls.train.before_batch.fs if f.name != 'noop']) != 0:
        print("\nApplying before_batch to the list of samples")
        s = _apply_pipeline(dls.train.before_batch, s)
    else:
        print("\nNo before_batch transform to apply")

    print("\nCollating items in a batch")
    try:
        b = dls.train.create_batch(s)
        b = retain_types(b, s[0] if is_listy(s) else s)
    except Exception as e:
        print("Error! It's not possible to collate your items in a batch")
        why = _find_fail_collate(s)
        print("Make sure all parts of your samples are tensors of the same size" if why is None else why)
        raise e

    if len([f for f in dls.train.after_batch.fs if f.name != 'noop']) != 0:
        print("\nApplying batch_tfms to the batch built")
        b = to_device(b, dls.device)
        b = _apply_pipeline(dls.train.after_batch, b)
    else:
        print("\nNo batch_tfms to apply")

    if show_batch:
        dls.show_batch(**kwargs)


show_doc(DataBlock.summary)

# Besides stepping through the transformation, `summary()`  provides a shortcut `dls.show_batch(...)`, to see the data.  E.g.
#
# ```
# pets.summary(path/"images", bs=8, show_batch=True, unique=True,...)
# ```
#
# is a shortcut to:
# ```
# pets.summary(path/"images", bs=8)
# dls = pets.dataloaders(path/"images", bs=8)
# dls.show_batch(unique=True,...)  # See different tfms effect on the same image.
# ```

# ## Export -

# hide
notebook2script()
