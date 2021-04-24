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
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter, _SingleProcessDataLoaderIter, _DatasetKind
from subprocess import Popen, PIPE
from nbdev.export import notebook2script
from nbdev.showdoc import *
from fastai.torch_basics import *
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# +
# default_exp data.load
# -

# export
_loaders = (_MultiProcessingDataLoaderIter, _SingleProcessDataLoaderIter)

# hide

bs = 4
letters = list(string.ascii_lowercase)


# ## DataLoader helpers

# fastai includes a replacement for Pytorch's *DataLoader* which is largely API-compatible, and adds a lot of useful functionality and flexibility. Before we look at the class, there are a couple of helpers we'll need to define.

# +
# export
def _wif(worker_id):
    set_num_threads(1)
    info = get_worker_info()
    ds = info.dataset.d
    ds.num_workers, ds.offs = info.num_workers, info.id
    set_seed(info.seed)
    ds.wif()


class _FakeLoader:
    def _fn_noops(self, x=None, *args, **kwargs): return x

    _IterableDataset_len_called, _auto_collation, collate_fn, drop_last = None, False, _fn_noops, False
    _index_sampler, generator, prefetch_factor = Inf.count, None, 2
    dataset_kind = _dataset_kind = _DatasetKind.Iterable

    def __init__(self, d, pin_memory, num_workers, timeout, persistent_workers):
        self.dataset, self.default, self.worker_init_fn = self, d, _wif
        store_attr('d,pin_memory,num_workers,timeout,persistent_workers')

    def __iter__(self): return iter(self.d.create_batches(self.d.sample()))

    @property
    def multiprocessing_context(self): return (None, multiprocessing)[self.num_workers > 0]

    @contextmanager
    def no_multiproc(self):
        old_num_workers = self.num_workers
        try:
            self.num_workers = 0
            yield self.d
        finally:
            self.num_workers = old_num_workers


_collate_types = (ndarray, Tensor, typing.Mapping, str)


# -

# export
def fa_collate(t):
    "A replacement for PyTorch `default_collate` which maintains types and handles `Sequence`s"
    b = t[0]
    return (default_collate(t) if isinstance(b, _collate_types)
            else type(t[0])([fa_collate(s) for s in zip(*t)]) if isinstance(b, Sequence)
            else default_collate(t))


# +
#e.g. x is int, y is tuple
t = [(1, (2, 3)), (1, (2, 3))]
test_eq(fa_collate(t), default_collate(t))
test_eq(L(fa_collate(t)).map(type), [Tensor, tuple])

t = [(1, (2, (3, 4))), (1, (2, (3, 4)))]
test_eq(fa_collate(t), default_collate(t))
test_eq(L(fa_collate(t)).map(type), [Tensor, tuple])
test_eq(L(fa_collate(t)[1]).map(type), [Tensor, tuple])


# -

# export
def fa_convert(t):
    "A replacement for PyTorch `default_convert` which maintains types and handles `Sequence`s"
    return (default_convert(t) if isinstance(t, _collate_types)
            else type(t)([fa_convert(s) for s in t]) if isinstance(t, Sequence)
            else default_convert(t))


# +
t0 = array([1, 2])
t = [t0, (t0, t0)]

test_eq(fa_convert(t), default_convert(t))
test_eq(L(fa_convert(t)).map(type), [Tensor, tuple])


# -

# export
class SkipItemException(Exception):
    "Raised to notify `DataLoader` to skip an item"
    pass


show_doc(SkipItemException, title_level=3)


# ## DataLoader -

# export
@funcs_kwargs
class DataLoader(GetAttr):
    _noop_methods = 'wif before_iter after_item before_batch after_batch after_iter'.split()
    for o in _noop_methods:
        exec(f"def {o}(self, x=None, *args, **kwargs): return x")
    _methods = _noop_methods + 'create_batches create_item create_batch retain \
        get_idxs sample shuffle_fn do_batch create_batch'.split()
    _default = 'dataset'

    def __init__(self, dataset=None, bs=None, num_workers=0, pin_memory=False, timeout=0, batch_size=None,
                 shuffle=False, drop_last=False, indexed=None, n=None, device=None, persistent_workers=False, **kwargs):
        if batch_size is not None:
            bs = batch_size  # PyTorch compatibility
        assert not (bs is None and drop_last)
        if indexed is None:
            indexed = (hasattr(dataset, '__getitem__')
                       and not isinstance(dataset, IterableDataset))
        if not indexed and shuffle:
            raise ValueError("Can only shuffle an indexed dataset (not an iterable one).")
        if n is None:
            try:
                n = len(dataset)
            except TypeError:
                pass
        store_attr('dataset,bs,shuffle,drop_last,indexed,n,pin_memory,timeout,device')
        self.rng, self.num_workers, self.offs = random.Random(random.randint(0, 2**32 - 1)), 1, 0
        if sys.platform == "win32" and IN_NOTEBOOK and num_workers > 0:
            print("Due to IPython and Windows limitation, python multiprocessing isn't available now.")
            print("So `number_workers` is changed to 0 to avoid getting stuck")
            num_workers = 0
        self.fake_l = _FakeLoader(self, pin_memory, num_workers, timeout, persistent_workers=persistent_workers)

    def __len__(self):
        if self.n is None:
            raise TypeError
        if self.bs is None:
            return self.n
        return self.n // self.bs + (0 if self.drop_last or self.n % self.bs == 0 else 1)

    def get_idxs(self):
        idxs = Inf.count if self.indexed else Inf.nones
        if self.n is not None:
            idxs = list(itertools.islice(idxs, self.n))
        if self.shuffle:
            idxs = self.shuffle_fn(idxs)
        return idxs

    def sample(self):
        return (b for i, b in enumerate(self.__idxs) if i // (self.bs or 1) % self.num_workers == self.offs)

    def __iter__(self):
        self.randomize()
        self.before_iter()
        self.__idxs = self.get_idxs()  # called in context of main process (not workers/subprocesses)
        for b in _loaders[self.fake_l.num_workers == 0](self.fake_l):
            if self.device is not None:
                b = to_device(b, self.device)
            yield self.after_batch(b)
        self.after_iter()
        if hasattr(self, 'it'):
            del(self.it)

    def create_batches(self, samps):
        if self.dataset is not None:
            self.it = iter(self.dataset)
        res = filter(lambda o: o is not None, map(self.do_item, samps))
        yield from map(self.do_batch, self.chunkify(res))

    def new(self, dataset=None, cls=None, **kwargs):
        if dataset is None:
            dataset = self.dataset
        if cls is None:
            cls = type(self)
        cur_kwargs = dict(dataset=dataset, num_workers=self.fake_l.num_workers, pin_memory=self.pin_memory, timeout=self.timeout,
                          bs=self.bs, shuffle=self.shuffle, drop_last=self.drop_last, indexed=self.indexed, device=self.device)
        for n in self._methods:
            o = getattr(self, n)
            if not isinstance(o, MethodType):
                cur_kwargs[n] = o
        return cls(**merge(cur_kwargs, kwargs))

    @property
    def prebatched(self): return self.bs is None

    def do_item(self, s):
        try:
            return self.after_item(self.create_item(s))
        except SkipItemException:
            return None

    def chunkify(self, b): return b if self.prebatched else chunked(b, self.bs, self.drop_last)
    def shuffle_fn(self, idxs): return self.rng.sample(idxs, len(idxs))
    def randomize(self): self.rng = random.Random(self.rng.randint(0, 2**32 - 1))
    def retain(self, res, b): return retain_types(res, b[0] if is_listy(b) else b)

    def create_item(self, s):
        if self.indexed:
            return self.dataset[s or 0]
        elif s is None:
            return next(self.it)
        else:
            raise IndexError("Cannot index an iterable dataset numerically - must use `None`.")

    def create_batch(self, b): return (fa_collate, fa_convert)[self.prebatched](b)
    def do_batch(self, b): return self.retain(self.create_batch(self.before_batch(b)), b)
    def to(self, device): self.device = device

    def one_batch(self):
        if self.n is not None and len(self) == 0:
            raise ValueError(f'This DataLoader does not contain any batches')
        with self.fake_l.no_multiproc():
            res = first(self)
        if hasattr(self, 'it'):
            delattr(self, 'it')
        return res


# export
add_docs(DataLoader, "API compatible with PyTorch DataLoader, with a lot more callbacks and flexibility",
         get_idxs="Return a list of indices to reference the dataset. Calls `shuffle_fn` internally if `shuffle=True`.",
         sample="Same as `get_idxs` but returns a generator of indices to reference the dataset.",
         create_batches="Takes output of `sample` as input, and returns batches of data. Does not apply `after_batch`.",
         new="Create a new `DataLoader` with given arguments keeping remaining arguments same as original `DataLoader`.",
         prebatched="Check if `bs` is None.",
         do_item="Combines `after_item` and `create_item` to get an item from dataset by providing index as input.",
         chunkify="Used by `create_batches` to turn generator of items (`b`) into batches.",
         shuffle_fn="Returns a random permutation of `idxs`.",
         randomize="Set's `DataLoader` random number generator state.",
         retain="Cast each item of `res` to type of matching item in `b` if its a superclass.",
         create_item="Subset of the dataset containing the index values of sample if exists, else next iterator.",
         create_batch="Collate a list of items into a batch.",
         do_batch="Combines `create_batch` and `before_batch` to get a batch of items. Input is a list of items to collate.",
         to="Sets `self.device=device`.",
         one_batch="Return one batch from `DataLoader`.",
         wif="See pytorch `worker_init_fn` for details.",
         before_iter="Called before `DataLoader` starts to read/iterate over the dataset.",
         after_item="Takes output of `create_item` as input and applies this function on it.",
         before_batch="It is called before collating a list of items into a batch. Input is a list of items.",
         after_batch="After collating mini-batch of items, the mini-batch is passed through this function.",
         after_iter="Called after `DataLoader` has fully read/iterated over the dataset.")


# Arguments to `DataLoader`:
# * `dataset`: dataset from which to load the data. Can be either map-style or iterable-style dataset.
# * `bs` (int): how many samples per batch to load (if `batch_size` is provided then `batch_size` will override `bs`). If `bs=None`, then it is assumed that `dataset.__getitem__` returns a batch.
# * `num_workers` (int): how many subprocesses to use for data loading. `0` means that the data will be loaded in the main process.
# * `pin_memory` (bool): If `True`, the data loader will copy Tensors into CUDA pinned memory before returning them.
# * `timeout` (float>0): the timeout value in seconds for collecting a batch from workers.
# * `batch_size` (int): It is only provided for PyTorch compatibility. Use `bs`.
# * `shuffle` (bool): If `True`, then data is shuffled every time dataloader is fully read/iterated.
# * `drop_last` (bool): If `True`, then the last incomplete batch is dropped.
# * `indexed` (bool): The `DataLoader` will make a guess as to whether the dataset can be indexed (or is iterable), but you can override it with this parameter. `True` by default.
# * `n` (int): Defaults to `len(dataset)`. If you are using iterable-style dataset, you can specify the size with `n`.
# * `device` (torch.device): Defaults to `default_device()` which is CUDA by default. You can specify device as `torch.device('cpu').

# Override `item` and use the default infinite sampler to get a stream of unknown length (`stop()` when you want to stop the stream).

# +
class RandDL(DataLoader):
    def create_item(self, s):
        r = random.random()
        return r if r < 0.95 else stop()


L(RandDL())
# -

L(RandDL(bs=4, drop_last=True)).map(len)

dl = RandDL(bs=4, num_workers=4, drop_last=True)
L(dl).map(len)

test_num_workers = 0 if sys.platform == "win32" else 4
test_eq(dl.fake_l.num_workers, test_num_workers)
with dl.fake_l.no_multiproc():
    test_eq(dl.fake_l.num_workers, 0)
    L(dl).map(len)
test_eq(dl.fake_l.num_workers, test_num_workers)


# +
def _rand_item(s):
    r = random.random()
    return r if r < 0.95 else stop()


L(DataLoader(create_item=_rand_item))
# -

# If you don't set `bs`, then `dataset` is assumed to provide an iterator or a `__getitem__` that returns a batch.

# +
ds1 = DataLoader(letters)
test_eq(L(ds1), letters)
test_eq(len(ds1), 26)

test_shuffled(L(DataLoader(letters, shuffle=True)), letters)

ds1 = DataLoader(letters, indexed=False)
test_eq(L(ds1), letters)
test_eq(len(ds1), 26)

t2 = L(tensor([0, 1, 2]), tensor([3, 4, 5]))
ds2 = DataLoader(t2)
test_eq_type(L(ds2), t2)

t3 = L(array([0, 1, 2], dtype=np.int64), array([3, 4, 5], dtype=np.int64))
ds3 = DataLoader(t3)
test_eq_type(L(ds3), t3.map(tensor))

ds4 = DataLoader(t3, create_batch=noop, after_iter=lambda: setattr(t3, 'f', 1))
test_eq_type(L(ds4), t3)
test_eq(t3.f, 1)


# -

# If you do set `bs`, then `dataset` is assumed to provide an iterator or a `__getitem__` that returns a single item of a batch.

def twoepochs(d): return ' '.join(''.join(list(o)) for _ in range(2) for o in d)


# +
ds1 = DataLoader(letters, bs=4, drop_last=True, num_workers=0)
test_eq(twoepochs(ds1), 'abcd efgh ijkl mnop qrst uvwx abcd efgh ijkl mnop qrst uvwx')

ds1 = DataLoader(letters, 4, num_workers=2)
test_eq(twoepochs(ds1), 'abcd efgh ijkl mnop qrst uvwx yz abcd efgh ijkl mnop qrst uvwx yz')

ds1 = DataLoader(range(12), bs=4, num_workers=3)
test_eq_type(L(ds1), L(tensor([0, 1, 2, 3]), tensor([4, 5, 6, 7]), tensor([8, 9, 10, 11])))

ds1 = DataLoader([str(i) for i in range(11)], bs=4, after_iter=lambda: setattr(t3, 'f', 2))
test_eq_type(L(ds1), L(['0', '1', '2', '3'], ['4', '5', '6', '7'], ['8', '9', '10']))
test_eq(t3.f, 2)

it = iter(DataLoader(map(noop, range(20)), bs=4, num_workers=1))
test_eq_type([next(it) for _ in range(3)], [tensor([0, 1, 2, 3]), tensor([4, 5, 6, 7]), tensor([8, 9, 10, 11])])


# -

# Iterable dataloaders require specific tests.

# +
class DummyIterableDataset(IterableDataset):
    def __iter__(self):
        yield from range(11)


ds1 = DataLoader(DummyIterableDataset(), bs=4)
# Check it yields fine, and check we can do multiple passes
for i in range(3):
    test_eq_type(L(ds1), L(tensor([0, 1, 2, 3]), tensor([4, 5, 6, 7]), tensor([8, 9, 10])))

# Check `drop_last` works fine (with multiple passes, since this will prematurely terminate the iterator)
ds1 = DataLoader(DummyIterableDataset(), bs=4, drop_last=True)
for i in range(3):
    test_eq_type(L(ds1), L(tensor([0, 1, 2, 3]), tensor([4, 5, 6, 7])))


# +
class SleepyDL(list):
    def __getitem__(self, i):
        time.sleep(random.random() / 50)
        return super().__getitem__(i)


t = SleepyDL(letters)

# %time test_eq(DataLoader(t, num_workers=0), letters)
# %time test_eq(DataLoader(t, num_workers=2), letters)
# %time test_eq(DataLoader(t, num_workers=4), letters)

dl = DataLoader(t, shuffle=True, num_workers=1)
test_shuffled(L(dl), letters)
test_shuffled(L(dl), L(dl))
L(dl)


# +
class SleepyQueue():
    "Simulate a queue with varying latency"

    def __init__(self, q): self.q = q

    def __iter__(self):
        while True:
            time.sleep(random.random() / 100)
            try:
                yield self.q.get_nowait()
            except queues.Empty:
                return


q = Queue()
for o in range(30):
    q.put(o)
it = SleepyQueue(q)

if not (sys.platform == "win32" and IN_NOTEBOOK):
    %time test_shuffled(L(DataLoader(it, num_workers=4)), L(range(30)))


# +
class A(TensorBase):
    pass


for nw in (0, 2):
    t = A(tensor([1, 2]))
    dl = DataLoader([t, t, t, t, t, t, t, t], bs=4, num_workers=nw)
    b = first(dl)
    test_eq(type(b), A)

    t = (A(tensor([1, 2])),)
    dl = DataLoader([t, t, t, t, t, t, t, t], bs=4, num_workers=nw)
    b = first(dl)
    test_eq(type(b[0]), A)
# -

list(DataLoader(list(range(50)), bs=32, shuffle=True, num_workers=3))


# +
class A(TensorBase):
    pass


t = A(tensor(1, 2))

tdl = DataLoader([t, t, t, t, t, t, t, t], bs=4, num_workers=2, after_batch=to_device)
b = first(tdl)
test_eq(type(b), A)

# Unknown attributes are delegated to `dataset`
test_eq(tdl.pop(), tensor(1, 2))


# -

# Override `get_idxs` to return the same index until consumption of the DL. This is intented to test consistent sampling behavior when `num_workers`>1.

# +
class AdamantDL(DataLoader):
    def get_idxs(self):
        r = random.randint(0, self.n - 1)
        return [r] * self.n


test_eq(torch.cat(tuple(AdamantDL((list(range(50))), bs=16, num_workers=4))).unique().numel(), 1)
# -

# ## Export -

# hide
notebook2script()

# test num_workers > 0 in scripts works when python process start method is spawn
process = Popen(["python", "dltest.py"], stdout=PIPE)
_, err = process.communicate(timeout=15)
exit_code = process.wait()
test_eq(exit_code, 0)
