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
from fastai.data.load import _FakeLoader, _loaders
from nbdev.export import notebook2script
from fastai.callback.data import WeightedDL
from torch.nn.parallel import DistributedDataParallel, DataParallel
from fastai.callback.progress import ProgressCallback
from fastai.basics import *
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# +
# default_exp distributed
# -

# export


# # Distributed and parallel training
#
# > Callbacks and helper functions to train in parallel or use distributed training

# ## Parallel

# Patch the parallel models so they work with RNNs

# export
@patch
def reset(self: DataParallel):
    if hasattr(self.module, 'reset'):
        self.module.reset()


# export
@log_args
class ParallelTrainer(Callback):
    run_after, run_before = TrainEvalCallback, Recorder
    def __init__(self, device_ids): self.device_ids = device_ids
    def before_fit(self): self.learn.model = DataParallel(self.learn.model, device_ids=self.device_ids)
    def after_fit(self): self.learn.model = self.learn.model.module


# export
@patch
def to_parallel(self: Learner, device_ids=None):
    self.add_cb(ParallelTrainer(device_ids))
    return self


# export
@patch
def detach_parallel(self: Learner):
    "Remove ParallelTrainer callback from Learner."
    self.remove_cb(ParallelTrainer)
    return self


# export
@patch
@contextmanager
def parallel_ctx(self: Learner, device_ids=None):
    "A context manager to adapt a learner to train in data parallel mode."
    try:
        self.to_parallel(device_ids)
        yield self
    finally:
        self.detach_parallel()


# ## Distributed

# Patch the parallel models so they work with RNNs

# export
@patch
def reset(self: DistributedDataParallel):
    if hasattr(self.module, 'reset'):
        self.module.reset()


# Convenience functions to set up/tear down torch distributed data parallel mode.

# export
def setup_distrib(gpu=None):
    if gpu is None:
        return gpu
    gpu = int(gpu)
    torch.cuda.set_device(int(gpu))
    if num_distrib() > 0:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    return gpu


# export
def teardown_distrib():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


# ### DataLoader

# We need to change the dataloaders so that they only get one part of the batch each (otherwise there is no point in using distributed training).

# export
@log_args(but_as=TfmdDL.__init__)
class DistributedDL(TfmdDL):

    def _round_to_multiple(number, multiple): return int(math.ceil(number / multiple) * multiple)

    def _broadcast(self, t, rank):
        "Broadcasts t from rank `rank` to all other ranks. Returns t so t is same for all ranks after call."
        t = LongTensor(t).cuda()  # nccl only works with cuda tensors
        torch.distributed.broadcast(t, rank)
        return t.cpu().tolist()

    def _to_detach(self, b, cpu=True, gather=True): return to_detach(b, cpu, gather)  # member func so we can override for test

    def __len__(self):
        return DistributedDL._round_to_multiple(len(self.dl), self.world_size) // self.world_size

    def get_idxs(self):
        idxs = self.dl.get_idxs()       # compute get_idxs in all ranks (we'll only use rank 0 but size must be consistent)
        idxs = self._broadcast(idxs, 0)  # broadcast and receive it from rank 0 to all
        self.n = len(idxs)              # we assumed n was dl.n but we really care about number of idxs
        # add extra samples to make it evenly divisible
        self.n_padded = DistributedDL._round_to_multiple(self.n, self.world_size)
        idxs += (idxs * (self.n_padded // self.n))[:self.n_padded - self.n]  # idx needs to be repeated when n_padded>>n
        # slice padded idxs so that each rank gets self.n_padded//self.world_size tensors
        return idxs[self.rank * self.n_padded // self.world_size:(self.rank + 1) * self.n_padded // self.world_size]

    def before_iter(self):
        self.i = 0
        self.dl.before_iter()

    def randomize(self): self.dl.randomize()

    def after_batch(self, b):
        self.i += find_bs(b)
        return self.dl.after_batch(b)

    def after_iter(self):
        self.dl.after_iter()

    def create_batches(self, samps): return self.dl.create_batches(samps)

    def to_detach(self, b, cpu=True, gather=True):
        b = self._to_detach(b, cpu, gather)

        def _inner(b):
            if b.ndim > 0:
                # for each rank, compute overflow of read idxs vs self.n and accumulate them to unpad totals after gathering
                n = sum([min(0, max(-len(b) // self.world_size,
                                    self.n - (self.i + r * self.n_padded // self.world_size))) for r in range(self.world_size)])
                b = b[:n or None]
            return b
        return apply(_inner, b) if gather and hasattr(self, 'i') and hasattr(self, 'n') and hasattr(self, 'n_padded') else b

    def __init__(self, dl, rank, world_size):
        store_attr('dl,rank,world_size')
        self.bs, self.device, self.drop_last, self.dataset = dl.bs, dl.device, dl.drop_last, dl.dataset
        self.fake_l = _FakeLoader(self, dl.fake_l.pin_memory, dl.fake_l.num_workers, dl.fake_l.timeout)


# hide
_tmp_file = tempfile.NamedTemporaryFile().name  # i tried putting this inside self / _broadcast to no avail
# patch _broadcast with a mocked version so we can test DistributedDL w/o a proper DDP setup


@patch
def _broadcast(self: DistributedDL, t, rank):
    t = LongTensor(t)
    if rank == self.rank:
        torch.save(t, _tmp_file)
    else:
        t.data = torch.load(_tmp_file)
    return t.tolist()
# patch _to_detach with a mocked version that will return right gathered size but -100 for other rank tensors


@patch
def _to_detach(self: DistributedDL, b, cpu=True, gather=True):
    b = to_detach(b, cpu, gather)
    if not gather:
        return b

    def _inner(b, cpu, gather):
        if b.ndim == 0:
            b = b[None]
        b = torch.cat([b if i == self.rank else torch.full_like(b, -100) for i in range(self.world_size)])
        return b if b.ndim > 0 else b.mean()
    return apply(_inner, b, cpu, gather)


dl = TfmdDL(list(range(50)), bs=12, num_workers=2)
for i in range(4):
    dl1 = DistributedDL(dl, i, 4)
    test_eq(list(dl1), (torch.arange(i * 13, i * 13 + 12) % 50, torch.tensor([i * 13 + 12]) % 50))

dl = TfmdDL(list(zip(range(50), range(100, 150))), bs=12, num_workers=4)
for i in range(4):
    dl1 = DistributedDL(dl, i, 4)
    test_eq(list(dl1), [(torch.arange(i * 13, i * 13 + 12) % 50, 100 + torch.arange(i * 13, i * 13 + 12) % 50),
                        ((torch.tensor([i * 13 + 12]) % 50), 100 + torch.tensor([i * 13 + 12]) % 50)])

dl = TfmdDL(list(range(50)), bs=12, num_workers=2, drop_last=True)
for i in range(4):
    dl1 = DistributedDL(dl, i, 4)
    test_eq(list(dl1), [torch.arange(i * 13, i * 13 + 12) % 50])

dl = TfmdDL(list(zip(range(12), range(100, 112))), bs=12, num_workers=4)
res, dls = [], []
for i in range(5):
    dls.append(DistributedDL(dl, i, 5))
for b in zip(*dls):
    for r in range(5):
        d = L(dls[r].to_detach(b[r]))
        test_eq(d.map(lambda x: (x != -100).sum().item()), (3, 3) if r != 4 else (0, 0))

dl = TfmdDL(list(range(10)), bs=4, num_workers=2, shuffle=True)
res = []
for i in range(3):
    dl1 = DistributedDL(dl, i, 3)
    b = list(dl1)[0]
    bd = dl1.to_detach(b)
    test_eq(b[:None if i < 2 else 2], bd[4 * i:4 * (i + 1)])


dl = WeightedDL(list(range(50)), bs=16, num_workers=2, shuffle=True, wgts=list(np.arange(50) >= 25))
res = []
for i in range(4):
    dl1 = DistributedDL(dl, i, 4)
    res += list(dl1)[0].tolist()
test(res, [25] * len(res), operator.ge)        # all res >=25
test(res, [25] * len(res), lambda a, b: ~(a < b))  # all res NOT < 25


# export
@log_args
class DistributedTrainer(Callback):
    run_after, run_before = TrainEvalCallback, Recorder
    fup = None  # for `find_unused_parameters` in DistributedDataParallel()
    def __init__(self, cuda_id=0, sync_bn=True): store_attr('cuda_id,sync_bn')

    def before_fit(self):
        opt_kwargs = {'find_unused_parameters': DistributedTrainer.fup} if DistributedTrainer.fup is not None else {}
        self.learn.model = DistributedDataParallel(
            nn.SyncBatchNorm.convert_sync_batchnorm(self.model) if self.sync_bn else self.model,
            device_ids=[self.cuda_id], output_device=self.cuda_id, **opt_kwargs)
        self.old_dls = list(self.dls)
        self.learn.dls.loaders = [self._wrap_dl(dl) for dl in self.dls]
        if rank_distrib() > 0:
            self.learn.logger = noop

    def _wrap_dl(self, dl):
        return dl if isinstance(dl, DistributedDL) else DistributedDL(dl, rank_distrib(), num_distrib())

    def before_train(self): self.learn.dl = self._wrap_dl(self.learn.dl)
    def before_validate(self): self.learn.dl = self._wrap_dl(self.learn.dl)

    def after_fit(self):
        self.learn.model = self.learn.model.module
        self.learn.dls.loaders = self.old_dls


# Attach, remove a callback which adapts the model to use DistributedDL to train in distributed data parallel mode.

# export
@patch
def to_distributed(self: Learner, cuda_id, sync_bn=True):
    self.add_cb(DistributedTrainer(cuda_id, sync_bn))
    if rank_distrib() > 0:
        self.remove_cb(ProgressCallback)
    return self


# export
@patch
def detach_distributed(self: Learner):
    if num_distrib() <= 1:
        return self
    self.remove_cb(DistributedTrainer)
    if rank_distrib() > 0 and not hasattr(self, 'progress'):
        self.add_cb(ProgressCallback())
    return self


# export
@patch
@contextmanager
def distrib_ctx(self: Learner, cuda_id=None, sync_bn=True):
    "A context manager to adapt a learner to train in distributed data parallel mode."
    # Figure out the GPU to use from rank.  Create a dpg if none exists yet.
    if cuda_id is None:
        cuda_id = rank_distrib()
    if not torch.distributed.is_initialized():
        setup_distrib(cuda_id)
        cleanup_dpg = torch.distributed.is_initialized()
    else:
        cleanup_dpg = False
    # Adapt self to DistributedDataParallel, yield, and cleanup afterwards.
    try:
        if num_distrib() > 1:
            self.to_distributed(cuda_id, sync_bn)
        yield self
    finally:
        self.detach_distributed()
        if cleanup_dpg:
            teardown_distrib()


# ### `distrib_ctx` context manager
#
# **`distrib_ctx(cuda_id)`** prepares a learner to train in distributed data parallel mode.  It assumes these [environment variables](https://pytorch.org/tutorials/intermediate/dist_tuto.html#initialization-methods) have all been setup properly, such as those launched by [`python -m fastai.launch`](https://github.com/fastai/fastai/blob/master/fastai/launch.py).
#
# #### Typical usage:
# ```
# with learn.distrib_ctx(): learn.fit(.....)
# ```
#
# It attaches a `DistributedTrainer` callback and `DistributedDL` data loader to  the learner, then executes `learn.fit(.....)`.  Upon exiting the context, it removes the `DistributedTrainer` and `DistributedDL`, and destroys any locally created distributed process group.  The process is still attached to the GPU though.
#

# export
def rank0_first(func):
    "Execute `func` in the Rank-0 process first, then in other ranks in parallel."
    dummy_l = Learner(DataLoaders(device='cpu'), nn.Linear(1, 1), loss_func=lambda: 0)
    with dummy_l.distrib_ctx():
        if rank_distrib() == 0:
            res = func()
        distrib_barrier()
        if rank_distrib() != 0:
            res = func()
    return res


# **`rank0_first(f)`** calls `f()` in rank-0 process first, then in parallel on the rest, in distributed training mode. In single process, non-distributed training mode, `f()` is called only once as expected.
#
# One application of `rank0_first()` is to make fresh downloads via `untar_data()` safe in distributed training scripts launched by `python -m fastai.launch <script>`:
#
# > <code>path = untar_data(URLs.IMDB)</code>
#
# becomes:
#
# > <code>path = <b>rank0_first(lambda:</b> untar_data(URLs.IMDB))</code>
#
#
# Some learner factory methods may use `untar_data()` to **download pretrained models** by default:
#
# > <code>learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)</code>
#
# becomes:
#
# > <code>learn = <b>rank0_first(lambda:</b> text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy))</code>
#
# Otherwise, multiple processes will download at the same time and corrupt the data.
#

# ## Export -

# hide
notebook2script()
