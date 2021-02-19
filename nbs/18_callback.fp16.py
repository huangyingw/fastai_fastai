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
from torch.cuda.amp import GradScaler, autocast
from nbdev.export import *
from torch.nn.utils import parameters_to_vector
from fastai.fp16_utils import convert_network, model_grads_to_master_grads, master_params_to_model_params
from nbdev.showdoc import *
from fastai.test_utils import *
from torch.cuda.amp.grad_scaler import OptState
from fastai.callback.progress import *
from fastai.basics import *
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# +
# all_cuda

# +
# export


# +
# default_exp callback.fp16
# -

# hide


# # Mixed precision training
#
# > Callback and utility functions to allow mixed precision training

# ## A little bit of theory

# A very nice and clear introduction to mixed precision training is [this video from NVIDIA](http://on-demand.gputechconf.com/gtc/2018/video/S81012/).

# ### What's half precision?

# In neural nets, all the computations are usually done in single precision, which means all the floats in all the arrays that represent inputs, activations, weights... are 32-bit floats (FP32 in the rest of this post). An idea to reduce memory usage (and avoid those annoying cuda errors) has been to try and do the same thing in half-precision, which means using 16-bits floats (or FP16 in the rest of this post). By definition, they take half the space in RAM, and in theory could allow you to double the size of your model and double your batch size.
#
# Another very nice feature is that NVIDIA developed its latest GPUs (the Volta generation) to take fully advantage of half-precision tensors. Basically, if you give half-precision tensors to those, they'll stack them so that each core can do more operations at the same time, and theoretically gives an 8x speed-up (sadly, just in theory).
#
# So training at half precision is better for your memory usage, way faster if you have a Volta GPU (still a tiny bit faster if you don't since the computations are easiest). How do we do it? Super easily in pytorch, we just have to put .half() everywhere: on the inputs of our model and all the parameters. Problem is that you usually won't see the same accuracy in the end (so it happens sometimes) because half-precision is... well... not as precise ;).

# ### Problems with half-precision:

# To understand the problems with half precision, let's look briefly at what an FP16 looks like (more information [here](https://en.wikipedia.org/wiki/Half-precision_floating-point_format)).
#
# ![half float](images/half.png)
#
# The sign bit gives us +1 or -1, then we have 5 bits to code an exponent between -14 and 15, while the fraction part has the remaining 10 bits. Compared to FP32, we have a smaller range of possible values (2e-14 to 2e15 roughly, compared to 2e-126 to 2e127 for FP32) but also a smaller *offset*.
#
# For instance, between 1 and 2, the FP16 format only represents the number 1, 1+2e-10, 1+2*2e-10... which means that 1 + 0.0001 = 1 in half precision. That's what will cause a certain numbers of problems, specifically three that can occur and mess up your training.
# 1. The weight update is imprecise: inside your optimizer, you basically do w = w - lr * w.grad for each weight of your network. The problem in performing this operation in half precision is that very often, w.grad is several orders of magnitude below w, and the learning rate is also small. The situation where w=1 and lr*w.grad is 0.0001 (or lower) is therefore very common, but the update doesn't do anything in those cases.
# 2. Your gradients can underflow. In FP16, your gradients can easily be replaced by 0 because they are too low.
# 3. Your activations or loss can overflow. The opposite problem from the gradients: it's easier to hit nan (or infinity) in FP16 precision, and your training might more easily diverge.

# ### The solution: mixed precision training

# To address those three problems, we don't fully train in FP16 precision. As the name mixed training implies, some of the operations will be done in FP16, others in FP32. This is mainly to take care of the first problem listed above. For the next two there are additional tricks.
#
# The main idea is that we want to do the forward pass and the gradient computation in half precision (to go fast) but the update in single precision (to be more precise). It's okay if w and grad are both half floats, but when we do the operation w = w - lr * grad, we need to compute it in FP32. That way our 1 + 0.0001 is going to be 1.0001.
#
# This is why we keep a copy of the weights in FP32 (called master model). Then, our training loop will look like:
# 1. compute the output with the FP16 model, then the loss
# 2. back-propagate the gradients in half-precision.
# 3. copy the gradients in FP32 precision
# 4. do the update on the master model (in FP32 precision)
# 5. copy the master model in the FP16 model.
#
# Note that we lose precision during step 5, and that the 1.0001 in one of the weights will go back to 1. But if the next update corresponds to add 0.0001 again, since the optimizer step is done on the master model, the 1.0001 will become 1.0002 and if we eventually go like this up to 1.0005, the FP16 model will be able to tell the difference.
#
# That takes care of problem 1. For the second problem, we use something called gradient scaling: to avoid the gradients getting zeroed by the FP16 precision, we multiply the loss by a scale factor (scale=512 for instance). That way we can push the gradients to the right in the next figure, and have them not become zero.
#
# ![half float representation](images/half_representation.png)
#
# Of course we don't want those 512-scaled gradients to be in the weight update, so after converting them into FP32, we can divide them by this scale factor (once they have no risks of becoming 0). This changes the loop to:
# 1. compute the output with the FP16 model, then the loss.
# 2. multiply the loss by scale then back-propagate the gradients in half-precision.
# 3. copy the gradients in FP32 precision then divide them by scale.
# 4. do the update on the master model (in FP32 precision).
# 5. copy the master model in the FP16 model.
#
# For the last problem, the tricks offered by NVIDIA are to leave the batchnorm layers in single precision (they don't have many weights so it's not a big memory challenge) and compute the loss in single precision (which means converting the last output of the model in single precision before passing it to the loss).
#
# ![Mixed precision training](images/Mixed_precision.jpeg)

# ### Dynamic loss scaling

# The only annoying thing with the previous implementation of mixed precision training is that it introduces one new hyper-parameter to tune, the value of the loss scaling. Fortunately for us, there is a way around this. We want the loss scaling to be as high as possible so that our gradients can use the whole range of representation, so let's first try a really high value. In all likelihood, this will cause our gradients or our loss to overflow, and we will try again with half that big value, and again, until we get to the largest loss scale possible that doesn't make our gradients overflow.
#
# This value will be perfectly fitted to our model and can continue to be dynamically adjusted as the training goes, if it's still too high, by just halving it each time we overflow. After a while though, training will converge and gradients will start to get smaller, so we al
# so need a mechanism to get this dynamic loss scale larger if it's safe to do so. The strategy used in the Apex library is to multiply the loss scale by 2 each time we had a given number of iterations without overflowing.

# ## MixedPrecision -

# export
@delegates(GradScaler)
class MixedPrecision(Callback):
    "Mixed precision training using Pytorch's `autocast` and `GradScaler`"
    order = 10
    def __init__(self, **kwargs): self.kwargs, self.autocast = kwargs, autocast()
    def before_fit(self): self.learn.scaler, self.scales = GradScaler(**self.kwargs), L()
    def before_batch(self): self.autocast.__enter__()

    def after_pred(self):
        if listify(self.pred)[0].dtype == torch.float16:
            self.learn.pred = to_float(self.pred)

    def after_loss(self): self.autocast.__exit__()
    def before_backward(self): self.learn.loss_grad = self.scaler.scale(self.loss_grad)

    def before_step(self):
        self.skipped = True
        self.scaler.step(self)
        if self.skipped:
            raise CancelStepException()
        self.scales.append(self.scaler.get_scale())

    def after_step(self): self.learn.scaler.update()

    @property  # pretend to be an optimizer for `GradScaler`
    def param_groups(self): return self.opt.param_groups
    def step(self, *args, **kwargs): self.skipped = False


# export
class FP16TestCallback(Callback):
    "Asserts that predictions are `float16` values"
    order = 9
    def after_pred(self): assert listify(self.pred)[0].dtype == torch.float16


# cuda
set_seed(99, True)
learn = synth_learner(cbs=[MixedPrecision, FP16TestCallback], cuda=True)
learn.model = nn.Sequential(nn.Linear(1, 1), nn.Linear(1, 1)).cuda()
learn.opt_func = partial(SGD, mom=0.)
learn.splitter = lambda m: [list(m[0].parameters()), list(m[1].parameters())]
learn.fit(3)
assert learn.recorder.values[-1][-1] < learn.recorder.values[0][-1]

# hide
# cuda
# Multioutput version
set_seed(99, True)
learn = synth_learner(cbs=[MixedPrecision, FP16TestCallback], cuda=True)


class MultiOutputModel(Module):
    def __init__(self): self.linear1, self.linear2 = nn.Linear(1, 1), nn.Linear(1, 1)
    def forward(self, x): return self.linear1(x), self.linear2(x)


def multioutputloss(pred, val): return (val - pred[0]).abs() + 0.5 * (val - pred[1]).abs()


learn.model = MultiOutputModel()
learn.opt_func = partial(SGD, mom=0.)
learn.splitter = lambda m: [list(m.linear1.parameters()), list(m.linear2.parameters())]
learn.loss_func = multioutputloss
learn.fit(3)
assert learn.recorder.values[-1][-1] < learn.recorder.values[0][-1]


# export
@patch
@delegates(GradScaler)
def to_fp16(self: Learner, **kwargs): return self.add_cb(MixedPrecision(**kwargs))


# export
@patch
def to_fp32(self: Learner): return self.remove_cb(MixedPrecision)


# ## Util functions

# Before going in the main `Callback` we will need some helper functions. We use the ones from the [APEX library](https://github.com/NVIDIA/apex).

# export

# ### Converting the model to FP16

# We will need a function to convert all the layers of the model to FP16 precision except the BatchNorm-like layers (since those need to be done in FP32 precision to be stable). In Apex, the function that does this for us is `convert_network`. We can use it to put the model in FP16 or back to FP32.

# +
model = nn.Sequential(nn.Linear(10, 30), nn.BatchNorm1d(30), nn.Linear(30, 2)).cuda()
model = convert_network(model, torch.float16)

for i, t in enumerate([torch.float16, torch.float32, torch.float16]):
    test_eq(model[i].weight.dtype, t)
    test_eq(model[i].bias.dtype, t)

model = nn.Sequential(nn.Linear(10, 30), BatchNorm(30, ndim=1), nn.Linear(30, 2)).cuda()
model = convert_network(model, torch.float16)

for i, t in enumerate([torch.float16, torch.float32, torch.float16]):
    test_eq(model[i].weight.dtype, t)
    test_eq(model[i].bias.dtype, t)
# -

# ### Creating the master copy of the parameters

# From our model parameters (mostly in FP16), we'll want to create a copy in FP32 (master parameters) that we will use for the step in the optimizer. Optionally, we concatenate all the parameters to do one flat big tensor, which can make that step a little bit faster.
#
# We can't use the FP16 util function here as it doesn't handle multiple parameter groups, which is the thing we use to
# - do transfer learning and freeze some layers
# - apply discriminative learning rates
# - don't apply weight decay to some layers (like BatchNorm) or the bias terms

# export


# export
def get_master(opt, flat_master=False):
    model_params = [[param for param in pg if getattr(param, 'requires_grad', False) and hasattr(param, 'data')] for pg in opt.param_lists]
    if flat_master:
        master_params = []
        for pg in model_params:
            mp = parameters_to_vector([param.data.float() for param in pg])
            mp = nn.Parameter(mp, requires_grad=True)
            if mp.grad is None:
                mp.grad = mp.new(*mp.size())
            master_params.append([mp])
    else:
        master_params = [[nn.Parameter(param.data.clone().float().detach(), requires_grad=True) for param in pg] for pg in model_params]
    return model_params, master_params


# hide
# cuda
learn = synth_learner()
learn.model = convert_network(nn.Sequential(nn.Linear(1, 1), nn.Linear(1, 1)), torch.float16).cuda()
learn.splitter = lambda m: [list(m[0].parameters()), list(m[1].parameters())]
learn.opt = learn.opt_func(learn.splitter(learn.model), learn.lr)
model_p, master_p = get_master(learn.opt)
test_eq(len(model_p), 2)  # 2 pqrqm groups
test_eq(len(master_p), 2)
for pg1, pg2 in zip(model_p, master_p):
    test_eq([p.float() for p in pg1], pg2)  # Same values but different types
    for p in pg1:
        assert p.dtype == torch.float16

# hide
# cuda
# Flattened version
model_pf, master_pf = get_master(learn.opt, flat_master=True)
test_eq(len(model_pf), 2)  # 2 pqrqm groups
test_eq(len(master_pf), 2)
for pg1, pg2 in zip(model_pf, master_pf):
    test_eq(len(pg2), 1)  # One flattened tensor
    test_eq([p.float().squeeze() for p in pg1], [p for p in pg2[0]])  # Same values but different types
    for p in pg1:
        assert p.dtype == torch.float16


# ### Copy the gradients from model params to master params

# After the backward pass, all gradients must be copied to the master params before the optimizer step can be done in FP32. The corresponding function in the Apex utils is `model_grads_to_master_grads` but we need to adapt it to work with param groups.

# export
def to_master_grads(model_pgs, master_pgs, flat_master=False):
    for (model_params, master_params) in zip(model_pgs, master_pgs):
        model_grads_to_master_grads(model_params, master_params, flat_master=flat_master)


# hide
# cuda
xb, yb = learn.dls.one_batch()
pred = learn.model.cuda()(xb.cuda().half())
loss = F.mse_loss(pred, yb.cuda().half())
loss.backward()
to_master_grads(model_p, master_p)
to_master_grads(model_pf, master_pf, flat_master=True)
test_eq([[p.grad.float() for p in pg] for pg in model_p],
        [[p.grad for p in pg] for pg in master_p])
test_eq([[p.grad.float().squeeze() for p in pg] for pg in model_pf],
        [[p for p in pg[0].grad] for pg in master_pf])
xb.shape


# ### Copy the master params to the model params

# After the step, we need to copy back the master parameters to the model parameters for the next update. The corresponding function in Apex is `master_params_to_model_params`.

# export
def to_model_params(model_pgs, master_pgs, flat_master=False) -> None:
    for (model_params, master_params) in zip(model_pgs, master_pgs):
        master_params_to_model_params(model_params, master_params, flat_master=flat_master)


# hide
# cuda
learn.opt.params = master_p
learn.opt.step()
to_model_params(model_p, master_p)
test_close([p.float() for pg in model_p for p in pg], [p for pg in master_p for p in pg], eps=1e-3)

# hide
# cuda
learn.opt.params = master_pf
learn.opt.step()
to_model_params(model_pf, master_pf, flat_master=True)
test_close([p.float().squeeze() for pg in model_pf for p in pg], [p for pg in master_pf for p in pg[0]], eps=1e-3)


# ### Checking for overflow

# For dynamic loss scaling, we need to know when the gradients have gone up to infinity. It's faster to check it on the sum than to do `torch.isinf(x).any()`.

# export
def test_overflow(x):
    s = float(x.float().sum())
    return (s == float('inf') or s == float('-inf') or s != s)


x = torch.randn(3, 4)
assert not test_overflow(x)
x[1, 2] = float('inf')
assert test_overflow(x)


# Then we can use it in the following function that checks for gradient overflow:

# export
def grad_overflow(pgs):
    for pg in pgs:
        for p in pg:
            if p.grad is not None and test_overflow(p.grad.data):
                return True
    return False


# hide
# cuda
assert not grad_overflow(model_p)
assert not grad_overflow(model_pf)
model_p[1][0].grad.data[0, 0] = float('inf')
model_pf[0][1].grad.data[0] = float('inf')
assert grad_overflow(model_p)
assert grad_overflow(model_pf)


# ## NonNativeMixedPrecision -

# export
def copy_clone(d):
    return {k: (v.detach().clone().float() if isinstance(v, Tensor) else v) for k, v in d.items()}


# export
def _copy_state(opt, pgs1, pgs2):
    opt.param_lists = pgs2
    for pg1, pg2 in zip(pgs1, pgs2):
        for p1, p2 in zip(pg1, pg2):
            opt.state[p2] = copy_clone(opt.state.pop(p1, {}))


# export
class ModelToHalf(Callback):
    "Use with NonNativeMixedPrecision callback (but it needs to run at the very beginning)"
    order = -50
    def before_fit(self): self.learn.model = convert_network(self.model, dtype=torch.float16)
    def after_fit(self): self.learn.model = convert_network(self.model, dtype=torch.float32)


# export
@docs
class NonNativeMixedPrecision(Callback):
    "Run training in mixed precision"
    order = 10

    def __init__(self, loss_scale=512, flat_master=False, dynamic=True, max_loss_scale=2.**24,
                 div_factor=2., scale_wait=500, clip=None):
        assert torch.backends.cudnn.enabled, "Mixed precision training requires cudnn."
        self.flat_master, self.dynamic, self.max_loss_scale = flat_master, dynamic, max_loss_scale
        self.div_factor, self.scale_wait, self.clip = div_factor, scale_wait, clip
        self.loss_scale = max_loss_scale if dynamic else loss_scale

    def before_fit(self):
        assert self.dls.device.type == 'cuda', "Mixed-precision training requires a GPU, remove the call `to_fp16`"
        if self.learn.opt is None:
            self.learn.create_opt()
        self.model_pgs, self.master_pgs = get_master(self.opt, self.flat_master)
        self.old_pgs = self.opt.param_lists
        # Changes the optimizer so that the optimization step is done in FP32.
        _copy_state(self.learn.opt, self.model_pgs, self.master_pgs)
        if self.dynamic:
            self.count = 0

    def before_batch(self): self.learn.xb = to_half(self.xb)
    def after_pred(self): self.learn.pred = to_float(self.pred)
    def before_backward(self): self.learn.loss_grad *= self.loss_scale

    def before_step(self):
        # First, check for an overflow
        if self.dynamic and grad_overflow(self.model_pgs):
            self.loss_scale /= self.div_factor
            self.learn.loss_grad /= self.div_factor  # to record correct loss
            self.model.zero_grad()
            raise CancelBatchException()  # skip step and zero_grad
        to_master_grads(self.model_pgs, self.master_pgs, self.flat_master)
        for master_params in self.master_pgs:
            for param in master_params:
                if param.grad is not None:
                    param.grad.div_(self.loss_scale)
        if self.clip is not None:
            for group in self.master_pgs:
                nn.utils.clip_grad_norm_(group, self.clip)
        # Check if it's been long enough without overflow
        if self.dynamic:
            self.count += 1
            if self.count == self.scale_wait:
                self.count = 0
                self.loss_scale *= self.div_factor

    def after_step(self):
        self.model.zero_grad()  # Zero the gradients of the model manually (optimizer disconnected)
        to_model_params(self.model_pgs, self.master_pgs, self.flat_master)

    def after_batch(self):
        if self.training:
            self.learn.loss_grad /= self.loss_scale  # Log correct loss

    def after_fit(self):
        if not hasattr(self, 'master_pgs'):
            return
        _copy_state(self.learn.opt, self.master_pgs, self.model_pgs)
        self.learn.opt.param_lists = self.old_pgs
        delattr(self, "master_pgs")
        delattr(self, "model_pgs")
        delattr(self, "old_pgs")

    _docs = dict(before_fit="Put the model in FP16 and prepare the two copies of the parameters",
                 before_batch="Put the input in FP16",
                 after_pred="Put the output back to FP32 so that the loss is computed in FP32",
                 before_backward="Apply loss scaling to avoid gradient underflow",
                 before_step="Copy the gradients to the master param and undo the loss scaling",
                 after_step="Copy the master params to the model params",
                 after_batch="Ensure loss is logged correctly",
                 after_fit="Put the model back in FP32")


# +
# hide
class TestBeforeMixedPrecision(Callback):
    order = -55
    def before_fit(self): test_eq(first(self.model.parameters()).dtype, torch.float32)
    def before_batch(self): test_eq(self.x.dtype, torch.float32)
    def after_pred(self): test_eq(self.pred.dtype, torch.float16)
    def after_loss(self): self.tst_loss = self.learn.loss_grad.detach().clone()

    def before_step(self):
        self.learn.has_overflown = grad_overflow(self.non_native_mixed_precision.model_pgs)
        self.grads = [p.grad.data.clone() for p in self.model.parameters()]
        self.old_params = [p.data.clone() for p in self.model.parameters()]

    def after_cancel_step(self): assert self.has_overflown


class TestAfterMixedPrecision(Callback):
    order = 65
    def before_fit(self): test_eq(first(self.model.parameters()).dtype, torch.float16)
    def after_fit(self): test_eq(first(self.model.parameters()).dtype, torch.float32)
    def before_batch(self): test_eq(self.x.dtype, torch.float16)
    def after_pred(self): test_eq(self.pred.dtype, torch.float32)

    def before_backward(self):
        loss_scale = self.non_native_mixed_precision.loss_scale if self.training else 1.
        test_eq(self.loss_grad, self.test_before_mixed_precision.tst_loss * loss_scale)

    def before_step(self):
        tbmp = self.test_before_mixed_precision
        test_eq(self.loss_grad, tbmp.loss_grad)
        # Test gradients have been copied and scaled back
        test_close(sum([[p.grad.data for p in pg] for pg in self.non_native_mixed_precision.master_pgs], []),
                   [g.float() / self.non_native_mixed_precision.loss_scale for g in tbmp.grads])

    def after_batch(self):
        if self.has_overflown:
            return
        tbmp, mp = self.test_before_mixed_precision, self.non_native_mixed_precision
        # Test master params have been copied to model
        test_close(sum([[p.data for p in pg] for pg in mp.master_pgs], []),
                   [p.data.float() for p in self.model.parameters()], eps=1e-3)
        # Test update has been done properly
        for p, g, op in zip(self.model.parameters(), tbmp.grads, tbmp.old_params):
            test_close(p.data.float(), op.float() - self.lr * g.float() / self.non_native_mixed_precision.loss_scale, eps=1e-3)


# -

# hide
# cuda
learn = synth_learner(cbs=[ModelToHalf(), NonNativeMixedPrecision()], cuda=True)
learn.model = nn.Sequential(nn.Linear(1, 1), nn.Linear(1, 1)).cuda()
learn.opt_func = partial(SGD, mom=0.)
learn.splitter = lambda m: [list(m[0].parameters()), list(m[1].parameters())]
learn.fit(3, cbs=[TestAfterMixedPrecision(), TestBeforeMixedPrecision()])
# Check loss scale did change
assert 1 < learn.non_native_mixed_precision.loss_scale < 2**24
# Check the model did train
for v1, v2 in zip(learn.recorder.values[0], learn.recorder.values[-1]):
    assert v2 < v1

# hide
# cuda
learn = synth_learner(cbs=[ModelToHalf(), NonNativeMixedPrecision(dynamic=False)], cuda=True)
learn.model = nn.Sequential(nn.Linear(1, 1), nn.Linear(1, 1)).cuda()
learn.opt_func = partial(SGD, mom=0.)
learn.splitter = lambda m: [list(m[0].parameters()), list(m[1].parameters())]
learn.fit(3, cbs=[TestAfterMixedPrecision(), TestBeforeMixedPrecision()])
# Check loss scale did mot change
test_eq(learn.non_native_mixed_precision.loss_scale, 512)
# Check the model did train
for v1, v2 in zip(learn.recorder.values[0], learn.recorder.values[-1]):
    assert v2 < v1


# export
@patch
@delegates(NonNativeMixedPrecision.__init__)
def to_non_native_fp16(self: Learner, **kwargs): return self.add_cbs([ModelToHalf(), NonNativeMixedPrecision(**kwargs)])


# cuda
learn = synth_learner(cuda=True)
learn.model = nn.Sequential(nn.Linear(1, 1), nn.Linear(1, 1)).cuda()
learn.opt_func = partial(SGD, mom=0.)
learn.splitter = lambda m: [list(m[0].parameters()), list(m[1].parameters())]
learn.to_non_native_fp16()
learn.fit(3, cbs=[TestAfterMixedPrecision(), TestBeforeMixedPrecision()])
# Check the model did train
for v1, v2 in zip(learn.recorder.values[0], learn.recorder.values[-1]):
    assert v2 < v1

# hide
# cuda
learn = synth_learner(cuda=True)
learn.model = nn.Sequential(nn.Linear(1, 1), nn.Linear(1, 1)).cuda()
learn.opt_func = partial(SGD, mom=0.9)
learn.splitter = lambda m: [list(m[0].parameters()), list(m[1].parameters())]
learn.to_non_native_fp16()
learn.freeze()
learn.create_opt()
init_ps = [p for pg in learn.opt.param_groups for p in pg]
learn.fit(3)
final_ps = [p for pg in learn.opt.param_groups for p in pg]
for p1, p2 in zip(init_ps, final_ps):
    test_is(p1, p2)
# First param groups has no state because not trained
test_eq([learn.opt.state[p] for p in learn.opt.param_lists[0]], [{}, {'do_wd': False}])
# Second param groups has state
for p in learn.opt.param_lists[1]:
    assert 'grad_avg' in learn.opt.state[p]


# export
@patch
def to_non_native_fp32(self: Learner): return self.remove_cbs([ModelToHalf, NonNativeMixedPrecision])


# cuda
learn = learn.to_non_native_fp32()

# ## Export -

# hide
notebook2script()
