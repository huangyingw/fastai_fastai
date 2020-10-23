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
from fastai.test_utils import *
import scipy.stats as scs
import sklearn.metrics as skm
from nbdev.showdoc import *
from fastai.learner import *
from fastai.optimizer import *
from fastai.data.all import *
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# export

# hide

# +
# default_exp metrics
# default_cls_lvl 3
# -

# # Metrics
#
# > Definition of the metrics that can be used in training models

# ## Core metric

# This is where the function that converts scikit-learn metrics to fastai metrics is defined. You should skip this section unless you want to know all about the internals of fastai.

# export

# export


# export torch_core
def flatten_check(inp, targ):
    "Check that `out` and `targ` have the same number of elements and flatten them."
    inp, targ = inp.contiguous().view(-1), targ.contiguous().view(-1)
    test_eq(len(inp), len(targ))
    return inp, targ


x1, x2 = torch.randn(5, 4), torch.randn(20)
x1, x2 = flatten_check(x1, x2)
test_eq(x1.shape, [20])
test_eq(x2.shape, [20])
x1, x2 = torch.randn(5, 4), torch.randn(21)
test_fail(lambda: flatten_check(x1, x2))

# export
mk_class('ActivationType', **{o: o.lower() for o in ['No', 'Sigmoid', 'Softmax', 'BinarySoftmax']},
         doc="All possible activation classes for `AccumMetric")


# export
class AccumMetric(Metric):
    "Stores predictions and targets on CPU in accumulate to perform final calculations with `func`."

    def __init__(self, func, dim_argmax=None, activation=ActivationType.No, thresh=None, to_np=False,
                 invert_arg=False, flatten=True, **kwargs):
        store_attr('func,dim_argmax,activation,thresh,flatten')
        self.to_np, self.invert_args, self.kwargs = to_np, invert_arg, kwargs

    def reset(self):
        "Clear all targs and preds"
        self.targs, self.preds = [], []

    def accumulate(self, learn):
        "Store targs and preds from `learn`, using activation function and argmax as appropriate"
        pred = learn.pred
        if self.activation in [ActivationType.Softmax, ActivationType.BinarySoftmax]:
            pred = F.softmax(pred, dim=self.dim_argmax)
            if self.activation == ActivationType.BinarySoftmax:
                pred = pred[:, -1]
        elif self.activation == ActivationType.Sigmoid:
            pred = torch.sigmoid(pred)
        elif self.dim_argmax:
            pred = pred.argmax(dim=self.dim_argmax)
        if self.thresh:
            pred = (pred >= self.thresh)
        self.accum_values(pred, learn.y, learn)

    def accum_values(self, preds, targs, learn=None):
        "Store targs and preds"
        to_d = learn.to_detach if learn is not None else to_detach
        preds, targs = to_d(preds), to_d(targs)
        if self.flatten:
            preds, targs = flatten_check(preds, targs)
        self.preds.append(preds)
        self.targs.append(targs)

    def __call__(self, preds, targs):
        "Calculate metric on one batch of data"
        self.reset()
        self.accum_values(preds, targs)
        return self.value

    @property
    def value(self):
        "Value of the metric using accumulated preds and targs"
        if len(self.preds) == 0:
            return
        preds, targs = torch.cat(self.preds), torch.cat(self.targs)
        if self.to_np:
            preds, targs = preds.numpy(), targs.numpy()
        return self.func(targs, preds, **self.kwargs) if self.invert_args else self.func(preds, targs, **self.kwargs)

    @property
    def name(self): return self.func.func.__name__ if hasattr(self.func, 'func') else self.func.__name__


# `func` is only applied to the accumulated predictions/targets when the `value` attribute is asked for (so at the end of a validation/training phase, in use with `Learner` and its `Recorder`).The signature of `func` should be `inp,targ` (where `inp` are the predictions of the model and `targ` the corresponding labels).
#
# For classification problems with single label, predictions need to be transformed with a softmax then an argmax before being compared to the targets. Since a softmax doesn't change the order of the numbers, we can just apply the argmax. Pass along `dim_argmax` to have this done by `AccumMetric` (usually -1 will work pretty well). If you need to pass to your metrics the probabilities and not the predictions, use `softmax=True`.
#
# For classification problems with multiple labels, or if your targets are one-hot encoded, predictions may need to pass through a sigmoid (if it wasn't included in your model) then be compared to a given threshold (to decide between 0 and 1), this is done by `AccumMetric` if you pass `sigmoid=True` and/or a value for `thresh`.
#
# If you want to use a metric function sklearn.metrics, you will need to convert predictions and labels to numpy arrays with `to_np=True`. Also, scikit-learn metrics adopt the convention `y_true`, `y_preds` which is the opposite from us, so you will need to pass `invert_arg=True` to make `AccumMetric` do the inversion for you.

# For testing: a fake learner and a metric that isn't an average
@delegates()
class TstLearner(Learner):
    def __init__(self, dls=None, model=None, **kwargs): self.pred, self.xb, self.yb = None, None, None


# +
def _l2_mean(x, y): return torch.sqrt((x.float() - y.float()).pow(2).mean())

# Go through a fake cycle with various batch sizes and computes the value of met


def compute_val(met, x1, x2):
    met.reset()
    vals = [0, 6, 15, 20]
    learn = TstLearner()
    for i in range(3):
        learn.pred, learn.yb = x1[vals[i]:vals[i + 1]], (x2[vals[i]:vals[i + 1]],)
        met.accumulate(learn)
    return met.value


# +
x1, x2 = torch.randn(20, 5), torch.randn(20, 5)
tst = AccumMetric(_l2_mean)
test_close(compute_val(tst, x1, x2), _l2_mean(x1, x2))
test_eq(torch.cat(tst.preds), x1.view(-1))
test_eq(torch.cat(tst.targs), x2.view(-1))

# test argmax
x1, x2 = torch.randn(20, 5), torch.randint(0, 5, (20,))
tst = AccumMetric(_l2_mean, dim_argmax=-1)
test_close(compute_val(tst, x1, x2), _l2_mean(x1.argmax(dim=-1), x2))

# test thresh
x1, x2 = torch.randn(20, 5), torch.randint(0, 2, (20, 5)).bool()
tst = AccumMetric(_l2_mean, thresh=0.5)
test_close(compute_val(tst, x1, x2), _l2_mean((x1 >= 0.5), x2))

# test sigmoid
x1, x2 = torch.randn(20, 5), torch.randn(20, 5)
tst = AccumMetric(_l2_mean, activation=ActivationType.Sigmoid)
test_close(compute_val(tst, x1, x2), _l2_mean(torch.sigmoid(x1), x2))

# test to_np
x1, x2 = torch.randn(20, 5), torch.randn(20, 5)
tst = AccumMetric(lambda x, y: isinstance(x, np.ndarray) and isinstance(y, np.ndarray), to_np=True)
assert compute_val(tst, x1, x2)

# test invert_arg
x1, x2 = torch.randn(20, 5), torch.randn(20, 5)
tst = AccumMetric(lambda x, y: torch.sqrt(x.pow(2).mean()))
test_close(compute_val(tst, x1, x2), torch.sqrt(x1.pow(2).mean()))
tst = AccumMetric(lambda x, y: torch.sqrt(x.pow(2).mean()), invert_arg=True)
test_close(compute_val(tst, x1, x2), torch.sqrt(x2.pow(2).mean()))


# -

# hide
def _l2_mean(x, y): return torch.sqrt((x.argmax(dim=-1).float() - y.float()).pow(2).mean())


x1, x2 = torch.randn(20, 5), torch.randint(0, 5, (20,))
tst = AccumMetric(_l2_mean, dim_argmax=-1, flatten=False, activation=ActivationType.Softmax)
test_close(compute_val(tst, x1, x2), _l2_mean(F.softmax(x1, dim=-1), x2))


# export
def skm_to_fastai(func, is_class=True, thresh=None, axis=-1, activation=None, **kwargs):
    "Convert `func` from sklearn.metrics to a fastai metric"
    dim_argmax = axis if is_class and thresh is None else None
    if activation is None:
        activation = ActivationType.Sigmoid if (is_class and thresh is not None) else ActivationType.No
    return AccumMetric(func, dim_argmax=dim_argmax, activation=activation, thresh=thresh,
                       to_np=True, invert_arg=True, **kwargs)


# This is the quickest way to use a scikit-learn metric in a fastai training loop. `is_class` indicates if you are in a classification problem or not. In this case:
# - leaving `thresh` to `None` indicates it's a single-label classification problem and predictions will pass through an argmax over `axis` before being compared to the targets
# - setting a value for `thresh` indicates it's a multi-label classification problem and predictions will pass through a sigmoid (can be deactivated with `sigmoid=False`) and be compared to `thresh` before being compared to the targets
#
# If `is_class=False`, it indicates you are in a regression problem, and predictions are compared to the targets without being modified. In all cases, `kwargs` are extra keyword arguments passed to `func`.

tst_single = skm_to_fastai(skm.precision_score)
x1, x2 = torch.randn(20, 2), torch.randint(0, 2, (20,))
test_close(compute_val(tst_single, x1, x2), skm.precision_score(x2, x1.argmax(dim=-1)))

# +
tst_multi = skm_to_fastai(skm.precision_score, thresh=0.2)
x1, x2 = torch.randn(20), torch.randint(0, 2, (20,))
test_close(compute_val(tst_multi, x1, x2), skm.precision_score(x2, torch.sigmoid(x1) >= 0.2))

tst_multi = skm_to_fastai(skm.precision_score, thresh=0.2, activation=ActivationType.No)
x1, x2 = torch.randn(20), torch.randint(0, 2, (20,))
test_close(compute_val(tst_multi, x1, x2), skm.precision_score(x2, x1 >= 0.2))
# -

tst_reg = skm_to_fastai(skm.r2_score, is_class=False)
x1, x2 = torch.randn(20, 5), torch.randn(20, 5)
test_close(compute_val(tst_reg, x1, x2), skm.r2_score(x2.view(-1), x1.view(-1)))

test_close(tst_reg(x1, x2), skm.r2_score(x2.view(-1), x1.view(-1)))


# export
def optim_metric(f, argname, bounds, tol=0.01, do_neg=True, get_x=False):
    "Replace metric `f` with a version that optimizes argument `argname`"
    def _f(preds, targs):
        def minfunc(x):
            kwargs = {argname: x}
            res = f(preds, targs, **kwargs)
            return -res if do_neg else res
        optres = scipy.optimize.minimize_scalar(minfunc, bounds=bounds, method='bounded',
                                                options={'xatol': 0.01})
        fun = -optres.fun if do_neg else optres.fun
        return (fun, optres.x) if get_x else fun
    _f.__name__ = f'opt_{f.__name__}'
    return _f


# ## Single-label classification

# > Warning: All functions defined in this section are intended for single-label classification and targets that are not one-hot encoded. For multi-label problems or one-hot encoded targets, use the version suffixed with multi.

# > Warning: Many metrics in fastai are thin wrappers around sklearn functionality. However, sklearn metrics can handle python list strings, amongst other things, whereas fastai metrics work with PyTorch, and thus require tensors. The arguments that are passed to metrics are after all transformations, such as categories being converted to indices, have occurred. This means that when you pass a label of a metric, for instance, that you must pass indices, not strings. This can be converted with `vocab.map_obj`.

# export
def accuracy(inp, targ, axis=-1):
    "Compute accuracy with `targ` when `pred` is bs * n_classes"
    pred, targ = flatten_check(inp.argmax(dim=axis), targ)
    return (pred == targ).float().mean()


# For testing
def change_targ(targ, n, c):
    idx = torch.randperm(len(targ))[:n]
    res = targ.clone()
    for i in idx:
        res[i] = (res[i] + random.randint(1, c - 1)) % c
    return res


x = torch.randn(4, 5)
y = x.argmax(dim=1)
test_eq(accuracy(x, y), 1)
y1 = change_targ(y, 2, 5)
test_eq(accuracy(x, y1), 0.5)
test_eq(accuracy(x.unsqueeze(1).expand(4, 2, 5), torch.stack([y, y1], dim=1)), 0.75)


# export
def error_rate(inp, targ, axis=-1):
    "1 - `accuracy`"
    return 1 - accuracy(inp, targ, axis=axis)


x = torch.randn(4, 5)
y = x.argmax(dim=1)
test_eq(error_rate(x, y), 0)
y1 = change_targ(y, 2, 5)
test_eq(error_rate(x, y1), 0.5)
test_eq(error_rate(x.unsqueeze(1).expand(4, 2, 5), torch.stack([y, y1], dim=1)), 0.25)


# export
def top_k_accuracy(inp, targ, k=5, axis=-1):
    "Computes the Top-k accuracy (`targ` is in the top `k` predictions of `inp`)"
    inp = inp.topk(k=k, dim=axis)[1]
    targ = targ.unsqueeze(dim=axis).expand_as(inp)
    return (inp == targ).sum(dim=-1).float().mean()


x = torch.randn(6, 5)
y = torch.arange(0, 6)
test_eq(top_k_accuracy(x[:5], y[:5]), 1)
test_eq(top_k_accuracy(x, y), 5 / 6)


# export
def APScoreBinary(axis=-1, average='macro', pos_label=1, sample_weight=None):
    "Average Precision for single-label binary classification problems"
    return skm_to_fastai(skm.average_precision_score, axis=axis, activation=ActivationType.BinarySoftmax,
                         average=average, pos_label=pos_label, sample_weight=sample_weight)


# See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score) for more details.

# export
def BalancedAccuracy(axis=-1, sample_weight=None, adjusted=False):
    "Balanced Accuracy for single-label binary classification problems"
    return skm_to_fastai(skm.balanced_accuracy_score, axis=axis,
                         sample_weight=sample_weight, adjusted=adjusted)


# See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html#sklearn.metrics.balanced_accuracy_score) for more details.

# export
def BrierScore(axis=-1, sample_weight=None, pos_label=None):
    "Brier score for single-label classification problems"
    return skm_to_fastai(skm.brier_score_loss, axis=axis,
                         sample_weight=sample_weight, pos_label=pos_label)


# See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html#sklearn.metrics.brier_score_loss) for more details.

# export
def CohenKappa(axis=-1, labels=None, weights=None, sample_weight=None):
    "Cohen kappa for single-label classification problems"
    return skm_to_fastai(skm.cohen_kappa_score, axis=axis, labels=labels, weights=weights,
                         sample_weight=sample_weight)


# See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html#sklearn.metrics.cohen_kappa_score) for more details.

# export
def F1Score(axis=-1, labels=None, pos_label=1, average='binary', sample_weight=None):
    "F1 score for single-label classification problems"
    return skm_to_fastai(skm.f1_score, axis=axis,
                         labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight)


# See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score) for more details.

# export
def FBeta(beta, axis=-1, labels=None, pos_label=1, average='binary', sample_weight=None):
    "FBeta score with `beta` for single-label classification problems"
    return skm_to_fastai(skm.fbeta_score, axis=axis,
                         beta=beta, labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight)


# See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html#sklearn.metrics.fbeta_score) for more details.

# export
def HammingLoss(axis=-1, sample_weight=None):
    "Hamming loss for single-label classification problems"
    return skm_to_fastai(skm.hamming_loss, axis=axis,
                         sample_weight=sample_weight)


# See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hamming_loss.html#sklearn.metrics.hamming_loss) for more details.

# export
def Jaccard(axis=-1, labels=None, pos_label=1, average='binary', sample_weight=None):
    "Jaccard score for single-label classification problems"
    return skm_to_fastai(skm.jaccard_score, axis=axis,
                         labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight)


# See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html#sklearn.metrics.jaccard_score) for more details.

# export
def Precision(axis=-1, labels=None, pos_label=1, average='binary', sample_weight=None):
    "Precision for single-label classification problems"
    return skm_to_fastai(skm.precision_score, axis=axis,
                         labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight)


# See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score) for more details.

# export
def Recall(axis=-1, labels=None, pos_label=1, average='binary', sample_weight=None):
    "Recall for single-label classification problems"
    return skm_to_fastai(skm.recall_score, axis=axis,
                         labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight)


# See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score) for more details.

# export
def RocAuc(axis=-1, average='macro', sample_weight=None, max_fpr=None, multi_class='ovr'):
    "Area Under the Receiver Operating Characteristic Curve for single-label multiclass classification problems"
    assert multi_class in ['ovr', 'ovo']
    return skm_to_fastai(skm.roc_auc_score, axis=axis, activation=ActivationType.Softmax, flatten=False,
                         average=average, sample_weight=sample_weight, max_fpr=max_fpr, multi_class=multi_class)


# See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score) for more details.

# export
def RocAucBinary(axis=-1, average='macro', sample_weight=None, max_fpr=None, multi_class='raise'):
    "Area Under the Receiver Operating Characteristic Curve for single-label binary classification problems"
    return skm_to_fastai(skm.roc_auc_score, axis=axis, activation=ActivationType.BinarySoftmax,
                         average=average, sample_weight=sample_weight, max_fpr=max_fpr, multi_class=multi_class)


# See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score) for more details.

# export
def MatthewsCorrCoef(sample_weight=None, **kwargs):
    "Matthews correlation coefficient for single-label classification problems"
    return skm_to_fastai(skm.matthews_corrcoef, sample_weight=sample_weight, **kwargs)


# See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html#sklearn.metrics.matthews_corrcoef) for more details.

# +
# export
class Perplexity(AvgLoss):
    "Perplexity (exponential of cross-entropy loss) for Language Models"
    @property
    def value(self): return torch.exp(self.total / self.count) if self.count != 0 else None
    @property
    def name(self): return "perplexity"


perplexity = Perplexity()
# -

x1, x2 = torch.randn(20, 5), torch.randint(0, 5, (20,))
tst = perplexity
tst.reset()
vals = [0, 6, 15, 20]
learn = TstLearner()
for i in range(3):
    learn.yb = (x2[vals[i]:vals[i + 1]],)
    learn.loss = F.cross_entropy(x1[vals[i]:vals[i + 1]], x2[vals[i]:vals[i + 1]])
    tst.accumulate(learn)
test_close(tst.value, torch.exp(F.cross_entropy(x1, x2)))


# ## Multi-label classification

# export
def accuracy_multi(inp, targ, thresh=0.5, sigmoid=True):
    "Compute accuracy when `inp` and `targ` are the same size."
    inp, targ = flatten_check(inp, targ)
    if sigmoid:
        inp = inp.sigmoid()
    return ((inp > thresh) == targ.bool()).float().mean()


# For testing
def change_1h_targ(targ, n):
    idx = torch.randperm(targ.numel())[:n]
    res = targ.clone().view(-1)
    for i in idx:
        res[i] = 1 - res[i]
    return res.view(targ.shape)


# +
x = torch.randn(4, 5)
y = (torch.sigmoid(x) >= 0.5).byte()
test_eq(accuracy_multi(x, y), 1)
test_eq(accuracy_multi(x, 1 - y), 0)
y1 = change_1h_targ(y, 5)
test_eq(accuracy_multi(x, y1), 0.75)

# Different thresh
y = (torch.sigmoid(x) >= 0.2).byte()
test_eq(accuracy_multi(x, y, thresh=0.2), 1)
test_eq(accuracy_multi(x, 1 - y, thresh=0.2), 0)
y1 = change_1h_targ(y, 5)
test_eq(accuracy_multi(x, y1, thresh=0.2), 0.75)

# No sigmoid
y = (x >= 0.5).byte()
test_eq(accuracy_multi(x, y, sigmoid=False), 1)
test_eq(accuracy_multi(x, 1 - y, sigmoid=False), 0)
y1 = change_1h_targ(y, 5)
test_eq(accuracy_multi(x, y1, sigmoid=False), 0.75)


# -

# export
def APScoreMulti(sigmoid=True, average='macro', pos_label=1, sample_weight=None):
    "Average Precision for multi-label classification problems"
    activation = ActivationType.Sigmoid if sigmoid else ActivationType.No
    return skm_to_fastai(skm.average_precision_score, activation=activation, flatten=False,
                         average=average, pos_label=pos_label, sample_weight=sample_weight)


# See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score) for more details.

# export
def BrierScoreMulti(thresh=0.5, sigmoid=True, sample_weight=None, pos_label=None):
    "Brier score for multi-label classification problems"
    activation = ActivationType.Sigmoid if sigmoid else ActivationType.No
    return skm_to_fastai(skm.brier_score_loss, thresh=thresh, activation=activation, flatten=False,
                         sample_weight=sample_weight, pos_label=pos_label)


# See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html#sklearn.metrics.brier_score_loss) for more details.

# export
def F1ScoreMulti(thresh=0.5, sigmoid=True, labels=None, pos_label=1, average='macro', sample_weight=None):
    "F1 score for multi-label classification problems"
    activation = ActivationType.Sigmoid if sigmoid else ActivationType.No
    return skm_to_fastai(skm.f1_score, thresh=thresh, activation=activation, flatten=False,
                         labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight)


# See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score) for more details.

# export
def FBetaMulti(beta, thresh=0.5, sigmoid=True, labels=None, pos_label=1, average='macro', sample_weight=None):
    "FBeta score with `beta` for multi-label classification problems"
    activation = ActivationType.Sigmoid if sigmoid else ActivationType.No
    return skm_to_fastai(skm.fbeta_score, thresh=thresh, activation=activation, flatten=False,
                         beta=beta, labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight)


# See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html#sklearn.metrics.fbeta_score) for more details.

# export
def HammingLossMulti(thresh=0.5, sigmoid=True, labels=None, sample_weight=None):
    "Hamming loss for multi-label classification problems"
    activation = ActivationType.Sigmoid if sigmoid else ActivationType.No
    return skm_to_fastai(skm.hamming_loss, thresh=thresh, activation=activation, flatten=False,
                         sample_weight=sample_weight)


# See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.hamming_loss.html#sklearn.metrics.hamming_loss) for more details.

# export
def JaccardMulti(thresh=0.5, sigmoid=True, labels=None, pos_label=1, average='macro', sample_weight=None):
    "Jaccard score for multi-label classification problems"
    activation = ActivationType.Sigmoid if sigmoid else ActivationType.No
    return skm_to_fastai(skm.jaccard_score, thresh=thresh, activation=activation, flatten=False,
                         labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight)


# See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html#sklearn.metrics.jaccard_score) for more details.

# export
def MatthewsCorrCoefMulti(thresh=0.5, sigmoid=True, sample_weight=None):
    "Matthews correlation coefficient for multi-label classification problems"
    activation = ActivationType.Sigmoid if sigmoid else ActivationType.No
    return skm_to_fastai(skm.matthews_corrcoef, thresh=thresh, activation=activation, flatten=False, sample_weight=sample_weight)


# See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html#sklearn.metrics.matthews_corrcoef) for more details.

# export
def PrecisionMulti(thresh=0.5, sigmoid=True, labels=None, pos_label=1, average='macro', sample_weight=None):
    "Precision for multi-label classification problems"
    activation = ActivationType.Sigmoid if sigmoid else ActivationType.No
    return skm_to_fastai(skm.precision_score, thresh=thresh, activation=activation, flatten=False,
                         labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight)


# See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score) for more details.

# export
def RecallMulti(thresh=0.5, sigmoid=True, labels=None, pos_label=1, average='macro', sample_weight=None):
    "Recall for multi-label classification problems"
    activation = ActivationType.Sigmoid if sigmoid else ActivationType.No
    return skm_to_fastai(skm.recall_score, thresh=thresh, activation=activation, flatten=False,
                         labels=labels, pos_label=pos_label, average=average, sample_weight=sample_weight)


# See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score) for more details.

# export
def RocAucMulti(sigmoid=True, average='macro', sample_weight=None, max_fpr=None):
    "Area Under the Receiver Operating Characteristic Curve for multi-label binary classification problems"
    activation = ActivationType.Sigmoid if sigmoid else ActivationType.No
    return skm_to_fastai(skm.roc_auc_score, activation=activation, flatten=False,
                         average=average, sample_weight=sample_weight, max_fpr=max_fpr)


roc_auc_metric = RocAucMulti(sigmoid=False)
x, y = torch.tensor([np.arange(start=0, stop=0.2, step=0.04)] * 20), torch.tensor([0, 0, 1, 1]).repeat(5)
assert compute_val(roc_auc_metric, x, y) == 0.5


# See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score) for more details.

# ## Regression

# export
def mse(inp, targ):
    "Mean squared error between `inp` and `targ`."
    return F.mse_loss(*flatten_check(inp, targ))


x1, x2 = torch.randn(4, 5), torch.randn(4, 5)
test_close(mse(x1, x2), (x1 - x2).pow(2).mean())


# export
def _rmse(inp, targ): return torch.sqrt(F.mse_loss(inp, targ))


rmse = AccumMetric(_rmse)
rmse.__doc__ = "Root mean squared error"

show_doc(rmse, name="rmse")

x1, x2 = torch.randn(20, 5), torch.randn(20, 5)
test_eq(compute_val(rmse, x1, x2), torch.sqrt(F.mse_loss(x1, x2)))


# export
def mae(inp, targ):
    "Mean absolute error between `inp` and `targ`."
    inp, targ = flatten_check(inp, targ)
    return torch.abs(inp - targ).mean()


x1, x2 = torch.randn(4, 5), torch.randn(4, 5)
test_eq(mae(x1, x2), torch.abs(x1 - x2).mean())


# export
def msle(inp, targ):
    "Mean squared logarithmic error between `inp` and `targ`."
    inp, targ = flatten_check(inp, targ)
    return F.mse_loss(torch.log(1 + inp), torch.log(1 + targ))


x1, x2 = torch.randn(4, 5), torch.randn(4, 5)
x1, x2 = torch.relu(x1), torch.relu(x2)
test_close(msle(x1, x2), (torch.log(x1 + 1) - torch.log(x2 + 1)).pow(2).mean())


# export
def _exp_rmspe(inp, targ):
    inp, targ = torch.exp(inp), torch.exp(targ)
    return torch.sqrt(((targ - inp) / targ).pow(2).mean())


exp_rmspe = AccumMetric(_exp_rmspe)
exp_rmspe.__doc__ = "Root mean square percentage error of the exponential of  predictions and targets"

show_doc(exp_rmspe, name="exp_rmspe")

x1, x2 = torch.randn(20, 5), torch.randn(20, 5)
test_eq(compute_val(exp_rmspe, x1, x2), torch.sqrt((((torch.exp(x2) - torch.exp(x1)) / torch.exp(x2))**2).mean()))


# export
def ExplainedVariance(sample_weight=None):
    "Explained variance between predictions and targets"
    return skm_to_fastai(skm.explained_variance_score, is_class=False, sample_weight=sample_weight)


# See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.explained_variance_score.html#sklearn.metrics.explained_variance_score) for more details.

# export
def R2Score(sample_weight=None):
    "R2 score between predictions and targets"
    return skm_to_fastai(skm.r2_score, is_class=False, sample_weight=sample_weight)


# See the [scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score) for more details.

# export
@delegates(AccumMetric)
def PearsonCorrCoef(dim_argmax=None, **kwargs):
    "Pearson correlation coefficient for regression problem"
    def pearsonr(x, y): return scs.pearsonr(x, y)[0]
    return AccumMetric(pearsonr, invert_arg=False, dim_argmax=dim_argmax, **kwargs)


# See the [scipy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html?highlight=pearson#scipy.stats.pearsonr) for more details.

x = torch.randint(-999, 999, (20,))
y = torch.randint(-999, 999, (20,))
test_eq(compute_val(PearsonCorrCoef(), x, y), scs.pearsonr(x.view(-1), y.view(-1))[0])


# export
@delegates(AccumMetric)
def SpearmanCorrCoef(dim_argmax=None, axis=0, nan_policy='propagate', **kwargs):
    "Spearman correlation coefficient for regression problem"
    def spearmanr(a, b=None, **kwargs): return scs.spearmanr(a, b, **kwargs)[0]
    return AccumMetric(partial(spearmanr, axis=axis, nan_policy=nan_policy),
                       invert_arg=False, dim_argmax=dim_argmax, **kwargs)


# See the [scipy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html?highlight=spearman#scipy.stats.spearmanr) for more details.

x = torch.randint(-999, 999, (20,))
y = torch.randint(-999, 999, (20,))
test_eq(compute_val(SpearmanCorrCoef(), x, y), scs.spearmanr(x.view(-1), y.view(-1))[0])


# ## Segmentation

# export
def foreground_acc(inp, targ, bkg_idx=0, axis=1):
    "Computes non-background accuracy for multiclass segmentation"
    targ = targ.squeeze(1)
    mask = targ != bkg_idx
    return (inp.argmax(dim=axis)[mask] == targ[mask]).float().mean()


x = torch.randn(4, 5, 3, 3)
y = x.argmax(dim=1)[:, None]
test_eq(foreground_acc(x, y), 1)
y[0] = 0  # the 0s are ignored so we get the same value
test_eq(foreground_acc(x, y), 1)


# export
class Dice(Metric):
    "Dice coefficient metric for binary target in segmentation"

    def __init__(self, axis=1): self.axis = axis
    def reset(self): self.inter, self.union = 0, 0

    def accumulate(self, learn):
        pred, targ = flatten_check(learn.pred.argmax(dim=self.axis), learn.y)
        self.inter += (pred * targ).float().sum().item()
        self.union += (pred + targ).float().sum().item()

    @property
    def value(self): return 2. * self.inter / self.union if self.union > 0 else None


x1 = torch.randn(20, 2, 3, 3)
x2 = torch.randint(0, 2, (20, 3, 3))
pred = x1.argmax(1)
inter = (pred * x2).float().sum().item()
union = (pred + x2).float().sum().item()
test_eq(compute_val(Dice(), x1, x2), 2 * inter / union)


# export
class JaccardCoeff(Dice):
    "Implementation of the Jaccard coefficient that is lighter in RAM"
    @property
    def value(self): return self.inter / (self.union - self.inter) if self.union > 0 else None


x1 = torch.randn(20, 2, 3, 3)
x2 = torch.randint(0, 2, (20, 3, 3))
pred = x1.argmax(1)
inter = (pred * x2).float().sum().item()
union = (pred + x2).float().sum().item()
test_eq(compute_val(JaccardCoeff(), x1, x2), inter / (union - inter))


# ## NLP

# export
class CorpusBLEUMetric(Metric):
    def __init__(self, vocab_sz=5000, axis=-1):
        "BLEU Metric calculated over the validation corpus"
        self.metric_name = 'CorpusBLEU'
        self.axis, self.vocab_sz = axis, vocab_sz
        self.pred_len, self.targ_len, self.samp_idx, self.corrects, self.counts, = 0, 0, 0, [0] * 4, [0] * 4

    def reset(self):
        self.pred_len, self.targ_len, self.corrects, self.counts = 0, 0, [0] * 4, [0] * 4

    class NGram():
        def __init__(self, ngram, max_n=5000): self.ngram, self.max_n = ngram, max_n

        def __eq__(self, other):
            if len(self.ngram) != len(other.ngram):
                return False
            return np.all(np.array(self.ngram) == np.array(other.ngram))

        def __hash__(self): return int(sum([o * self.max_n**i for i, o in enumerate(self.ngram)]))

    def get_grams(self, x, n, max_n=5000):
        return x if n == 1 else [self.NGram(x[i:i + n], max_n=max_n) for i in range(len(x) - n + 1)]

    def get_correct_ngrams(self, pred, targ, n, max_n=5000):
        pred_grams, targ_grams = self.get_grams(pred, n, max_n=max_n), self.get_grams(targ, n, max_n=max_n)
        pred_cnt, targ_cnt = Counter(pred_grams), Counter(targ_grams)
        return sum([min(c, targ_cnt[g]) for g, c in pred_cnt.items()]), len(pred_grams)

    def accumulate(self, learn):
        if learn.training:
            return None
        else:
            last_output = learn.pred.argmax(dim=self.axis)
            last_target = learn.y
            for pred, targ in zip(last_output.cpu().numpy(), last_target.cpu().numpy()):
                self.pred_len += len(pred)
                self.targ_len += len(targ)
                smooth_mteval = 1
                for i in range(4):
                    c, t = self.get_correct_ngrams(pred, targ, i + 1, max_n=self.vocab_sz)
                    if c == 0:
                        smooth_mteval *= 2
                        c = 1 / smooth_mteval    # exp smoothing, method 3 from http://acl2014.org/acl2014/W14-33/pdf/W14-3346.pdf
                    self.corrects[i] += c
                    self.counts[i] += t

    @property
    def value(self):
        if self.counts == 0:
            return None
        elif max(self.corrects) == 0:
            return 0.0
        else:
            precs = [c / t for c, t in zip(self.corrects, self.counts)]
            len_penalty = math.exp(1 - self.targ_len / self.pred_len) if self.pred_len < self.targ_len else 1
            return len_penalty * ((precs[0] * precs[1] * precs[2] * precs[3]) ** 0.25)


# +
def create_vcb_emb(pred, targ):
    # create vocab "embedding" for predictions
    vcb_sz = max(torch.unique(torch.cat([pred, targ]))) + 1
    pred_emb = torch.zeros(pred.size()[0], pred.size()[1], vcb_sz)
    for i, v in enumerate(pred):
        pred_emb[i].scatter_(1, v.view(len(v), 1), 1)
    return pred_emb


def compute_bleu_val(met, x1, x2):
    met.reset()
    learn = TstLearner()
    learn.training = False
    for i in range(len(x1)):
        learn.pred, learn.yb = x1, (x2,)
        met.accumulate(learn)
    return met.value


targ = torch.tensor([[1, 2, 3, 4, 5, 6, 1, 7, 8]])
pred = torch.tensor([[1, 9, 3, 4, 5, 6, 1, 10, 8]])
pred_emb = create_vcb_emb(pred, targ)
test_close(compute_bleu_val(CorpusBLEUMetric(), pred_emb, targ), 0.48549)

targ = torch.tensor([[1, 2, 3, 4, 5, 6, 1, 7, 8], [1, 2, 3, 4, 5, 6, 1, 7, 8]])
pred = torch.tensor([[1, 9, 3, 4, 5, 6, 1, 10, 8], [1, 9, 3, 4, 5, 6, 1, 10, 8]])
pred_emb = create_vcb_emb(pred, targ)
test_close(compute_bleu_val(CorpusBLEUMetric(), pred_emb, targ), 0.48549)


# -

# The BLEU metric was introduced in [this article](https://www.aclweb.org/anthology/P02-1040) to come up with a way to evaluate the performance of translation models. It's based on the precision of n-grams in your prediction compared to your target. See the [fastai NLP course BLEU notebook](https://github.com/fastai/course-nlp/blob/master/bleu_metric.ipynb) for a more detailed description of BLEU.
#
# The smoothing used in the precision calculation is the same as in [SacreBLEU](https://github.com/mjpost/sacrebleu/blob/32c54cdd0dfd6a9fadd5805f2ea189ac0df63907/sacrebleu/sacrebleu.py#L540-L542), which in turn is "method 3" from the [Chen & Cherry, 2014](http://acl2014.org/acl2014/W14-33/pdf/W14-3346.pdf) paper.

# ## LossMetrics -

# export
class LossMetric(AvgMetric):
    "Create a metric from `loss_func.attr` named `nm`"

    def __init__(self, attr, nm=None): store_attr('attr,nm')

    def accumulate(self, learn):
        bs = find_bs(learn.yb)
        self.total += learn.to_detach(getattr(learn.loss_func, self.attr, 0)) * bs
        self.count += bs

    @property
    def name(self): return self.attr if self.nm is None else self.nm


# export
def LossMetrics(attrs, nms=None):
    "List of `LossMetric` for each of `attrs` and `nms`"
    if isinstance(attrs, str):
        attrs = attrs.split(',')
    nms = attrs if nms is None else nms.split(',') if isinstance(nms, str) else nms
    return [LossMetric(a, n) for a, n in zip(attrs, nms)]


# hide


class CombineL1L2(Module):
    def forward(self, out, targ):
        self.l1 = F.l1_loss(out, targ)
        self.l2 = F.mse_loss(out, targ)
        return self.l1 + self.l2


learn = synth_learner(metrics=LossMetrics('l1,l2'))
learn.loss_func = CombineL1L2()
learn.fit(2)

# ## Export -

# hide
notebook2script()
