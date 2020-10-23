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
from fastai.callback.hook import num_features_model
from nbdev.showdoc import *
from fastai.vision import models
from fastai.vision.augment import *
from fastai.vision.data import *
from fastai.vision.core import *
from fastai.basics import *
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# export

# +
# default_exp vision.learner
# -

# hide


# # Learner for the vision applications
#
# > All the functions necessary to build `Learner` suitable for transfer learning in computer vision

# The most important functions of this module are `cnn_learner` and `unet_learner`. They will help you define a `Learner` using a pretrained model. See the [vision tutorial](http://docs.fast.ai/tutorial.vision) for examples of use.

# ## Cut a pretrained model

# export
def _is_pool_type(l): return re.search(r'Pool[123]d$', l.__class__.__name__)


# hide
m = nn.Sequential(nn.AdaptiveAvgPool2d(5), nn.Linear(2, 3), nn.Conv2d(2, 3, 1), nn.MaxPool3d(5))
test_eq([bool(_is_pool_type(m_)) for m_ in m.children()], [True, False, False, True])


# By default, the fastai library cuts a pretrained model at the pooling layer. This function helps detecting it.

# export
def has_pool_type(m):
    "Return `True` if `m` is a pooling layer or has one in its children"
    if _is_pool_type(m):
        return True
    for l in m.children():
        if has_pool_type(l):
            return True
    return False


m = nn.Sequential(nn.AdaptiveAvgPool2d(5), nn.Linear(2, 3), nn.Conv2d(2, 3, 1), nn.MaxPool3d(5))
assert has_pool_type(m)
test_eq([has_pool_type(m_) for m_ in m.children()], [True, False, False, True])


# export
def _get_first_layer(m):
    "Access first layer of a model"
    c, p, n = m, None, None  # child, parent, name
    for n in next(m.named_parameters())[0].split('.')[:-1]:
        p, c = c, getattr(c, n)
    return c, p, n


# export
def _load_pretrained_weights(new_layer, previous_layer):
    "Load pretrained weights based on number of input channels"
    n_in = getattr(new_layer, 'in_channels')
    if n_in == 1:
        # we take the sum
        new_layer.weight.data = previous_layer.weight.data.sum(dim=1, keepdim=True)
    elif n_in == 2:
        # we take first 2 channels + 50%
        new_layer.weight.data = previous_layer.weight.data[:, :2] * 1.5
    else:
        # keep 3 channels weights and set others to null
        new_layer.weight.data[:, :3] = previous_layer.weight.data
        new_layer.weight.data[:, 3:].zero_()


# export
def _update_first_layer(model, n_in, pretrained):
    "Change first layer based on number of input channels"
    if n_in == 3:
        return
    first_layer, parent, name = _get_first_layer(model)
    assert isinstance(first_layer, nn.Conv2d), f'Change of input channels only supported with Conv2d, found {first_layer.__class__.__name__}'
    assert getattr(first_layer, 'in_channels') == 3, f'Unexpected number of input channels, found {getattr(first_layer, "in_channels")} while expecting 3'
    params = {attr: getattr(first_layer, attr) for attr in 'out_channels kernel_size stride padding dilation groups padding_mode'.split()}
    params['bias'] = getattr(first_layer, 'bias') is not None
    params['in_channels'] = n_in
    new_layer = nn.Conv2d(**params)
    if pretrained:
        _load_pretrained_weights(new_layer, first_layer)
    setattr(parent, name, new_layer)


# export
def create_body(arch, n_in=3, pretrained=True, cut=None):
    "Cut off the body of a typically pretrained `arch` as determined by `cut`"
    model = arch(pretrained=pretrained)
    _update_first_layer(model, n_in, pretrained)
    #cut = ifnone(cut, cnn_config(arch)['cut'])
    if cut is None:
        ll = list(enumerate(model.children()))
        cut = next(i for i, o in reversed(ll) if has_pool_type(o))
    if isinstance(cut, int):
        return nn.Sequential(*list(model.children())[:cut])
    elif callable(cut):
        return cut(model)
    else:
        raise NamedError("cut must be either integer or a function")


# `cut` can either be an integer, in which case we cut the model at the corresponding layer, or a function, in which case, this function returns `cut(model)`. It defaults to the first layer that contains some pooling otherwise.

# +
def tst(pretrained): return nn.Sequential(nn.Conv2d(3, 5, 3), nn.BatchNorm2d(5), nn.AvgPool2d(1), nn.Linear(3, 4))


m = create_body(tst)
test_eq(len(m), 2)

m = create_body(tst, cut=3)
test_eq(len(m), 3)

m = create_body(tst, cut=noop)
test_eq(len(m), 4)

for n in range(1, 5):
    m = create_body(tst, n_in=n)
    test_eq(_get_first_layer(m)[0].in_channels, n)


# -

# ## Head and model

# export
def create_head(nf, n_out, lin_ftrs=None, ps=0.5, concat_pool=True, bn_final=False, lin_first=False, y_range=None):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and out `n_out` classes."
    lin_ftrs = [nf, 512, n_out] if lin_ftrs is None else [nf] + lin_ftrs + [n_out]
    ps = L(ps)
    if len(ps) == 1:
        ps = [ps[0] / 2] * (len(lin_ftrs) - 2) + ps
    actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs) - 2) + [None]
    pool = AdaptiveConcatPool2d() if concat_pool else nn.AdaptiveAvgPool2d(1)
    layers = [pool, Flatten()]
    if lin_first:
        layers.append(nn.Dropout(ps.pop(0)))
    for ni, no, p, actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):
        layers += LinBnDrop(ni, no, bn=True, p=p, act=actn, lin_first=lin_first)
    if lin_first:
        layers.append(nn.Linear(lin_ftrs[-2], n_out))
    if bn_final:
        layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    if y_range is not None:
        layers.append(SigmoidRange(*y_range))
    return nn.Sequential(*layers)


# The head begins with fastai's `AdaptiveConcatPool2d` if `concat_pool=True` otherwise, it uses traditional average pooling. Then it uses a `Flatten` layer before going on blocks of `BatchNorm`, `Dropout` and `Linear` layers (if `lin_first=True`, those are `Linear`, `BatchNorm`, `Dropout`).
#
# Those blocks start at `nf`, then every element of `lin_ftrs` (defaults to `[512]`) and end at `n_out`. `ps` is a list of probabilities used for the dropouts (if you only pass 1, it will use half the value then that value as many times as necessary).
#
# If `bn_final=True`, a final `BatchNorm` layer is added. If `y_range` is passed, the function adds a `SigmoidRange` to that range.

tst = create_head(5, 10)
tst

# +
# hide
mods = list(tst.children())
test_eq(len(mods), 9)
assert isinstance(mods[2], nn.BatchNorm1d)
assert isinstance(mods[-1], nn.Linear)

tst = create_head(5, 10, lin_first=True)
mods = list(tst.children())
test_eq(len(mods), 8)
assert isinstance(mods[2], nn.Dropout)
# -

# export


# export
@delegates(create_head)
def create_cnn_model(arch, n_out, cut=None, pretrained=True, n_in=3, init=nn.init.kaiming_normal_, custom_head=None,
                     concat_pool=True, **kwargs):
    "Create custom convnet architecture using `arch`, `n_in` and `n_out`"
    body = create_body(arch, n_in, pretrained, cut)
    if custom_head is None:
        nf = num_features_model(nn.Sequential(*body.children())) * (2 if concat_pool else 1)
        head = create_head(nf, n_out, concat_pool=concat_pool, **kwargs)
    else:
        head = custom_head
    model = nn.Sequential(body, head)
    if init is not None:
        apply_init(model[1], init)
    return model


show_doc(create_cnn_model)

# The model is cut according to `cut` and it may be `pretrained`, in which case, the proper set of weights is downloaded then loaded. `init` is applied to the head of the model, which is either created by `create_head` (with `lin_ftrs`, `ps`, `concat_pool`, `bn_final`, `lin_first` and `y_range`) or is `custom_head`.

tst = create_cnn_model(models.resnet18, 10, None, True)
tst = create_cnn_model(models.resnet18, 10, None, True, n_in=1)


# export
@delegates(create_cnn_model)
def cnn_config(**kwargs):
    "Convenience function to easily create a config for `create_cnn_model`"
    return kwargs


# +
pets = DataBlock(blocks=(ImageBlock, CategoryBlock),
                 get_items=get_image_files,
                 splitter=RandomSplitter(),
                 get_y=RegexLabeller(pat=r'/([^/]+)_\d+.jpg$'))

dls = pets.dataloaders(untar_data(URLs.PETS) / "images", item_tfms=RandomResizedCrop(300, min_scale=0.5), bs=64,
                       batch_tfms=[*aug_transforms(size=224)])


# +
# TODO: refactor, i.e. something like this?
# class ModelSplitter():
#     def __init__(self, idx): self.idx = idx
#     def split(self, m): return L(m[:self.idx], m[self.idx:]).map(params)
#     def __call__(self,): return {'cut':self.idx, 'split':self.split}
# -

# export
def default_split(m):
    "Default split of a model between body and head"
    return L(m[0], m[1:]).map(params)


# To do transfer learning, you need to pass a `splitter` to `Learner`. This should be a function taking the model and returning a collection of parameter groups, e.g. a list of list of parameters.

# +
# export
def _xresnet_split(m): return L(m[0][:3], m[0][3:], m[1:]).map(params)


def _resnet_split(m): return L(m[0][:6], m[0][6:], m[1:]).map(params)


def _squeezenet_split(m: nn.Module): return L(m[0][0][:5], m[0][0][5:], m[1:]).map(params)


def _densenet_split(m: nn.Module): return L(m[0][0][:7], m[0][0][7:], m[1:]).map(params)


def _vgg_split(m: nn.Module): return L(m[0][0][:22], m[0][0][22:], m[1:]).map(params)


def _alexnet_split(m: nn.Module): return L(m[0][0][:6], m[0][0][6:], m[1:]).map(params)


_default_meta = {'cut': None, 'split': default_split}
_xresnet_meta = {'cut': -4, 'split': _xresnet_split, 'stats': imagenet_stats}
_resnet_meta = {'cut': -2, 'split': _resnet_split, 'stats': imagenet_stats}
_squeezenet_meta = {'cut': -1, 'split': _squeezenet_split, 'stats': imagenet_stats}
_densenet_meta = {'cut': -1, 'split': _densenet_split, 'stats': imagenet_stats}
_vgg_meta = {'cut': -2, 'split': _vgg_split, 'stats': imagenet_stats}
_alexnet_meta = {'cut': -2, 'split': _alexnet_split, 'stats': imagenet_stats}
# -

# export
model_meta = {
    models.xresnet.xresnet18: {**_xresnet_meta}, models.xresnet.xresnet34: {**_xresnet_meta},
    models.xresnet.xresnet50: {**_xresnet_meta}, models.xresnet.xresnet101: {**_xresnet_meta},
    models.xresnet.xresnet152: {**_xresnet_meta},

    models.resnet18: {**_resnet_meta}, models.resnet34: {**_resnet_meta},
    models.resnet50: {**_resnet_meta}, models.resnet101: {**_resnet_meta},
    models.resnet152: {**_resnet_meta},

    models.squeezenet1_0: {**_squeezenet_meta},
    models.squeezenet1_1: {**_squeezenet_meta},

    models.densenet121: {**_densenet_meta}, models.densenet169: {**_densenet_meta},
    models.densenet201: {**_densenet_meta}, models.densenet161: {**_densenet_meta},
    models.vgg11_bn: {**_vgg_meta}, models.vgg13_bn: {**_vgg_meta}, models.vgg16_bn: {**_vgg_meta}, models.vgg19_bn: {**_vgg_meta},
    models.alexnet: {**_alexnet_meta}}


# ## `Learner` convenience functions

# export
def _add_norm(dls, meta, pretrained):
    if not pretrained:
        return
    after_batch = dls.after_batch
    if first(o for o in after_batch.fs if isinstance(o, Normalize)):
        return
    stats = meta.get('stats')
    if stats is None:
        return
    after_batch.add(Normalize.from_stats(*stats))


# export
@log_args(to_return=True, but_as=Learner.__init__)
@delegates(Learner.__init__)
def cnn_learner(dls, arch, loss_func=None, pretrained=True, cut=None, splitter=None,
                y_range=None, config=None, n_out=None, normalize=True, **kwargs):
    "Build a convnet style learner from `dls` and `arch`"
    if config is None:
        config = {}
    meta = model_meta.get(arch, _default_meta)
    if n_out is None:
        n_out = get_c(dls)
    assert n_out, "`n_out` is not defined, and could not be inferred from data, set `dls.c` or pass `n_out`"
    if normalize:
        _add_norm(dls, meta, pretrained)
    if y_range is None and 'y_range' in config:
        y_range = config.pop('y_range')
    model = create_cnn_model(arch, n_out, ifnone(cut, meta['cut']), pretrained, y_range=y_range, **config)
    learn = Learner(dls, model, loss_func=loss_func, splitter=ifnone(splitter, meta['split']), **kwargs)
    if pretrained:
        learn.freeze()
    return learn


# The model is built from `arch` using the number of final activations inferred from `dls` if possible (otherwise pass a value to `n_out`). It might be `pretrained` and the architecture is cut and split using the default metadata of the model architecture (this can be customized by passing a `cut` or a `splitter`).
#
# To customize the model creation, use `cnn_config` and pass the result to the `config` argument. There is just easy access to `y_range` because this argument is often used.
#
# If `normalize` and `pretrained` are `True`, this function adds a `Normalization` transform to the `dls` (if there is not already one) using the statistics of the pretrained model. That way, you won't ever forget to normalize your data in transfer learning.
#
# All other arguments are passed to `Learner`.

path = untar_data(URLs.PETS)
fnames = get_image_files(path / "images")
pat = r'^(.*)_\d+.jpg$'
dls = ImageDataLoaders.from_name_re(path, fnames, pat, item_tfms=Resize(224))

learn = cnn_learner(dls, models.resnet34, loss_func=CrossEntropyLossFlat(), config=cnn_config(ps=0.25))

# hide
test_eq(to_cpu(dls.after_batch[1].mean[0].squeeze()), tensor(imagenet_stats[0]))


# export
@delegates(models.unet.DynamicUnet.__init__)
def unet_config(**kwargs):
    "Convenience function to easily create a config for `DynamicUnet`"
    return kwargs


# export
@log_args(to_return=True, but_as=Learner.__init__)
@delegates(Learner.__init__)
def unet_learner(dls, arch, loss_func=None, pretrained=True, cut=None, splitter=None, config=None, n_in=3, n_out=None,
                 normalize=True, **kwargs):
    "Build a unet learner from `dls` and `arch`"
    if config is None:
        config = unet_config()
    meta = model_meta.get(arch, _default_meta)
    body = create_body(arch, n_in, pretrained, ifnone(cut, meta['cut']))
    size = dls.one_batch()[0].shape[-2:]
    if n_out is None:
        n_out = get_c(dls)
    assert n_out, "`n_out` is not defined, and could not be inferred from data, set `dls.c` or pass `n_out`"
    if normalize:
        _add_norm(dls, meta, pretrained)
    model = models.unet.DynamicUnet(body, n_out, size, **config)
    learn = Learner(dls, model, loss_func=loss_func, splitter=ifnone(splitter, meta['split']), **kwargs)
    if pretrained:
        learn.freeze()
    return learn


# The model is built from `arch` using the number of final filters inferred from `dls` if possible (otherwise pass a value to `n_out`). It might be `pretrained` and the architecture is cut and split using the default metadata of the model architecture (this can be customized by passing a `cut` or a `splitter`).
#
# To customize the model creation, use `unet_config` and pass the result to the `config` argument.
#
# If `normalize` and `pretrained` are `True`, this function adds a `Normalization` transform to the `dls` (if there is not already one) using the statistics of the pretrained model. That way, you won't ever forget to normalize your data in transfer learning.
#
# All other arguments are passed to `Learner`.

# +
path = untar_data(URLs.CAMVID_TINY)
fnames = get_image_files(path / 'images')


def label_func(x): return path / 'labels' / f'{x.stem}_P{x.suffix}'


codes = np.loadtxt(path / 'codes.txt', dtype=str)

dls = SegmentationDataLoaders.from_label_func(path, fnames, label_func, codes=codes)
# -

learn = unet_learner(dls, models.resnet34, loss_func=CrossEntropyLossFlat(axis=1))

# hide
learn = unet_learner(dls, models.resnet34, pretrained=True, n_in=4)


# ## Show functions -

# export
@typedispatch
def show_results(x: TensorImage, y, samples, outs, ctxs=None, max_n=10, nrows=None, ncols=None, figsize=None, **kwargs):
    if ctxs is None:
        ctxs = get_grid(min(len(samples), max_n), nrows=nrows, ncols=ncols, add_vert=1, figsize=figsize)
    ctxs = show_results[object](x, y, samples, outs, ctxs=ctxs, max_n=max_n, **kwargs)
    return ctxs


# export
@typedispatch
def show_results(x: TensorImage, y: TensorCategory, samples, outs, ctxs=None, max_n=10, nrows=None, ncols=None, figsize=None, **kwargs):
    if ctxs is None:
        ctxs = get_grid(min(len(samples), max_n), nrows=nrows, ncols=ncols, add_vert=1, figsize=figsize)
    for i in range(2):
        ctxs = [b.show(ctx=c, **kwargs) for b, c, _ in zip(samples.itemgot(i), ctxs, range(max_n))]
    ctxs = [r.show(ctx=c, color='green' if b == r else 'red', **kwargs)
            for b, r, c, _ in zip(samples.itemgot(1), outs.itemgot(0), ctxs, range(max_n))]
    return ctxs


# export
@typedispatch
def show_results(x: TensorImage, y: (TensorMask, TensorPoint, TensorBBox), samples, outs, ctxs=None, max_n=6,
                 nrows=None, ncols=1, figsize=None, **kwargs):
    if ctxs is None:
        ctxs = get_grid(min(len(samples), max_n), nrows=nrows, ncols=ncols, add_vert=1, figsize=figsize, double=True,
                        title='Target/Prediction')
    for i in range(2):
        ctxs[::2] = [b.show(ctx=c, **kwargs) for b, c, _ in zip(samples.itemgot(i), ctxs[::2], range(2 * max_n))]
    for o in [samples, outs]:
        ctxs[1::2] = [b.show(ctx=c, **kwargs) for b, c, _ in zip(o.itemgot(0), ctxs[1::2], range(2 * max_n))]
    return ctxs


# export
@typedispatch
def show_results(x: TensorImage, y: TensorImage, samples, outs, ctxs=None, max_n=10, figsize=None, **kwargs):
    if ctxs is None:
        ctxs = get_grid(3 * min(len(samples), max_n), ncols=3, figsize=figsize, title='Input/Target/Prediction')
    for i in range(2):
        ctxs[i::3] = [b.show(ctx=c, **kwargs) for b, c, _ in zip(samples.itemgot(i), ctxs[i::3], range(max_n))]
    ctxs[2::3] = [b.show(ctx=c, **kwargs) for b, c, _ in zip(outs.itemgot(0), ctxs[2::3], range(max_n))]
    return ctxs


# export
@typedispatch
def plot_top_losses(x: TensorImage, y: TensorCategory, samples, outs, raws, losses, nrows=None, ncols=None, figsize=None, **kwargs):
    axs = get_grid(len(samples), nrows=nrows, ncols=ncols, add_vert=1, figsize=figsize, title='Prediction/Actual/Loss/Probability')
    for ax, s, o, r, l in zip(axs, samples, outs, raws, losses):
        s[0].show(ctx=ax, **kwargs)
        ax.set_title(f'{o[0]}/{s[1]} / {l.item():.2f} / {r.max().item():.2f}')


# export
@typedispatch
def plot_top_losses(x: TensorImage, y: TensorMultiCategory, samples, outs, raws, losses, nrows=None, ncols=None, figsize=None, **kwargs):
    axs = get_grid(len(samples), nrows=nrows, ncols=ncols, add_vert=1, figsize=figsize)
    for i, (ax, s) in enumerate(zip(axs, samples)):
        s[0].show(ctx=ax, title=f'Image {i}', **kwargs)
    rows = get_empty_df(len(samples))
    outs = L(s[1:] + o + (TitledStr(r), TitledFloat(l.item())) for s, o, r, l in zip(samples, outs, raws, losses))
    for i, l in enumerate(["target", "predicted", "probabilities", "loss"]):
        rows = [b.show(ctx=r, label=l, **kwargs) for b, r in zip(outs.itemgot(i), rows)]
    display_df(pd.DataFrame(rows))


# ## Export -

# hide
notebook2script()
