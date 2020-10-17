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

# %matplotlib inline

# # Lesson 6: pets revisited

from fastai.vision.all import *
from nbdev.showdoc import *

bs = 64

path = untar_data(URLs.PETS)

# ## Data augmentation

tfms = partial(aug_transforms, max_rotate=20, max_zoom=1.3, max_lighting=0.4, max_warp=0.4,
               p_affine=1., p_lighting=1.)

doc(aug_transforms)


def repeat_one(source, n=128): return [get_image_files(source)[0]] * n


def get_dls(size, bs, pad_mode='reflection', batch=False):
    pets = DataBlock(blocks=(ImageBlock, CategoryBlock),
                     get_items=repeat_one,
                     splitter=RandomSplitter(0.2, seed=2),
                     get_y=RegexLabeller(r'([^/]+)_\d+.jpg$'),
                     item_tfms=Resize(460),
                     batch_tfms=[*tfms(size=size, pad_mode=pad_mode, batch=batch), Normalize.from_stats(*imagenet_stats)])
    return pets.dataloaders(path / 'images', path=path, bs=bs)


dls = get_dls(224, bs, 'zeros')

dls.show_batch(max_n=9, figsize=(8, 8))

dls = get_dls(224, bs)

dls.show_batch(max_n=9, figsize=(8, 8))

# `batch=True` means we pick the same random augmentation for all the images in the batch.

dls = get_dls(224, bs, batch=True)

dls.show_batch(max_n=9, figsize=(8, 8))


# ## Train a model

def get_dls(size, bs, pad_mode='reflection'):
    pets = DataBlock(blocks=(ImageBlock, CategoryBlock),
                     get_items=get_image_files,
                     splitter=RandomSplitter(0.2, seed=2),
                     get_y=RegexLabeller(r'([^/]+)_\d+.jpg$'),
                     item_tfms=RandomResizedCrop(460, min_scale=0.75),
                     batch_tfms=[*tfms(size=size, pad_mode=pad_mode), Normalize.from_stats(*imagenet_stats)])
    return pets.dataloaders(path / 'images', path=path, bs=bs)


dls = get_dls(224, bs)

learn = cnn_learner(dls, resnet34, metrics=error_rate, config=cnn_config(bn_final=True))

learn.fit_one_cycle(3, slice(1e-2), pct_start=0.8)

learn.unfreeze()
learn.fit_one_cycle(2, lr_max=slice(1e-6, 1e-3), pct_start=0.8)

dls = get_dls(352, bs)
learn.dls = dls

learn.fit_one_cycle(2, lr_max=slice(1e-6, 1e-4))

learn.save('352')

# ## Convolution kernel

dls = get_dls(352, 16)

learn = cnn_learner(dls, resnet34, metrics=error_rate, config=cnn_config(bn_final=True)).load('352')

idx = 0
x, y = dls.valid_ds[idx]
show_at(dls.valid_ds, idx)

k = tensor([
    [0., -5 / 3, 1],
    [-5 / 3, -5 / 3, 1],
    [1., 1, 1],
]).expand(1, 3, 3, 3) / 6

k

k.shape

t = tensor(x).permute(2, 0, 1).float()
t.shape

t[None].shape

edge = F.conv2d(t[None], k)

show_image(edge[0], figsize=(5, 5))

dls.c

learn.model

print(learn.summary())

# ## Heatmap

m = learn.model.eval()

b = dls.one_batch()
xb_im = TensorImage(dls.train.decode(b)[0][0])
xb = b[0]


def hooked_backward(cat=y):
    with hook_output(m[0]) as hook_a:
        with hook_output(m[0], grad=True) as hook_g:
            preds = m(xb)
            preds[0, int(cat)].backward()
    return hook_a, hook_g


hook_a, hook_g = hooked_backward()

acts = hook_a.stored[0].cpu()
acts.shape

avg_acts = acts.mean(0)
avg_acts.shape


def show_heatmap(hm):
    _, ax = plt.subplots()
    xb_im.show(ctx=ax)
    ax.imshow(hm, alpha=0.6, extent=(0, 352, 352, 0),
              interpolation='bilinear', cmap='magma')


show_heatmap(avg_acts)

# ## Grad-CAM

# Paper: [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391)

grad = hook_g.stored[0][0].cpu()
grad_chan = grad.mean(1).mean(1)
grad.shape, grad_chan.shape

mult = (acts * grad_chan[..., None, None]).mean(0)

show_heatmap(mult)

fn = Path.home() / 'tmp/bulldog_maine.png'  # Replace with your own image

x = PILImage.create(fn)
x

dl = dls.test_dl([fn])
b = dl.one_batch()
xb_im = TensorImage(dls.train.decode(b)[0][0])
xb = b[0]

hook_a, hook_g = hooked_backward()

# +
acts = hook_a.stored[0].cpu()
grad = hook_g.stored[0][0].cpu()

grad_chan = grad.mean(1).mean(1)
mult = (acts * grad_chan[..., None, None]).mean(0)
# -

show_heatmap(mult)

dls.vocab[0]

hook_a, hook_g = hooked_backward(0)

# +
acts = hook_a.stored[0].cpu()
grad = hook_g.stored[0][0].cpu()

grad_chan = grad.mean(1).mean(1)
mult = (acts * grad_chan[..., None, None]).mean(0)
# -

show_heatmap(mult)

# ## fin
