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

# ## Image segmentation with CamVid

# %matplotlib inline

import gc
from fastai.basics import *
from fastai.callback.all import *
from fastai.vision.all import *
from nbdev.showdoc import *

path = untar_data(URLs.CAMVID)
path.ls()

path_lbl = path / 'labels'
path_img = path / 'images'

# ## Data

fnames = get_image_files(path_img)
fnames[:3]

lbl_names = get_image_files(path_lbl)
lbl_names[:3]

img_f = fnames[0]
img = PILImage.create(img_f)
img.show(figsize=(5, 5))


def get_y_fn(x): return path_lbl / f'{x.stem}_P{x.suffix}'


mask = PILMask.create(get_y_fn(img_f))
mask.show(figsize=(5, 5), alpha=1)

src_size = np.array(mask.shape[1:])
src_size, tensor(mask)

codes = np.loadtxt(path / 'codes.txt', dtype=str)
codes

# ## Datasets

size = src_size // 2
bs = 8

valid_fnames = (path / 'valid.txt').read().split('\n')


# export
def FileSplitter(fname):
    "Split `items` depending on the value of `mask`."
    valid = Path(fname).read().split('\n')
    def _func(x): return x.name in valid
    def _inner(o, **kwargs): return FuncSplitter(_func)(o)
    return _inner


camvid = DataBlock(blocks=(ImageBlock, MaskBlock(codes)),
                   get_items=get_image_files,
                   splitter=FileSplitter(path / 'valid.txt'),
                   get_y=lambda o: path / 'labels' / f'{o.stem}_P{o.suffix}',
                   batch_tfms=[*aug_transforms(size=(360, 480)), Normalize.from_stats(*imagenet_stats)])

dls = camvid.dataloaders(path / "images", bs=8, path=path)

show_at(dls.train_ds, 0)

b = dls.train.one_batch()

b = dls.train.decode(b)

b[0].shape, b[1].shape

dls.show_batch(max_n=4, figsize=(20, 14))

# ## Model

# +
name2id = {v: k for k, v in enumerate(codes)}
void_code = name2id['Void']


def acc_camvid(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask] == target[mask]).float().mean()


# -

metrics = acc_camvid
# metrics=accuracy

wd = 1e-2

dls.vocab = codes

learn = unet_learner(dls, resnet34, metrics=metrics)

learn.model

learn.lr_find()

lr = 3e-3

learn.fit_one_cycle(10, slice(lr), pct_start=0.9, wd=wd)

learn.save('stage-1')

learn.load('stage-1')

learn.show_results(max_n=4, figsize=(9, 4))

learn.unfreeze()

lrs = slice(lr / 400, lr / 4)

learn.fit_one_cycle(12, lrs, pct_start=0.8, wd=wd)

learn.save('stage-2')

# ## Go big

# You may have to restart your kernel and come back to this stage if you run out of memory, and may also need to decrease `bs`.

del learn
gc.collect()

size = src_size
bs = 3
# depending on your GPU RAM you may need to use
# bs=1
print(f"using bs={bs}, have {free}MB of GPU RAM free")

dls = camvid.dataloaders(path / "images", bs=3, path=path,
                         batch_tfms=[*aug_transforms(), Normalize.from_stats(*imagenet_stats)])

dls.vocab = codes

learn = unet_learner(dls, resnet34, metrics=metrics)

learn.load('stage-2')

learn.lr_find()

lr = 1e-3

learn.fit_one_cycle(10, slice(lr), pct_start=0.8)

learn.save('stage-1-big')

learn.load('stage-1-big')

learn.unfreeze()

lrs = slice(1e-6, lr / 10)

learn.fit_one_cycle(10, lrs)

learn.save('stage-2-big')

learn.load('stage-2-big')

learn.show_results(max_n=1, figsize=(20, 10), vmin=1, vmax=30)


# ## fin
