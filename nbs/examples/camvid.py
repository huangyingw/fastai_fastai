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

import torch

from fastai.basics import *
from fastai.callback.all import *
from fastai.vision.all import *

path = untar_data(URLs.CAMVID)

valid_fnames = (path / 'valid.txt').read().split('\n')


def ListSplitter(valid_items):
    def _inner(items):
        val_mask = tensor([o.name in valid_items for o in items])
        return [~val_mask, val_mask]
    return _inner


codes = np.loadtxt(path / 'codes.txt', dtype=str)

# +
camvid = DataBlock(blocks=(ImageBlock, MaskBlock(codes)),
                   get_items=get_image_files,
                   splitter=ListSplitter(valid_fnames),
                   get_y=lambda o: path / 'labels' / f'{o.stem}_P{o.suffix}',
                   batch_tfms=[*aug_transforms(size=(360, 480)), Normalize.from_stats(*imagenet_stats)])

dls = camvid.dataloaders(path / "images", bs=8)
# -

dls = SegmentationDataLoaders.from_label_func(path, bs=8,
                                              fnames=get_image_files(path / "images"),
                                              label_func=lambda o: path / 'labels' / f'{o.stem}_P{o.suffix}',
                                              codes=codes,
                                              batch_tfms=[*aug_transforms(size=(360, 480)), Normalize.from_stats(*imagenet_stats)])

dls.show_batch(max_n=2, rows=1, figsize=(20, 7))

dls.show_batch(max_n=4, figsize=(20, 14))

codes = np.loadtxt(path / 'codes.txt', dtype=str)
dls.vocab = codes

# +
name2id = {v: k for k, v in enumerate(codes)}
void_code = name2id['Void']

def acc_camvid(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask] == target[mask]).float().mean()


# +
opt_func = partial(Adam, lr=3e-3, wd=0.01)  # , eps=1e-8)

learn = unet_learner(dls, resnet34, loss_func=CrossEntropyLossFlat(axis=1), opt_func=opt_func, path=path, metrics=acc_camvid,
                     config=unet_config(norm_type=None), wd_bn_bias=True)
# -

get_c(dls)

learn.lr_find()

lr = 3e-3
learn.freeze()

learn.fit_one_cycle(10, slice(lr), pct_start=0.9, wd=1e-2)

learn.show_results(max_n=2, rows=2, vmin=1, vmax=30, figsize=(14, 10))

learn.save('stage-1')

learn.load('stage-1')
learn.unfreeze()

# +
# learn.opt.clear_state() #Not necessarily useful
# -

lrs = slice(lr / 400, lr / 4)

learn.fit_one_cycle(12, lrs, pct_start=0.8, wd=1e-2)

learn.show_results(max_n=4, vmin=1, vmax=30, figsize=(15, 6))
