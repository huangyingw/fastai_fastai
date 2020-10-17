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

# ## Regression with BIWI head pose dataset

# This is a more advanced example to show how to create custom datasets and do regression with images. Our task is to find the center of the head in each image. The data comes from the [BIWI head pose dataset](https://data.vision.ee.ethz.ch/cvl/gfanelli/head_pose/head_forest.html#db), thanks to Gabriele Fanelli et al. We have converted the images to jpeg format, so you should download the converted dataset from [this link](https://s3.amazonaws.com/fast-ai-imagelocal/biwi_head_pose.tgz).

# %matplotlib inline

from fastai.vision.all import *
from nbdev.showdoc import *

# ## Getting and converting the data

path = untar_data(URLs.BIWI_HEAD_POSE)

cal = np.genfromtxt(path / '01' / 'rgb.cal', skip_footer=6)
cal

fname = '09/frame_00667_rgb.jpg'


def img2txt_name(f): return path / f'{str(f)[:-7]}pose.txt'


img = PILImage.create(path / fname)
img.show()

ctr = np.genfromtxt(img2txt_name(fname), skip_header=3)
ctr


# +
def convert_biwi(coords):
    c1 = coords[0] * cal[0][0] / coords[2] + cal[0][2]
    c2 = coords[1] * cal[1][1] / coords[2] + cal[1][2]
    return tensor([c1, c2])

def get_ctr(f):
    ctr = np.genfromtxt(img2txt_name(f), skip_header=3)
    return convert_biwi(ctr)

def get_ip(img, pts): return TensorPoint.create(pts, img_size=img.size)


# -

get_ctr(fname)

ctr = get_ctr(fname)
ax = img.show(figsize=(6, 6))
get_ip(img, ctr).show(ctx=ax)

# ## Creating a dataset

dblock = DataBlock(blocks=(ImageBlock, PointBlock),
                   get_items=get_image_files,
                   splitter=FuncSplitter(lambda o: o.parent.name == '13'),
                   get_y=get_ctr,
                   batch_tfms=[*aug_transforms(size=(120, 160)), Normalize.from_stats(*imagenet_stats)])

dls = dblock.dataloaders(path, path=path, bs=64)

dls.show_batch(max_n=9, figsize=(9, 6))

# ## Train model

# TODO: look in after_item for c
dls.c = dls.train.after_item.c

learn = cnn_learner(dls, resnet34)

learn.lr_find()

lr = 2e-2

learn.fit_one_cycle(5, slice(lr))

learn.save('stage-1')

learn.load('stage-1')

learn.show_results(max_n=6)


# ## Data augmentation

def repeat_one_file(path):
    items = get_image_files(path)
    return [items[0]] * 500


dblock = DataBlock(blocks=(ImageBlock, PointBlock),
                   get_items=repeat_one_file,
                   splitter=RandomSplitter(),
                   get_y=get_ctr)

tfms = aug_transforms(max_rotate=20, max_zoom=1.5, max_lighting=0.5, max_warp=0.4, p_affine=1., p_lighting=1., size=(120, 160))

dls = dblock.dataloaders(path, path=path, bs=64, batch_tfms=[*tfms, Normalize.from_stats(*imagenet_stats)])

dls.show_batch(max_n=9, figsize=(8, 6))
