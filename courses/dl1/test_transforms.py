
# coding: utf-8

# ## Testing transforms.py

get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

# This file contains all the main external libs we'll use
from fastai.imports import *
from fastai.transforms import *
from fastai.plots import *
from fastai.dataset import *


PATH = "data/fish/"
PATH = "/data2/yinterian/fisheries-kaggle/"


# ### Fish with bounding box

fnames, corner_labels, _, _ = parse_csv_labels(f'{PATH}trn_bb_corners_labels', skip_header=False)


def get_x(f):
    return open_image(f'{PATH}/images/{f}')


f = 'img_02642.jpg'
x = get_x(f)
y = np.array(corner_labels[f], dtype=np.float32)
y


x.shape


rows = np.rint([y[0], y[0], y[2], y[2]]).astype(int)
rows


cols = np.rint([y[1], y[3], y[1], y[3]]).astype(int)
cols


corner_labels["img_02642.jpg"]


def create_corner_rect(bb, color='red'):
    bb = np.array(bb, dtype=np.float32)
    return plt.Rectangle((bb[1], bb[0]), bb[3] - bb[1], bb[2] - bb[0], color=color, fill=False, lw=3)

def show_corner_bb(f='img_04908.jpg'):
    file_path = f'{PATH}images/{f}'
    bb = corner_labels[f]
    plots_from_files([file_path])
    plt.gca().add_patch(create_corner_rect(bb))


show_corner_bb(f='img_02642.jpg')


def create_rect(bb, color='red'):
    return plt.Rectangle((bb[1], bb[0]), bb[3] - bb[1], bb[2] - bb[0], color=color, fill=False, lw=3)

def plotXY(x, y):
    plots([x])
    plt.gca().add_patch(create_rect(y))


plotXY(x, y)


# ## Scale

xx, yy = Scale(sz=350, tfm_y=TfmType.COORD)(x, y)


plotXY(xx, yy)


xx, yy = Scale(sz=350, tfm_y=TfmType.PIXEL)(x, x)
plots([xx, yy])


# ## RandomScale

xx, yy = RandomScale(sz=350, max_zoom=1.1, tfm_y=TfmType.COORD)(x, y)
plotXY(xx, yy)
print(yy)
print(y)


xx, yy = RandomScale(sz=350, max_zoom=1.1, tfm_y=TfmType.PIXEL)(x, x)
plots([xx, yy])


# ## RandomCrop

xx, yy = RandomCrop(targ=350, tfm_y=TfmType.COORD)(x, y)


plotXY(xx, yy)


xx, yy = RandomCrop(350, tfm_y=TfmType.PIXEL)(x, x)
plots([xx, yy])


# ## No Cropping

xx, yy = NoCrop(350, tfm_y=TfmType.COORD)(x, y)


print(yy)
plotXY(xx, yy)


xx, yy = NoCrop(350, tfm_y=TfmType.PIXEL)(x, x)
plots([xx, yy])


# ## CenterCrop

xx, yy = CenterCrop(350, tfm_y=TfmType.COORD)(x, y)


plotXY(xx, yy)


xx, yy = CenterCrop(350, tfm_y=TfmType.PIXEL)(x, x)
plots([xx, yy])


# ## Random Dihedral

xx, yy = RandomDihedral(TfmType.COORD)(x, y)


print(yy)


plotXY(xx, yy)


xx, yy = RandomDihedral(tfm_y=TfmType.PIXEL)(x, x)


plots([xx, yy])


# ## RandomFlipXY

xx, yy = RandomFlip(TfmType.COORD)(x, y)
print(yy)
plotXY(xx, yy)


xx, yy = RandomFlip(TfmType.PIXEL)(x, x)
plots([xx, yy])


# ## RandomLightingXY (talk to Jeremy about this)

xx, yy = RandomLighting(0.5, 0.5)(x, y)
plotXY(xx, yy)


# talk to Jeremy about this
xx, yy = RandomLighting(0.5, 0.5, TfmType.PIXEL)(x, x)
plots([xx, yy])


# ## RandomRotate

xx, yy = RandomRotate(deg=30, p=1, tfm_y=TfmType.COORD)(x, y)
plotXY(xx, yy)
print(yy)


xx, yy = RandomRotate(130, p=1.0, tfm_y=TfmType.COORD)(x, y)
plotXY(xx, yy)


xx, yy = RandomRotate(0.5, 0.5, TfmType.PIXEL)(x, x)
plots([xx, yy])
