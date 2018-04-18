
# coding: utf-8

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


from fastai.conv_learner import *
from fastai.dataset import *

from pathlib import Path
import json
torch.cuda.set_device(1)


# ## Data

# ### Setup

PATH = Path('data/carvana')
list(PATH.iterdir())


MASKS_FN = 'train_masks.csv'
META_FN = 'metadata.csv'
TRAIN_DN = 'train'
MASKS_DN = 'train_masks'


masks_csv = pd.read_csv(PATH / MASKS_FN)
masks_csv.head()


meta_csv = pd.read_csv(PATH / META_FN)
meta_csv.head()


def show_img(im, figsize=None, ax=None, alpha=None):
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(im, alpha=alpha)
    ax.set_axis_off()
    return ax


CAR_ID = '00087a6bd4dc'


list((PATH / TRAIN_DN).iterdir())[:5]


Image.open(PATH / TRAIN_DN / f'{CAR_ID}_01.jpg').resize((300, 200))


list((PATH / MASKS_DN).iterdir())[:5]


Image.open(PATH / MASKS_DN / f'{CAR_ID}_01_mask.gif').resize((300, 200))


ims = [open_image(PATH / TRAIN_DN / f'{CAR_ID}_{i+1:02d}.jpg') for i in range(16)]


fig, axes = plt.subplots(4, 4, figsize=(9, 6))
for i, ax in enumerate(axes.flat):
    show_img(ims[i], ax=ax)
plt.tight_layout(pad=0.1)


# ### Resize and convert

(PATH / 'train_masks_png').mkdir(exist_ok=True)


def convert_img(fn):
    fn = fn.name
    Image.open(PATH / 'train_masks' / fn).save(PATH / 'train_masks_png' / f'{fn[:-4]}.png')


files = list((PATH / 'train_masks').iterdir())
with ThreadPoolExecutor(8) as e:
    e.map(convert_img, files)


(PATH / 'train_masks-128').mkdir(exist_ok=True)


def resize_mask(fn):
    Image.open(fn).resize((128, 128)).save((fn.parent.parent) / 'train_masks-128' / fn.name)


files = list((PATH / 'train_masks_png').iterdir())
with ThreadPoolExecutor(8) as e:
    e.map(resize_img, files)


(PATH / 'train-128').mkdir(exist_ok=True)


def resize_img(fn):
    Image.open(fn).resize((128, 128)).save((fn.parent.parent) / 'train-128' / fn.name)


files = list((PATH / 'train').iterdir())
with ThreadPoolExecutor(8) as e:
    e.map(resize_img, files)


# ## Dataset

TRAIN_DN = 'train-128'
MASKS_DN = 'train_masks-128'
sz = 128
bs = 64


ims = [open_image(PATH / TRAIN_DN / f'{CAR_ID}_{i+1:02d}.jpg') for i in range(16)]
im_masks = [open_image(PATH / MASKS_DN / f'{CAR_ID}_{i+1:02d}_mask.png') for i in range(16)]


fig, axes = plt.subplots(4, 4, figsize=(9, 6))
for i, ax in enumerate(axes.flat):
    ax = show_img(ims[i], ax=ax)
    show_img(im_masks[i][..., 0], ax=ax, alpha=0.5)
plt.tight_layout(pad=0.1)


class MatchedFilesDataset(FilesDataset):
    def __init__(self, fnames, y, transform, path):
        self.y = y
        assert(len(fnames) == len(y))
        super().__init__(fnames, transform, path)

    def get_y(self, i): return open_image(os.path.join(self.path, self.y[i]))

    def get_c(self): return 0


x_names = np.array([Path(TRAIN_DN) / o for o in masks_csv['img']])
y_names = np.array([Path(MASKS_DN) / f'{o[:-4]}_mask.png' for o in masks_csv['img']])


len(x_names) // 16 // 5 * 16


val_idxs = list(range(1008))
((val_x, trn_x), (val_y, trn_y)) = split_by_idx(val_idxs, x_names, y_names)
len(val_x), len(trn_x)


aug_tfms = [RandomRotate(4, tfm_y=TfmType.CLASS),
            RandomFlip(tfm_y=TfmType.CLASS),
            RandomLighting(0.05, 0.05)]
# aug_tfms = []


tfms = tfms_from_model(resnet34, sz, crop_type=CropType.NO, tfm_y=TfmType.CLASS, aug_tfms=aug_tfms)
datasets = ImageData.get_ds(MatchedFilesDataset, (trn_x, trn_y), (val_x, val_y), tfms, path=PATH)
md = ImageData(PATH, datasets, bs, num_workers=8, classes=None)


denorm = md.trn_ds.denorm
x, y = next(iter(md.aug_dl))
x = denorm(x)


fig, axes = plt.subplots(5, 6, figsize=(12, 10))
for i, ax in enumerate(axes.flat):
    ax = show_img(x[i], ax=ax)
    show_img(y[i], ax=ax, alpha=0.5)
plt.tight_layout(pad=0.1)


# ## Model

class Empty(nn.Module):
    def forward(self, x): return x


models = ConvnetBuilder(resnet34, 0, 0, 0, custom_head=Empty())
learn = ConvLearner(md, models)
learn.summary()


class StdUpsample(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.conv = nn.ConvTranspose2d(nin, nout, 2, stride=2)
        self.bn = nn.BatchNorm2d(nout)

    def forward(self, x): return self.bn(F.relu(self.conv(x)))


flatten_channel = Lambda(lambda x: x[:, 0])


simple_up = nn.Sequential(
    nn.ReLU(),
    StdUpsample(512, 256),
    StdUpsample(256, 256),
    StdUpsample(256, 256),
    StdUpsample(256, 256),
    nn.ConvTranspose2d(256, 1, 2, stride=2),
    flatten_channel
)


models = ConvnetBuilder(resnet34, 0, 0, 0, custom_head=simple_up)
learn = ConvLearner(md, models)
learn.opt_fn = optim.Adam
learn.crit = nn.BCEWithLogitsLoss()
learn.metrics = [accuracy_thresh(0.5)]


learn.lr_find()
learn.sched.plot()


lr = 4e-2


learn.fit(lr, 1, cycle_len=5, use_clr=(20, 5))


learn.save('tmp')


learn.load('tmp')


py, ay = learn.predict_with_targs()


ay.shape


show_img(ay[0])


show_img(py[0] > 0)


learn.unfreeze()


learn.bn_freeze(True)


lrs = np.array([lr / 100, lr / 10, lr]) / 4


learn.fit(lrs, 1, cycle_len=20, use_clr=(20, 10))


learn.save('0')


x, y = next(iter(md.val_dl))
py = to_np(learn.model(V(x)))


ax = show_img(denorm(x)[0])
show_img(py[0] > 0, ax=ax, alpha=0.5)


ax = show_img(denorm(x)[0])
show_img(y[0], ax=ax, alpha=0.5)


# ## 512x512

TRAIN_DN = 'train'
MASKS_DN = 'train_masks_png'
sz = 512
bs = 16


x_names = np.array([Path(TRAIN_DN) / o for o in masks_csv['img']])
y_names = np.array([Path(MASKS_DN) / f'{o[:-4]}_mask.png' for o in masks_csv['img']])


((val_x, trn_x), (val_y, trn_y)) = split_by_idx(val_idxs, x_names, y_names)
len(val_x), len(trn_x)


tfms = tfms_from_model(resnet34, sz, crop_type=CropType.NO, tfm_y=TfmType.CLASS, aug_tfms=aug_tfms)
datasets = ImageData.get_ds(MatchedFilesDataset, (trn_x, trn_y), (val_x, val_y), tfms, path=PATH)
md = ImageData(PATH, datasets, bs, num_workers=8, classes=None)


denorm = md.trn_ds.denorm
x, y = next(iter(md.aug_dl))
x = denorm(x)


fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax = show_img(x[i], ax=ax)
    show_img(y[i], ax=ax, alpha=0.5)
plt.tight_layout(pad=0.1)


simple_up = nn.Sequential(
    nn.ReLU(),
    StdUpsample(512, 256),
    StdUpsample(256, 256),
    StdUpsample(256, 256),
    StdUpsample(256, 256),
    nn.ConvTranspose2d(256, 1, 2, stride=2),
    flatten_channel
)


models = ConvnetBuilder(resnet34, 0, 0, 0, custom_head=simple_up)
learn = ConvLearner(md, models)
learn.opt_fn = optim.Adam
learn.crit = nn.BCEWithLogitsLoss()
learn.metrics = [accuracy_thresh(0.5)]


learn.load('0')


learn.lr_find()
learn.sched.plot()


lr = 4e-2


learn.fit(lr, 1, cycle_len=5, use_clr=(20, 5))


learn.save('tmp')


learn.load('tmp')


learn.unfreeze()
learn.bn_freeze(True)


lrs = np.array([lr / 100, lr / 10, lr]) / 4


learn.fit(lrs, 1, cycle_len=8, use_clr=(20, 8))


learn.save('512')


x, y = next(iter(md.val_dl))
py = to_np(learn.model(V(x)))


ax = show_img(denorm(x)[0])
show_img(py[0] > 0, ax=ax, alpha=0.5)


ax = show_img(denorm(x)[0])
show_img(y[0], ax=ax, alpha=0.5)


# ## 1024x1024

sz = 1024
bs = 4


tfms = tfms_from_model(resnet34, sz, crop_type=CropType.NO, tfm_y=TfmType.CLASS, aug_tfms=aug_tfms)
datasets = ImageData.get_ds(MatchedFilesDataset, (trn_x, trn_y), (val_x, val_y), tfms, path=PATH)
md = ImageData(PATH, datasets, bs, num_workers=8, classes=None)


denorm = md.trn_ds.denorm
x, y = next(iter(md.aug_dl))
x = denorm(x)
y = to_np(y)


fig, axes = plt.subplots(2, 2, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    show_img(x[i], ax=ax)
    show_img(y[i], ax=ax, alpha=0.5)
plt.tight_layout(pad=0.1)


simple_up = nn.Sequential(
    nn.ReLU(),
    StdUpsample(512, 256),
    StdUpsample(256, 256),
    StdUpsample(256, 256),
    StdUpsample(256, 256),
    nn.ConvTranspose2d(256, 1, 2, stride=2),
    flatten_channel,
)


models = ConvnetBuilder(resnet34, 0, 0, 0, custom_head=simple_up)
learn = ConvLearner(md, models)
learn.opt_fn = optim.Adam
learn.crit = nn.BCEWithLogitsLoss()
learn.metrics = [accuracy_thresh(0.5)]


learn.load('512')


learn.lr_find()
learn.sched.plot()


lr = 4e-2


learn.fit(lr, 1, cycle_len=2, use_clr=(20, 4))


learn.save('tmp')


learn.load('tmp')


learn.unfreeze()
learn.bn_freeze(True)


lrs = np.array([lr / 100, lr / 10, lr]) / 8


learn.fit(lrs, 1, cycle_len=40, use_clr=(20, 10))


learn.save('1024')


x, y = next(iter(md.val_dl))
py = to_np(learn.model(V(x)))


ax = show_img(denorm(x)[0])
show_img(py[0][0] > 0, ax=ax, alpha=0.5)


ax = show_img(denorm(x)[0])
show_img(y[0, ..., -1], ax=ax, alpha=0.5)


show_img(py[0][0] > 0)


show_img(y[0, ..., -1])


# ## Fin
