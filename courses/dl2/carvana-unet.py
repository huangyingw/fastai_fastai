
# coding: utf-8

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


from fastai.conv_learner import *
from fastai.dataset import *

from pathlib import Path
import json


# ## Data

PATH = Path('data/carvana')
MASKS_FN = 'train_masks.csv'
META_FN = 'metadata.csv'
masks_csv = pd.read_csv(PATH / MASKS_FN)
meta_csv = pd.read_csv(PATH / META_FN)


def show_img(im, figsize=None, ax=None, alpha=None):
    if not ax:
        fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(im, alpha=alpha)
    ax.set_axis_off()
    return ax


TRAIN_DN = 'train-128'
MASKS_DN = 'train_masks-128'
sz = 128
bs = 64
nw = 16


class MatchedFilesDataset(FilesDataset):
    def __init__(self, fnames, y, transform, path):
        self.y = y
        assert(len(fnames) == len(y))
        super().__init__(fnames, transform, path)

    def get_y(self, i): return open_image(os.path.join(self.path, self.y[i]))

    def get_c(self): return 0


x_names = np.array([Path(TRAIN_DN) / o for o in masks_csv['img']])
y_names = np.array([Path(MASKS_DN) / f'{o[:-4]}_mask.png' for o in masks_csv['img']])


val_idxs = list(range(1008))
((val_x, trn_x), (val_y, trn_y)) = split_by_idx(val_idxs, x_names, y_names)


aug_tfms = [RandomRotate(4, tfm_y=TfmType.PIXEL),
            RandomFlip(tfm_y=TfmType.PIXEL),
            RandomLighting(0.05, 0.05)]


tfms = tfms_from_model(resnet34, sz, crop_type=CropType.NO, tfm_y=TfmType.PIXEL, aug_tfms=aug_tfms)
datasets = ImageData.get_ds(MatchedFilesDataset, (trn_x, trn_y), (val_x, val_y), tfms, path=PATH)
md = ImageData(PATH, datasets, bs, num_workers=16, classes=None)
denorm = md.trn_ds.denorm


# ## Simple upsample

f = resnet34
cut, lr_cut = model_meta[f]


def get_base():
    layers = cut_model(f(True), cut)
    return nn.Sequential(*layers)


def mask_loss(pred, targ):
    return F.binary_cross_entropy_with_logits(pred[:, 0], targ[..., 0])


def mask_acc(pred, targ): return accuracy_multi(pred[:, 0], targ[..., 0], 0.)


def dice(pred, targs):
    m1 = (pred[:, 0] > 0).float()
    m2 = targs[..., 0]
    return 2. * (m1 * m2).sum() / (m1 + m2).sum()


class StdUpsample(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.conv = nn.ConvTranspose2d(nin, nout, 2, stride=2)
        self.bn = nn.BatchNorm2d(nout)

    def forward(self, x): return self.bn(F.relu(self.conv(x)))


class Upsample34(nn.Module):
    def __init__(self, rn):
        super().__init__()
        self.rn = rn
        self.up1 = StdUpsample(512, 256)
        self.up2 = StdUpsample(256, 256)
        self.up3 = StdUpsample(256, 256)
        self.up4 = StdUpsample(256, 256)
        self.up5 = nn.ConvTranspose2d(256, 1, 2, stride=2)

    def forward(self, x):
        x = F.relu(self.rn(x))
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        return x


m_base = get_base()


class UpsampleModel():
    def __init__(self, model, name='upsample'):
        self.model, self.name = model, name

    def get_layer_groups(self, precompute):
        lgs = list(split_by_idxs(children(self.model.rn), [lr_cut]))
        return lgs + [children(self.model)[1:]]


m = to_gpu(Upsample34(m_base))
models = UpsampleModel(m)


learn = ConvLearner(md, models)
learn.opt_fn = optim.Adam
learn.crit = mask_loss
learn.metrics = [mask_acc, dice]


learn.freeze_to(1)


learn.lr_find()
learn.sched.plot()


lr = 4e-2


learn.fit(lr, 1, cycle_len=4, use_clr=(20, 8))


learn.save('tmp')


learn.load('tmp')


learn.unfreeze()
learn.bn_freeze(True)
lrs = np.array([lr / 100, lr / 10, lr]) / 2


learn.fit(lrs, 1, cycle_len=4, use_clr=(20, 8))


learn.save('128')


# ## U-net (ish)

[o.features.size() for o in m.sfs]


class SaveFeatures():
    features = None

    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output): self.features = output

    def remove(self): self.hook.remove()


class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, 1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        return self.bn(F.relu(cat_p))


class Unet34(nn.Module):
    def __init__(self, rn):
        super().__init__()
        self.rn = rn
        self.sfs = [SaveFeatures(rn[i]) for i in [2, 4, 5, 6]]
        self.up1 = UnetBlock(512, 256, 256)
        self.up2 = UnetBlock(256, 128, 256)
        self.up3 = UnetBlock(256, 64, 256)
        self.up4 = UnetBlock(256, 64, 256)
        self.up5 = nn.ConvTranspose2d(256, 1, 2, stride=2)

    def forward(self, x):
        x = F.relu(self.rn(x))
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        x = self.up5(x)
        return x


m_base = get_base()
m = to_gpu(Unet34(m_base))
models = UpsampleModel(m)


learn = ConvLearner(md, models)
learn.opt_fn = optim.Adam
learn.crit = mask_loss
learn.metrics = [mask_acc, dice]


learn.freeze_to(1)


learn.lr_find()
learn.sched.plot()


learn.fit(lr, 1, cycle_len=8, use_clr=(20, 8))


learn.save('128urn-tmp')


learn.load('128urn-tmp')


learn.unfreeze()
learn.bn_freeze(True)
lrs = np.array([lr / 100, lr / 10, lr])


learn.fit(lrs, 1, cycle_len=40, use_clr=(20, 10))


learn.save('128urn-0')


learn.load('128urn-0')


x, y = next(iter(md.val_dl))
py = to_np(learn.model(V(x)))


# ax = show_img(denorm(x)[0])
show_img(py[0][0] > 0)


# ax = show_img(denorm(x)[0])
show_img(y[0, ..., -1])


# ## 512x512

sz = 512
bs = 16


tfms = tfms_from_model(resnet34, sz, crop_type=CropType.NO, tfm_y=TfmType.PIXEL, aug_tfms=aug_tfms)
datasets = ImageData.get_ds(MatchedFilesDataset, (trn_x, trn_y), (val_x, val_y), tfms, path=PATH)
md = ImageData(PATH, datasets, bs, num_workers=4, classes=None)
denorm = md.trn_ds.denorm


m_base = get_base()
m = to_gpu(Unet34(m_base))
models = UpsampleModel(m)


learn = ConvLearner(md, models)
learn.opt_fn = optim.Adam
learn.crit = mask_loss
learn.metrics = [mask_acc, dice]

learn.freeze_to(1)


learn.load('128urn-0')


learn.fit(lr, 1, cycle_len=5, use_clr=(20, 5))


learn.save('512urn-tmp')


learn.unfreeze()
learn.bn_freeze(True)


learn.fit(lrs, 1, cycle_len=8, use_clr=(20, 8))


learn.save('512urn')


learn.load('512urn')


x, y = next(iter(md.val_dl))
py = to_np(learn.model(V(x)))


show_img(py[0][0] > 0)


show_img(y[0, ..., -1])


# ## 1024x1024

sz = 1024
bs = 4


tfms = tfms_from_model(resnet34, sz, crop_type=CropType.NO, tfm_y=TfmType.PIXEL, aug_tfms=aug_tfms)
datasets = ImageData.get_ds(MatchedFilesDataset, (trn_x, trn_y), (val_x, val_y), tfms, path=PATH)
md = ImageData(PATH, datasets, bs, num_workers=4, classes=None)
denorm = md.trn_ds.denorm


m_base = get_base()
m = to_gpu(Unet34(m_base))
models = UpsampleModel(m)


learn = ConvLearner(md, models)
learn.opt_fn = optim.Adam
learn.crit = mask_loss
learn.metrics = [mask_acc, dice]

learn.freeze_to(1)


learn.load('512urn')


learn.fit(lr, 1, cycle_len=2, use_clr=(20, 4))


learn.save('1024urn-tmp')


learn.unfreeze()
learn.bn_freeze(True)


learn.fit(lrs / 2, 1, cycle_len=4, use_clr=(20, 8))


learn.save('1024urn')


learn.load('1024urn')


x, y = next(iter(md.val_dl))
py = to_np(learn.model(V(x)))


show_img(py[0][0] > 0)


show_img(y[0, ..., -1])
