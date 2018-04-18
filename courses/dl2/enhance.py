
# coding: utf-8

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ## Super resolution data

from fastai.conv_learner import *
from pathlib import Path
torch.cuda.set_device(0)

torch.backends.cudnn.benchmark = True


PATH = Path('data/imgnet-samp/')
PATH_TRN = PATH / 'train'


fnames, label_arr, all_labels = folder_source(PATH, 'train')
fnames = ['/'.join(Path(fn).parts[-2:]) for fn in fnames]
list(zip(fnames[:5], label_arr[:5]))


all_labels[:5]


arch = resnet34
sz_hr, sz_lr = 288, 72
bs = 16


class MatchedFilesDataset(FilesDataset):
    def __init__(self, fnames, y, transform, path):
        self.y = y
        assert(len(fnames) == len(y))
        super().__init__(fnames, transform, path)

    def get_y(self, i): return open_image(os.path.join(self.path, self.y[i]))

    def get_c(self): return 0


aug_tfms = [RandomRotate(4, tfm_y=TfmType.PIXEL),
            RandomFlip(tfm_y=TfmType.PIXEL),
            RandomLighting(0.05, 0.05)]
# aug_tfms = []


val_idxs = get_cv_idxs(len(fnames), val_pct=0.1)
((val_x, trn_x), (val_y, trn_y)) = split_by_idx(val_idxs, np.array(fnames), np.array(fnames))
len(val_x), len(trn_x)


tfms = tfms_from_model(arch, sz_lr, tfm_y=TfmType.PIXEL, aug_tfms=aug_tfms, sz_y=sz_hr)
datasets = ImageData.get_ds(MatchedFilesDataset, (trn_x, trn_y), (val_x, val_y), tfms, path=PATH_TRN)
md = ImageData(PATH, datasets, bs, num_workers=8, classes=None)


denorm = md.trn_ds.denorm


def show_img(ims, idx, figsize=(5, 5)):
    plt.figure(figsize=figsize)
    plt.imshow(np.clip(denorm(ims), 0, 1)[idx], interpolation="bilinear")


x, y = next(iter(md.val_dl))
x.size(), y.size()


show_img(x, 0)


show_img(y, 0)


batches = [next(iter(md.aug_dl)) for i in range(9)]


fig, axes = plt.subplots(3, 6, figsize=(18, 9))
for i, (x, y) in enumerate(batches):
    axes.flat[i * 2].imshow(denorm(x)[0])
    axes.flat[i * 2 + 1].imshow(denorm(y)[0])


# ## Model

class BnReluConv(nn.Module):
    def __init__(self, ni, nf, kernel_size=3, bias=False):
        super().__init__()
        self.bn = nn.BatchNorm2d(ni)
        self.conv = nn.Conv2d(ni, nf, kernel_size, padding=kernel_size // 2, bias=bias)

    def forward(self, x): return self.conv(F.relu(self.bn(x), inplace=True))


class ResBlock(nn.Module):
    def __init__(self, ni, nf):
        super().__init__()
        self.conv1 = BnReluConv(ni, nf)
        self.conv2 = BnReluConv(ni, nf)

    def forward(self, x): return x + self.conv2(self.conv1(x))


class SrResnet(nn.Module):
    def __init__(self, do_sigmoid=False):
        super().__init__()
        self.do_sigmoid = do_sigmoid
        self.conv1 = nn.Conv2d(3, 64, 9, padding=4, bias=False)
        self.blocks = nn.ModuleList([ResBlock(64, 64) for i in range(4)])
        self.uscale = nn.Upsample(scale_factor=2, mode='bilinear')
        self.uconv1 = BnReluConv(64, 64)
        self.uconv2 = BnReluConv(64, 64)
        self.conv2 = BnReluConv(64, 3, kernel_size=9, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        for block in self.blocks:
            x = block(x)
        x = self.uconv1(self.uscale(x))
        x = self.uconv2(self.uscale(x))
        x = self.conv2(x)
        return F.sigmoid(x) if self.do_sigmoid else x


opt_fn = partial(optim.Adam, betas=(0.8, 0.99))


m = SrResnet()
learn = Learner(md, SingleModel(to_gpu(m)), opt_fn=opt_fn)
learn.crit = F.mse_loss


learn.lr_find()
learn.sched.plot()


lr = 1e-3


learn.fit(lr, 1, cycle_len=10, use_clr=(40, 10))


preds, y = learn.predict_with_targs()


idx = 1
show_img(y, idx)


show_img(preds, idx)


# ## Perceptual loss

m_vgg = vgg16(True).cuda().eval()
set_trainable(m_vgg, False)


# m_vgg = nn.Sequential(*children(m_vgg)[:13]).cuda().eval()


block_ends = [i - 1 for i, o in enumerate(children(m_vgg))
              if isinstance(o, nn.MaxPool2d)][1:]
block_ends


class SaveFeatures():
    features = None

    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output): self.features = output

    def remove(self): self.hook.remove()


class FeatureLoss(nn.Module):
    def __init__(self, m, layer_ids, layer_wgts):
        super().__init__()
        self.m, self.wgts = m, layer_wgts
        self.sfs = [SaveFeatures(m[i]) for i in layer_ids]

    def forward(self, input, target, sum_layers=True):
        res = [F.mse_loss(input, target) / 30]
        self.m(VV(target.data))
        targ_feat = [V(o.features.data.clone()) for o in self.sfs]
        self.m(V(input.data))
        res += [F.mse_loss(inp.features, targ) * wgt for inp, targ, wgt in zip(self.sfs, targ_feat, self.wgts)]
        if sum_layers:
            res = sum(res)
        return res

    def close(self):
        for o in self.sfs:
            o.remove()


layers = block_ends[:2]
wgts = [1, 1]


learn.crit.close()


m = SrResnet()
learn = Learner(md, SingleModel(to_gpu(m)), opt_fn=opt_fn)
learn.crit = FeatureLoss(m_vgg, layers, wgts)


learn.lr_find()
learn.sched.plot()


lr = 1e-3


learn.fit(lr, 1, cycle_len=50, use_clr=(40, 10))


learn.save('sr-samp')


learn.load('sr-samp')


x, y = next(iter(md.val_dl))


learn.model.eval()
preds = learn.model(V(x))
learn.crit(preds, V(y), sum_layers=False)


idx = 15
show_img(y, idx)


show_img(preds, idx)


show_img(x, idx)
