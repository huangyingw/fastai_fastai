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
from nbdev.showdoc import showdoc
from fastai.data.external import *
from fastai.vision.augment import *
from fastai.vision.core import *
from fastai.data.core import *
from fastai.data.source import *
from fastai.data.pipeline import *
from fastai.data.transform import *
from fastai.core import *
from fastai.test import *
from fastai.core.imports import *
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# +
# default_exp vision.rect_augment
# default_cls_lvl 3
# -

# # Rectangular computer vision augmentation
#
# > Transforms to apply data augmentation to rectangular images

# export

# hide

# ## SortARSampler

# - resize large images
# - sort by size (size group of size n=1000//bs\*bs) and AR
# - shufflish

path = untar_data(URLs.PETS)
items = get_image_files(path / 'images')
labeller = RegexLabeller(pat=r'/([^/]+)_\d+.jpg$')
split_idx = RandomSplitter()(items)
tfms = [PILImage.create, [labeller, Categorize()]]
tds = TfmdDS(items, tfms)
im = tds[0][0]
im.shape


# export
class SortARSampler(BatchSampler):
    def __init__(self, ds, items=None, bs=32, grp_sz=1000, shuffle=False, drop_last=False):
        if not items:
            items = ds.items
        self.shapes = [Image.open(it).shape for it in items]
        self.sizes = [h * w for h, w in self.shapes]
        self.ars = [h / w for h, w in self.shapes]
        self.ds, self.grp_sz, self.bs, self.shuffle, self.drop_last = ds, round_multiple(grp_sz, bs), bs, shuffle, drop_last
        self.grp_sz = round_multiple(grp_sz, bs)

        # reverse argsort of sizes
        idxs = [i for i, o in sorted(enumerate(self.sizes), key=itemgetter(1), reverse=True)]
        # create approx equal sized groups no larger than `grp_sz`
        grps = [idxs[i:i + self.grp_sz] for i in range(0, len(idxs), self.grp_sz)]
        # sort within groups by aspect ratio
        self.grps = [sorted(g, key=lambda o:self.ars[o]) for g in grps]

    def __iter__(self):
        grps = self.grps
        if self.shuffle:
            grps = [shufflish(o) for o in grps]
        grps = [g[i:i + self.bs] for g in grps for i in range(0, len(g), self.bs)]
        if self.drop_last and len(grps[-1]) != self.bs:
            del(grps[-1])
        # Shuffle all but first (so can have largest first)
        if self.shuffle:
            grps = random.sample(grps[1:], len(grps) - 1) + [grps[0]]
        return iter(grps)

    def __len__(self): return (len(self.ds) if self.drop_last else (len(self.ds) + self.bs - 1)) // self.bs


samp = SortARSampler(tds, shuffle=False)
test_eq(len(samp), (len(tds) - 1) // 32 + 1)

itr = iter(samp)
first = next(itr)
i = 1
for last in itr:
    i += 1
test_eq(len(samp), i)
first = [tds[i][0] for i in first]
last = [tds[i][0] for i in last]
# big images are first, smaller images last
assert np.mean([im.n_px for im in last]) * 5 < np.mean([im.n_px for im in first])
# Higher aspect ratios are first
assert np.mean([im.aspect for im in last]) * 2 < np.mean([im.aspect for im in first])
# In a batch with similar aspect ratio
assert np.std([im.aspect for im in first]) < 0.1
assert np.std([im.aspect for im in last]) < 0.1

samp = SortARSampler(tds, shuffle=True)
itr = iter(samp)
first = next(itr)
for last in itr:
    pass
first = [tds[i][0] for i in first]
last = [tds[i][0] for i in last]
# In a batch with similar aspect ratio
assert np.std([im.aspect for im in first]) < 0.1
assert np.std([im.aspect for im in last]) < 0.1


# ## ResizeCollate

# export
class ResizeCollate(TfmdCollate):
    def __init__(self, tfms=None, collate_fn=default_collate, sz=None, is_fixed_px=False, max_px=512 * 512, round_mult=None,
                 rand_min_scale=None, rand_ratio_pct=None):
        super().__init__(tfms, collate_fn)
        self.round_mult, self.is_fixed_px, self.max_px = round_mult, is_fixed_px, max_px
        self.is_rand = rand_min_scale or rand_ratio_pct
        if self.is_rand:
            self.inv_ratio = 1 - ifnone(rand_ratio_pct, 0.10)
            self.resize = RandomResizedCrop(1, min_scale=ifnone(rand_min_scale, 0.25), as_item=False)
        else:
            self.resize = Resize(1, as_item=False)
        self.sz = None if sz is None else (sz, sz) if isinstance(sz, int) else sz

    def __call__(self, samples):
        if self.sz is None:
            if self.is_fixed_px:
                px = self.max_px
            else:
                px = min(self.max_px, max(L(o[0].shape[0] * o[0].shape[1] for o in samples)))
            ar = np.median(L(o[0].aspect for o in samples))
            sz = int(math.sqrt(px * ar)), int(math.sqrt(px / ar))
        else:
            sz, ar = self.sz, self.sz[1] / self.sz[0]
        if self.round_mult is not None:
            sz = round_multiple(sz, self.round_mult, round_down=True)
        if self.is_rand:
            self.resize.ratio = (ar * self.inv_ratio, ar / self.inv_ratio)
        return super().__call__(self.resize(o, size=sz) for o in samples)


# +
samp = SortARSampler(tds, shuffle=True, bs=16)
collate_fn = ResizeCollate(max_px=10000)
tdl = TfmdDL(tds, batch_sampler=samp, collate_fn=collate_fn, num_workers=0)
batch = tdl.one_batch()

test_eq(L(batch).map(type), (TensorImage, Tensor))
b, c, h, w = batch[0].shape
assert 9000 < h * w <= 10000
test_eq(b, 16)
# -

collate_fn = ResizeCollate(is_fixed_px=True, max_px=500000)
tdl = TfmdDL(tds, batch_sampler=samp, collate_fn=collate_fn, num_workers=0)
batch = tdl.one_batch()
b, c, h, w = batch[0].shape
assert 490000 < h * w <= 500000

collate_fn = ResizeCollate(sz=128)
tdl = TfmdDL(tds, batch_sampler=samp, collate_fn=collate_fn, num_workers=0)
batch = tdl.one_batch()
test_eq(batch[0].shape[2:], [128, 128])

collate_fn = ResizeCollate(round_mult=32, max_px=10000)
tdl = TfmdDL(tds, batch_sampler=samp, collate_fn=collate_fn, num_workers=0)
batch = tdl.one_batch()
b, c, h, w = batch[0].shape
test_eq(h % 32, 0)
test_eq(w % 32, 0)
assert h * w <= 10000

collate_fn = ResizeCollate(sz=128, rand_min_scale=0.7)
tdl = TfmdDL(tds, batch_sampler=samp, collate_fn=collate_fn, num_workers=0)
_, axs = plt.subplots(3, 3, figsize=(9, 9))
tdl.show_batch(ctxs=axs.flatten())

collate_fn = ResizeCollate(rand_min_scale=0.25, rand_ratio_pct=0.3)
tdl = TfmdDL(tds, batch_sampler=samp, collate_fn=collate_fn, num_workers=0)
_, axs = plt.subplots(3, 3, figsize=(9, 9))
tdl.show_batch(ctxs=axs.flatten())


# ### On object detect

def bb_pad_collate(samples, pad_idx=0):
    "Function that collect `samples` of labelled bboxes and adds padding with `pad_idx`."
    if isinstance(samples[0][1], int):
        return data_collate(samples)
    max_len = max([len(s[1][1]) for s in samples])
    bboxes = torch.zeros(len(samples), max_len, 4)
    labels = torch.zeros(len(samples), max_len).long() + pad_idx
    imgs = []
    for i, s in enumerate(samples):
        imgs.append(s[0][None])
        bbs, lbls = s[1]
        if not (bbs.nelement() == 0):
            bboxes[i, -len(lbls):] = bbs
            labels[i, -len(lbls):] = tensor(lbls)
    return torch.cat(imgs, 0), (bboxes, labels)


path = untar_data(URLs.PASCAL_2007)

path.ls()

images, lbl_bbox = get_annotations(path / 'train.json')
img2bbox = dict(zip(images, lbl_bbox))
def _pascal_lbl(o): return BBox.create(img2bbox[o.name])

items = [path / 'train' / fn for fn in images]

pascal_tds = TfmdDS(items, [PILImage.create, [_pascal_lbl, BBoxCategorize()]], item_tfms=[BBoxScaler()])

pascal_tds[0]

collate_fn = ResizeCollate(rand_min_scale=0.25, rand_ratio_pct=0.3, collate_fn=bb_pad_collate)
tdl = TfmdDL(pascal_tds, batch_sampler=samp, collate_fn=collate_fn, num_workers=0)
_, axs = plt.subplots(3, 3, figsize=(9, 9))
tdl.show_batch(ctxs=axs.flatten())

# ## Rect training (not working well)

# For a rectangular training, we change the dataset transforms to use the flip only. We will resize the images when it's time to batch them only.

# +
#img_tfms = [FlipItem(0.5)]
#tfms = [PILImage.create, [parent_label, Categorize()]]
#dsets = Datasets(items, tfms, splits=split_idx, item_tfms=img_tfms)

#tfms = [Cuda(), IntToFloatTensor(), Normalize(*imagenet_stats)]
#bs = 64
# -

# We use a sampler that will group the images by batches of the close size and aspect ratio (with a bit of shuffle for the training set) and a collation function that will resize them to the mdeian aspect ratio and median number of pixel (bound by `max_px`). `rand_min_scale` is used to do a `RandomResizedCrop` to that size on the training set.

# +
#samp = SortARSampler(dsets.train, shuffle=True, bs=bs)
#collate_fn = ResizeCollate(max_px=128*128, rand_min_scale=0.35, rand_ratio_pct=0.33, round_mult=32)
#train_dl = TfmdDL(dsets.train, tfms, num_workers=8, batch_sampler=samp, collate_fn=collate_fn)

#samp = SortARSampler(dsets.valid, shuffle=False, bs=bs)
#collate_fn = ResizeCollate(max_px=128*128, round_mult=32)
#valid_dl = TfmdDL(dsets.valid, tfms, num_workers=8, batch_sampler=samp, collate_fn=collate_fn)
# -

# Then we create a `DataLoaders` with those two dataloaders.

# +
#dls1 = imagenette.dataloaders(source, bs=64, num_workers=8, item_tfms=item_img_tfms, batch_tfms=Normalize(*imagenet_stats))

#dls = DataLoaders(train_dl, valid_dl)
# dls.show_batch(max_n=9)

# +
#learn = cnn_learner(xresnet18, dls, LabelSmoothingCrossEntropy(), opt_func=opt_func, c_in=3, c_out=10, lr=1e-2, metrics=accuracy)
# learn.fit_one_cycle(1)
# -



# ## Export -

# hide
notebook2script()
