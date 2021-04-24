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

# ## Super resolution

from fastai.vision.all import *

path = untar_data(URLs.PETS)
path_hr = path / 'images'
path_lr = path / 'small-96'
path_mr = path / 'small-256'

items = get_image_files(path_hr)


def resize_one(fn, path, size):
    dest = path / fn.relative_to(path_hr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(fn)
    targ_sz = resize_to(img, size, use_min=True)
    img = img.resize(targ_sz, resample=Image.BILINEAR).convert('RGB')
    img.save(dest, quality=60)


# create smaller image sets the first time this nb is run
sets = [(path_lr, 96), (path_mr, 256)]
for p, size in sets:
    if not p.exists():
        print(f"resizing to {size} into {p}")
        parallel(partial(resize_one, path=p, size=size), items)

bs, size = 32, 128
arch = resnet34


def get_dls(bs, size):
    dblock = DataBlock(blocks=(ImageBlock, ImageBlock),
                       get_items=get_image_files,
                       get_y=lambda x: path_hr / x.name,
                       splitter=RandomSplitter(),
                       item_tfms=Resize(size),
                       batch_tfms=[*aug_transforms(max_zoom=2.), Normalize.from_stats(*imagenet_stats)])
    dls = dblock.dataloaders(path_lr, bs=bs, path=path, item_tfms=Resize(size))
    dls.c = 3
    return dls


dls = get_dls(bs, size)

dls.train.show_batch(max_n=4, figsize=(18, 9))

# ## Feature loss

t = tensor(dls.valid_ds[0][1]).float().permute(2, 0, 1) / 255.
t = torch.stack([t, t])


def gram_matrix(x):
    n, c, h, w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1, 2)) / (c * h * w)


t.shape

gram_matrix(t)

base_loss = F.l1_loss

vgg_m = vgg16_bn(True).features.cuda().eval()
vgg_m = vgg_m.requires_grad_(False)

blocks = [i - 1 for i, o in enumerate(vgg_m.children()) if isinstance(o, nn.MaxPool2d)]
blocks, [vgg_m[i] for i in blocks]


class FeatureLoss(Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel', ] + [f'feat_{i}' for i in range(len(layer_ids))
                                           ] + [f'gram_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]

    def forward(self, input, target, reduction='mean'):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [base_loss(input, target, reduction=reduction)]
        self.feat_losses += [base_loss(f_in, f_out, reduction=reduction) * w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [base_loss(gram_matrix(f_in), gram_matrix(f_out), reduction=reduction) * w**2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        if reduction == 'none':
            self.feat_losses = [f.mean(dim=[1, 2, 3]) for f in self.feat_losses[:4]] + [f.mean(dim=[1, 2]) for f in self.feat_losses[4:]]
        for n, l in zip(self.metric_names, self.feat_losses):
            setattr(self, n, l)
        return sum(self.feat_losses)

    def __del__(self): self.hooks.remove()


feat_loss = FeatureLoss(vgg_m, blocks[2:5], [5, 15, 2])

# ## Train

learn = unet_learner(dls, arch, loss_func=feat_loss, metrics=LossMetrics(feat_loss.metric_names),
                     config=unet_config(blur=True, norm_type=NormType.Weight))

learn.lr_find()

lr = 1e-3
wd = 1e-3


def do_fit(save_name, lrs=slice(lr), pct_start=0.9):
    learn.fit_one_cycle(10, lrs, pct_start=pct_start, wd=wd)
    learn.save(save_name)
    learn.show_results(ds_idx=1, max_n=2, figsize=(15, 11))


do_fit('1a', slice(lr * 10))

learn.unfreeze()

do_fit('1b', slice(1e-5, lr))

dls = get_dls(12, size * 2)

learn.dls = dls
learn.freeze()

learn.load('1b')

do_fit('2a')

learn.unfreeze()

do_fit('2b', slice(1e-6, 1e-4), pct_start=0.3)

# ## Test

learn = unet_learner(dls, arch, loss_func=feat_loss, metrics=LossMetrics(feat_loss.metric_names),
                     config=unet_config(blur=True, norm_type=NormType.Weight))

dblock = DataBlock(blocks=(ImageBlock, ImageBlock),
                   get_items=get_image_files,
                   get_y=lambda x: path_mr / x.name,
                   splitter=RandomSplitter(),
                   item_tfms=Resize(size),
                   batch_tfms=[*aug_transforms(), Normalize.from_stats(*imagenet_stats)])
dbunch_mr = dblock.dataloaders(path_mr, bs=1, val_bs=1, path=path)
dbunch_mr.c = 3

learn.load('2b')

learn.dls = dbunch_mr

fn = dbunch_mr.valid_ds.items[0]
fn

img_hr, *_ = learn.predict(fn)

img = PILImage.create(fn)
show_image(img, figsize=(18, 15), interpolation='nearest')

show_image(img_hr, figsize=(18, 15))
