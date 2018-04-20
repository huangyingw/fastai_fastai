
# coding: utf-8

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


from fastai.conv_learner import *
from fastai.dataset import *

from pathlib import Path
import json
from PIL import ImageDraw, ImageFont
from matplotlib import patches, patheffects
torch.cuda.set_device(3)


# ## Pascal VOC

# We will be looking at the [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) dataset. It's quite slow, so you may prefer to download from [this mirror](https://pjreddie.com/projects/pascal-voc-dataset-mirror/). There are two different competition/research datasets, from 2007 and 2012. We'll be using the 2007 version. You can use the larger 2012 for better results, or even combine them (but be careful to avoid data leakage between the validation sets if you do this).
#
# Unlike previous lessons, we are using the python 3 standard library `pathlib` for our paths and file access. Note that it returns an OS-specific class (on Linux, `PosixPath`) so your output may look a little different. Most libraries than take paths as input can take a pathlib object - although some (like `cv2`) can't, in which case you can use `str()` to convert it to a string.

PATH = Path('data/pascal')
list(PATH.iterdir())


# As well as the images, there are also *annotations* - *bounding boxes* showing where each object is. These were hand labeled. The original version were in XML, which is a little hard to work with nowadays, so we uses the more recent JSON version which you can download from [this link](https://storage.googleapis.com/coco-dataset/external/PASCAL_VOC.zip).
#
# You can see here how `pathlib` includes the ability to open files (amongst many other capabilities).

trn_j = json.load((PATH / 'pascal_train2007.json').open())
trn_j.keys()


IMAGES, ANNOTATIONS, CATEGORIES = ['images', 'annotations', 'categories']
trn_j[IMAGES][:5]


trn_j[ANNOTATIONS][:2]


trn_j[CATEGORIES][:4]


# It's helpful to use constants instead of strings, since we get tab-completion and don't mistype.

FILE_NAME, ID, IMG_ID, CAT_ID, BBOX = 'file_name', 'id', 'image_id', 'category_id', 'bbox'

cats = {o[ID]: o['name'] for o in trn_j[CATEGORIES]}
trn_fns = {o[ID]: o[FILE_NAME] for o in trn_j[IMAGES]}
trn_ids = [o[ID] for o in trn_j[IMAGES]]


list((PATH / 'VOCdevkit' / 'VOC2007').iterdir())


JPEGS = 'VOCdevkit/VOC2007/JPEGImages'


IMG_PATH = PATH / JPEGS
list(IMG_PATH.iterdir())[:5]


# Each image has a unique ID.

im0_d = trn_j[IMAGES][0]
im0_d[FILE_NAME], im0_d[ID]


# A `defaultdict` is useful any time you want to have a default dictionary entry for new keys. Here we create a dict from image IDs to a list of annotations (tuple of bounding box and class id).
#
# We convert VOC's height/width into top-left/bottom-right, and switch x/y coords to be consistent with numpy.

def hw_bb(bb): return np.array([bb[1], bb[0], bb[3] + bb[1] - 1, bb[2] + bb[0] - 1])

trn_anno = collections.defaultdict(lambda: [])
for o in trn_j[ANNOTATIONS]:
    if not o['ignore']:
        bb = o[BBOX]
        bb = hw_bb(bb)
        trn_anno[o[IMG_ID]].append((bb, o[CAT_ID]))
        
len(trn_anno)


im_a = trn_anno[im0_d[ID]]; im_a


im0_a = im_a[0]; im0_a


cats[7]


trn_anno[17]


cats[15], cats[13]


# Some libs take VOC format bounding boxes, so this let's us convert back when required:

bb_voc = [155, 96, 196, 174]
bb_fastai = hw_bb(bb_voc)


def bb_hw(a): return np.array([a[1], a[0], a[3] - a[1] + 1, a[2] - a[0] + 1])


f'expected: {bb_voc}, actual: {bb_hw(bb_fastai)}'


# You can use [Visual Studio Code](https://code.visualstudio.com/) (vscode - open source editor that comes with recent versions of Anaconda, or can be installed separately), or most editors and IDEs, to find out all about the `open_image` function. vscode things to know:
#
# - Command palette (<kbd>Ctrl-shift-p</kbd>)
# - Select interpreter (for fastai env)
# - Select terminal shell
# - Go to symbol (<kbd>Ctrl-t</kbd>)
# - Find references (<kbd>Shift-F12</kbd>)
# - Go to definition (<kbd>F12</kbd>)
# - Go back (<kbd>alt-left</kbd>)
# - View documentation
# - Hide sidebar (<kbd>Ctrl-b</kbd>)
# - Zen mode (<kbd>Ctrl-k,z</kbd>)

im = open_image(IMG_PATH / im0_d[FILE_NAME])


# Matplotlib's `plt.subplots` is a really useful wrapper for creating plots, regardless of whether you have more than one subplot. Note that Matplotlib has an optional object-oriented API which I think is much easier to understand and use (although few examples online use it!)

def show_img(im, figsize=None, ax=None):
    if not ax: fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax


# A simple but rarely used trick to making text visible regardless of background is to use white text with black outline, or visa versa. Here's how to do it in matplotlib.

def draw_outline(o, lw):
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])


# Note that `*` in argument lists is the [splat operator](https://stackoverflow.com/questions/5239856/foggy-on-asterisk-in-python). In this case it's a little shortcut compared to writing out `b[-2],b[-1]`.

def draw_rect(ax, b):
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor='white', lw=2))
    draw_outline(patch, 4)


def draw_text(ax, xy, txt, sz=14):
    text = ax.text(*xy, txt,
        verticalalignment='top', color='white', fontsize=sz, weight='bold')
    draw_outline(text, 1)


ax = show_img(im)
b = bb_hw(im0_a[0])
draw_rect(ax, b)
draw_text(ax, b[:2], cats[im0_a[1]])


def draw_im(im, ann):
    ax = show_img(im, figsize=(16, 8))
    for b, c in ann:
        b = bb_hw(b)
        draw_rect(ax, b)
        draw_text(ax, b[:2], cats[c], sz=16)


def draw_idx(i):
    im_a = trn_anno[i]
    im = open_image(IMG_PATH / trn_fns[i])
    print(im.shape)
    draw_im(im, im_a)


draw_idx(17)


# ## Largest item classifier

# A *lambda function* is simply a way to define an anonymous function inline. Here we use it to describe how to sort the annotation for each image - by bounding box size (descending).

def get_lrg(b):
    if not b: raise Exception()
    b = sorted(b, key=lambda x: np.product(x[0][-2:] - x[0][:2]), reverse=True)
    return b[0]


trn_lrg_anno = {a: get_lrg(b) for a, b in trn_anno.items()}


# Now we have a dictionary from image id to a single bounding box - the largest for that image.

b, c = trn_lrg_anno[23]
b = bb_hw(b)
ax = show_img(open_image(IMG_PATH / trn_fns[23]), figsize=(5, 10))
draw_rect(ax, b)
draw_text(ax, b[:2], cats[c], sz=16)


(PATH / 'tmp').mkdir(exist_ok=True)
CSV = PATH / 'tmp/lrg.csv'


# Often it's easiest to simply create a CSV of the data you want to model, rather than trying to create a custom dataset. Here we use Pandas to help us create a CSV of the image filename and class.

df = pd.DataFrame({'fn': [trn_fns[o] for o in trn_ids],
    'cat': [cats[trn_lrg_anno[o][1]] for o in trn_ids]}, columns=['fn', 'cat'])
df.to_csv(CSV, index=False)


f_model = resnet34
sz = 224
bs = 64


# From here it's just like Dogs vs Cats!

tfms = tfms_from_model(f_model, sz, aug_tfms=transforms_side_on, crop_type=CropType.NO)
md = ImageClassifierData.from_csv(PATH, JPEGS, CSV, tfms=tfms)


x, y = next(iter(md.val_dl))


show_img(md.val_ds.denorm(to_np(x))[0]);


learn = ConvLearner.pretrained(f_model, md, metrics=[accuracy])
learn.opt_fn = optim.Adam


lrf = learn.lr_find(1e-5, 100)


# When you LR finder graph looks like this, you can ask for more points on each end:

learn.sched.plot()


learn.sched.plot(n_skip=5, n_skip_end=1)


lr = 2e-2


learn.fit(lr, 1, cycle_len=1)


lrs = np.array([lr / 1000, lr / 100, lr])


learn.freeze_to(-2)


lrf = learn.lr_find(lrs / 1000)
learn.sched.plot(1)


learn.fit(lrs / 5, 1, cycle_len=1)


learn.unfreeze()


# Accuracy isn't improving much - since many images have multiple different objects, it's going to be impossible to be that accurate.

learn.fit(lrs / 5, 1, cycle_len=2)


learn.save('clas_one')


learn.load('clas_one')


x, y = next(iter(md.val_dl))
probs = F.softmax(predict_batch(learn.model, x), -1)
x, preds = to_np(x), to_np(probs)
preds = np.argmax(preds, -1)


# You can use the python debugger `pdb` to step through code.
#
# - `pdb.set_trace()` to set a breakpoint
# - `%debug` magic to trace an error
#
# Commands you need to know:
#
# - s / n / c
# - u / d
# - p
# - l

fig, axes = plt.subplots(3, 4, figsize=(12, 8))
for i, ax in enumerate(axes.flat):
    ima = md.val_ds.denorm(x)[i]
    b = md.classes[preds[i]]
    ax = show_img(ima, ax=ax)
    draw_text(ax, (0, 0), b)
plt.tight_layout()


# It's doing a pretty good job of classifying the largest object!

# ## Bbox only

# Now we'll try to find the bounding box of the largest object. This is simply a regression with 4 outputs. So we can use a CSV with multiple 'labels'.

BB_CSV = PATH / 'tmp/bb.csv'


bb = np.array([trn_lrg_anno[o][0] for o in trn_ids])
bbs = [' '.join(str(p) for p in o) for o in bb]

df = pd.DataFrame({'fn': [trn_fns[o] for o in trn_ids], 'bbox': bbs}, columns=['fn', 'bbox'])
df.to_csv(BB_CSV, index=False)


BB_CSV.open().readlines()[:5]


f_model = resnet34
sz = 224
bs = 64


# Set `continuous=True` to tell fastai this is a regression problem, which means it won't one-hot encode the labels, and will use MSE as the default crit.
#
# Note that we have to tell the transforms constructor that our labels are coordinates, so that it can handle the transforms correctly.
#
# Also, we use CropType.NO because we want to 'squish' the rectangular images into squares, rather than center cropping, so that we don't accidentally crop out some of the objects. (This is less of an issue in something like imagenet, where there is a single object to classify, and it's generally large and centrally located).

augs = [RandomFlip(),
        RandomRotate(30),
        RandomLighting(0.1, 0.1)]


tfms = tfms_from_model(f_model, sz, crop_type=CropType.NO, aug_tfms=augs)
md = ImageClassifierData.from_csv(PATH, JPEGS, BB_CSV, tfms=tfms, continuous=True, bs=4)


idx = 3
fig, axes = plt.subplots(3, 3, figsize=(9, 9))
for i, ax in enumerate(axes.flat):
    x, y = next(iter(md.aug_dl))
    ima = md.val_ds.denorm(to_np(x))[idx]
    b = bb_hw(to_np(y[idx]))
    print(b)
    show_img(ima, ax=ax)
    draw_rect(ax, b)


augs = [RandomFlip(tfm_y=TfmType.COORD),
        RandomRotate(30, tfm_y=TfmType.COORD),
        RandomLighting(0.1, 0.1, tfm_y=TfmType.COORD)]


tfms = tfms_from_model(f_model, sz, crop_type=CropType.NO, tfm_y=TfmType.COORD, aug_tfms=augs)
md = ImageClassifierData.from_csv(PATH, JPEGS, BB_CSV, tfms=tfms, continuous=True, bs=4)


idx = 3
fig, axes = plt.subplots(3, 3, figsize=(9, 9))
for i, ax in enumerate(axes.flat):
    x, y = next(iter(md.aug_dl))
    ima = md.val_ds.denorm(to_np(x))[idx]
    b = bb_hw(to_np(y[idx]))
    print(b)
    show_img(ima, ax=ax)
    draw_rect(ax, b)


tfm_y = TfmType.COORD
augs = [RandomFlip(tfm_y=tfm_y),
        RandomRotate(3, p=0.5, tfm_y=tfm_y),
        RandomLighting(0.05, 0.05, tfm_y=tfm_y)]

tfms = tfms_from_model(f_model, sz, crop_type=CropType.NO, tfm_y=tfm_y, aug_tfms=augs)
md = ImageClassifierData.from_csv(PATH, JPEGS, BB_CSV, tfms=tfms, continuous=True)


# fastai let's you use a `custom_head` to add your own module on top of a convnet, instead of the adaptive pooling and fully connected net which is added by default. In this case, we don't want to do any pooling, since we need to know the activations of each grid cell.
#
# The final layer has 4 activations, one per bounding box coordinate. Our target is continuous, not categorical, so the MSE loss function used does not do any sigmoid or softmax to the module outputs.

512 * 7 * 7


head_reg4 = nn.Sequential(Flatten(), nn.Linear(25088, 4))
learn = ConvLearner.pretrained(f_model, md, custom_head=head_reg4)
learn.opt_fn = optim.Adam
learn.crit = nn.L1Loss()


learn.summary()


learn.lr_find(1e-5, 100)
learn.sched.plot(5)


lr = 2e-3


learn.fit(lr, 2, cycle_len=1, cycle_mult=2)


lrs = np.array([lr / 100, lr / 10, lr])


learn.freeze_to(-2)


lrf = learn.lr_find(lrs / 1000)
learn.sched.plot(1)


learn.fit(lrs, 2, cycle_len=1, cycle_mult=2)


learn.freeze_to(-3)


learn.fit(lrs, 1, cycle_len=2)


learn.save('reg4')


learn.load('reg4')


x, y = next(iter(md.val_dl))
learn.model.eval()
preds = to_np(learn.model(VV(x)))


fig, axes = plt.subplots(3, 4, figsize=(12, 8))
for i, ax in enumerate(axes.flat):
    ima = md.val_ds.denorm(to_np(x))[i]
    b = bb_hw(preds[i])
    ax = show_img(ima, ax=ax)
    draw_rect(ax, b)
plt.tight_layout()


# ## Single object detection

f_model = resnet34
sz = 224
bs = 64

val_idxs = get_cv_idxs(len(trn_fns))


tfms = tfms_from_model(f_model, sz, crop_type=CropType.NO, tfm_y=TfmType.COORD, aug_tfms=augs)
md = ImageClassifierData.from_csv(PATH, JPEGS, BB_CSV, tfms=tfms,
    continuous=True, val_idxs=val_idxs)


md2 = ImageClassifierData.from_csv(PATH, JPEGS, CSV, tfms=tfms_from_model(f_model, sz))


# A dataset can be anything with `__len__` and `__getitem__`. Here's a dataset that adds a 2nd label to an existing dataset:

class ConcatLblDataset(Dataset):
    def __init__(self, ds, y2): self.ds, self.y2 = ds, y2

    def __len__(self): return len(self.ds)
    
    def __getitem__(self, i):
        x, y = self.ds[i]
        return (x, (y, self.y2[i]))


# We'll use it to add the classes to the bounding boxes labels.

trn_ds2 = ConcatLblDataset(md.trn_ds, md2.trn_y)
val_ds2 = ConcatLblDataset(md.val_ds, md2.val_y)


val_ds2[0][1]


# We can replace the dataloaders' datasets with these new ones.

md.trn_dl.dataset = trn_ds2
md.val_dl.dataset = val_ds2


# We have to `denorm`alize the images from the dataloader before they can be plotted.

x, y = next(iter(md.val_dl))
idx = 3
ima = md.val_ds.ds.denorm(to_np(x))[idx]
b = bb_hw(to_np(y[0][idx])); b


ax = show_img(ima)
draw_rect(ax, b)
draw_text(ax, b[:2], md2.classes[y[1][idx]])


# We need one output activation for each class (for its probability) plus one for each bounding box coordinate. We'll use an extra linear layer this time, plus some dropout, to help us train a more flexible model.

head_reg4 = nn.Sequential(
    Flatten(),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(25088, 256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Dropout(0.5),
    nn.Linear(256, 4 + len(cats)),
)
models = ConvnetBuilder(f_model, 0, 0, 0, custom_head=head_reg4)

learn = ConvLearner(md, models)
learn.opt_fn = optim.Adam


def detn_loss(input, target):
    bb_t, c_t = target
    bb_i, c_i = input[:, :4], input[:, 4:]
    bb_i = F.sigmoid(bb_i) * 224
    # I looked at these quantities separately first then picked a multiplier
    #   to make them approximately equal
    return F.l1_loss(bb_i, bb_t) + F.cross_entropy(c_i, c_t) * 20

def detn_l1(input, target):
    bb_t, _ = target
    bb_i = input[:, :4]
    bb_i = F.sigmoid(bb_i) * 224
    return F.l1_loss(V(bb_i), V(bb_t)).data

def detn_acc(input, target):
    _, c_t = target
    c_i = input[:, 4:]
    return accuracy(c_i, c_t)

learn.crit = detn_loss
learn.metrics = [detn_acc, detn_l1]


learn.lr_find()
learn.sched.plot()


lr = 1e-2


learn.fit(lr, 1, cycle_len=3, use_clr=(32, 5))


learn.save('reg1_0')


learn.freeze_to(-2)


lrs = np.array([lr / 100, lr / 10, lr])


learn.lr_find(lrs / 1000)
learn.sched.plot(0)


learn.fit(lrs / 5, 1, cycle_len=5, use_clr=(32, 10))


learn.save('reg1_1')


learn.load('reg1_1')


learn.unfreeze()


learn.fit(lrs / 10, 1, cycle_len=10, use_clr=(32, 10))


learn.save('reg1')


learn.load('reg1')


y = learn.predict()
x, _ = next(iter(md.val_dl))


from scipy.special import expit


fig, axes = plt.subplots(3, 4, figsize=(12, 8))
for i, ax in enumerate(axes.flat):
    ima = md.val_ds.ds.denorm(to_np(x))[i]
    bb = expit(y[i][:4]) * 224
    b = bb_hw(bb)
    c = np.argmax(y[i][4:])
    ax = show_img(ima, ax=ax)
    draw_rect(ax, b)
    draw_text(ax, b[:2], md2.classes[c])
plt.tight_layout()


# ## End
