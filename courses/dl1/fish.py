
# coding: utf-8

# # Fisheries competition

# In this notebook we're going to investigate a range of different techniques for the [Kaggle fisheries competition](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring). In this competition, The Nature Conservancy asks you to help them detect which species of fish appears on a fishing boat, based on images captured from boat cameras of various angles. Your goal is to predict the likelihood of fish species in each picture. Eight target categories are available in this dataset: Albacore tuna, Bigeye tuna, Yellowfin tuna, Mahi Mahi, Opah, Sharks, Other
#
# You can use [this](https://github.com/floydwch/kaggle-cli) api to download the data from Kaggle.

# Put these at the top of every notebook, to get automatic reloading and inline plotting
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

# This file contains all the main external libs we'll use
from fastai.imports import *
from fastai.plots import *
from fastai.io import get_data

PATH = "data/fish/"


# ## First look at fish pictures

get_ipython().system('ls {PATH}')


get_ipython().system('ls {PATH}train')


files = get_ipython().getoutput('ls {PATH}train/ALB | head')
files


img = plt.imread(f'{PATH}train/ALB/{files[0]}')
plt.imshow(img);


# ## Data pre-processing

# Here we are changing the structure of the training data to make it more convinient. We will have all images in a common directory `images` and will have a file `train.csv` with all labels.

from os import listdir
from os.path import join
train_path = f'{PATH}/train'


dirs = [d for d in listdir(train_path) if os.path.isdir(join(train_path, d))]
print(dirs)


train_dict = {d: listdir(join(train_path, d)) for d in dirs}


train_dict["LAG"][:10]


sum(len(v) for v in train_dict.values())


with open(f"{PATH}train.csv", "w") as csv:
    csv.write("img,label\n")
    for d in dirs:
        for f in train_dict[d]: csv.write(f'{f},{d}\n')


img_path = f'{PATH}images'
os.makedirs(img_path, exist_ok=True)


get_ipython().system('cp {PATH}train/*/*.jpg {PATH}images/')


# ## Our first model with Center  Cropping

# Here we import the libraries we need. We'll learn about what each does during the course.

from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *


sz = 350
bs = 64
csv_fname = os.path.join(PATH, "train.csv")
train_labels = list(open(csv_fname))
n = len(list(open(csv_fname))) - 1
val_idxs = get_cv_idxs(n)


tfms = tfms_from_model(resnet34, sz)
data = ImageClassifierData.from_csv(PATH, "images", csv_fname, bs, tfms, val_idxs)


learn = ConvLearner.pretrained(resnet34, data, precompute=True, opt_fn=optim.Adam, ps=0.5)


lrf = learn.lr_find()
learn.sched.plot()


learn.fit(0.01, 4, cycle_len=1, cycle_mult=2)


lrs = np.array([1e-4, 1e-3, 1e-2])
learn.precompute = False

learn.freeze_to(6)
lrf = learn.lr_find(lrs / 1e3)
learn.sched.plot()


# ## Same model with No cropping

# NOTE: Before running this remove the temp file under data/fish.

sz = 350
tfms = tfms_from_model(resnet34, sz, crop_type=CropType.NO)
data = ImageClassifierData.from_csv(PATH, "images", csv_fname, bs, tfms, val_idxs)


learn = ConvLearner.pretrained(resnet34, data, precompute=True, opt_fn=optim.Adam, ps=0.5)


lrf = learn.lr_find()
learn.sched.plot()


learn.fit(0.01, 4, cycle_len=1, cycle_mult=2)


lrs = np.array([1e-4, 1e-3, 1e-2])
learn.precompute = False

learn.unfreeze()
lrf = learn.lr_find(lrs / 1e3)
learn.sched.plot()


lrs = np.array([1e-5, 1e-4, 1e-3])
learn.fit(lrs, 5, cycle_len=1, cycle_mult=2)


# ## Predicting bounding boxes

# ### Getting bounding boxes data

# This part needs to run just the first time to get the file `trn_bb_labels`

import json
anno_classes = ['alb', 'bet', 'dol', 'lag', 'other', 'shark', 'yft']


def get_annotations():
    annot_urls = [
        '5458/bet_labels.json', '5459/shark_labels.json', '5460/dol_labels.json',
        '5461/yft_labels.json', '5462/alb_labels.json', '5463/lag_labels.json'
    ]
    cache_subdir = os.path.abspath(os.path.join(PATH, 'annos'))
    url_prefix = 'https://kaggle2.blob.core.windows.net/forum-message-attachments/147157/'
    os.makedirs(cache_subdir, exist_ok=True)
    
    for url_suffix in annot_urls:
        fname = url_suffix.rsplit('/', 1)[-1]
        get_data(url_prefix + url_suffix, f'{cache_subdir}/{fname}')


# run this code to get annotation files
get_annotations()


# creates a dictionary of all annotations per file
bb_json = {}
for c in anno_classes:
    if c == 'other': continue # no annotation file for "other" class
    j = json.load(open(f'{PATH}annos/{c}_labels.json', 'r'))
    for l in j:
        if 'annotations' in l.keys() and len(l['annotations']) > 0:
            bb_json[l['filename'].split('/')[-1]] = sorted(
                l['annotations'], key=lambda x: x['height'] * x['width'])[-1]
bb_json['img_04908.jpg']


raw_filenames = pd.read_csv(csv_fname)["img"].values


file2idx = {o: i for i, o in enumerate(raw_filenames)}

empty_bbox = {'height': 0., 'width': 0., 'x': 0., 'y': 0.}
for f in raw_filenames:
    if not f in bb_json.keys(): bb_json[f] = empty_bbox

bb_params = ['height', 'width', 'x', 'y']
def convert_bb(bb):
    bb = [bb[p] for p in bb_params]
    bb[2] = max(bb[2], 0)
    bb[3] = max(bb[3], 0)
    return bb


trn_bbox = np.stack([convert_bb(bb_json[f]) for f in raw_filenames]).astype(np.float32)
trn_bb_labels = [f + ',' + ' '.join(map(str, o)) + '\n' for f, o in zip(raw_filenames, trn_bbox)]


open(f'{PATH}trn_bb_labels', 'w').writelines(trn_bb_labels)


fnames, csv_labels, _, _ = parse_csv_labels(f'{PATH}trn_bb_labels', skip_header=False)


def bb_corners(bb):
    bb = np.array(bb, dtype=np.float32)
    row1 = bb[3]
    col1 = bb[2]
    row2 = row1 + bb[0]
    col2 = col1 + bb[1]
    return [row1, col1, row2, col2]


f = 'img_02642.jpg'
bb = csv_labels[f]
print(bb)
bb_corners(bb)


new_labels = [f + "," + " ".join(map(str, bb_corners(csv_labels[f]))) + "\n" for f in raw_filenames]


open(f'{PATH}trn_bb_corners_labels', 'w').writelines(new_labels)


# ### Looking at bounding boxes

# reading bb file
bbox = {}
bb_data = pd.read_csv(f'{PATH}trn_bb_labels', header=None)


fnames, csv_labels, _, _ = parse_csv_labels(f'{PATH}trn_bb_labels', skip_header=False)
fnames, corner_labels, _, _ = parse_csv_labels(f'{PATH}trn_bb_corners_labels', skip_header=False)


corner_labels["img_06297.jpg"]


csv_labels["img_06297.jpg"]


def create_rect(bb, color='red'):
    return plt.Rectangle((bb[2], bb[3]), bb[1], bb[0], color=color, fill=False, lw=3)

def show_bb(path, f='img_04908.jpg'):
    file_path = f'{path}images/{f}'
    bb = csv_labels[f]
    plots_from_files([file_path])
    plt.gca().add_patch(create_rect(bb))


def create_corner_rect(bb, color='red'):
    bb = np.array(bb, dtype=np.float32)
    return plt.Rectangle((bb[1], bb[0]), bb[3] - bb[1], bb[2] - bb[0], color=color, fill=False, lw=3)

def show_corner_bb(path, f='img_04908.jpg'):
    file_path = f'{path}images/{f}'
    bb = corner_labels[f]
    plots_from_files([file_path])
    plt.gca().add_patch(create_corner_rect(bb))


show_corner_bb(PATH, f='img_02642.jpg')


# ### Model predicting bounding boxes

sz = 299
bs = 64

label_csv = f'{PATH}trn_bb_corners_labels'
n = len(list(open(label_csv))) - 1
val_idxs = get_cv_idxs(n)


tfms = tfms_from_model(resnet34, sz, crop_type=CropType.NO, tfm_y=TfmType.COORD)
data = ImageClassifierData.from_csv(PATH, 'images', label_csv, tfms=tfms, val_idxs=val_idxs,
                                    continuous=True, skip_header=False)


trn_ds = data.trn_dl.dataset


x, y = trn_ds[0]


print(x.shape, y)


learn = ConvLearner.pretrained(resnet34, data, precompute=True, opt_fn=optim.Adam, ps=0.5)


lrf = learn.lr_find()
learn.sched.plot()


learn.fit(0.01, 5, cycle_len=1, cycle_mult=2)


lrs = np.array([1e-4, 1e-3, 1e-2])
learn.precompute = False

learn.unfreeze()
lrf = learn.lr_find(lrs / 1e3)
learn.sched.plot()


lrs = np.array([1e-5, 1e-4, 1e-3])
learn.fit(lrs, 5, cycle_len=1, cycle_mult=2)


# ## Looking into size of images

f = "img_06297.jpg"
PIL.Image.open(PATH + "images/" + f).size


sizes = [PIL.Image.open(PATH + f).size for f in data.trn_ds.fnames]
raw_val_sizes = [PIL.Image.open(PATH + f).size for f in data.val_ds.fnames]
