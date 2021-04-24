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

# +
# all_slow
# all_cuda
# -

# # Examples of many applications

# This notebook is a quick(ish) test of most of the main application people use, taken from `fastbook`.

# hide
from fastai.vision.all import *
from fastai.tabular.all import *
from fastai.text.all import *
from fastai.collab import *

# ## Image single classification

set_seed(99, True)
path = untar_data(URLs.PETS) / 'images'
dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2,
    label_func=lambda x: x[0].isupper(), item_tfms=Resize(224))

learn = cnn_learner(dls, resnet34, metrics=error_rate).to_fp16()
learn.fine_tune(1)

# Should be around 0.01 or less.

img = PILImage.create('../images/cat.jpg')
print(f"Probability it's a cat: {learn.predict(img)[2][1].item():.6f}")

# ## Segmentation

# +
path = untar_data(URLs.CAMVID_TINY)
dls = SegmentationDataLoaders.from_label_func(
    path, bs=8, fnames=get_image_files(path / "images"),
    label_func=lambda o: path / 'labels' / f'{o.stem}_P{o.suffix}',
    codes=np.loadtxt(path / 'codes.txt', dtype=str)
)

learn = unet_learner(dls, resnet34)
# -

learn.fine_tune(8)

# Should be a bit above 0.8.

learn.show_results(max_n=6, figsize=(7, 8))

# RHS pics should be similar to LHS pics.

# ## Text classification

path = untar_data(URLs.IMDB_SAMPLE)
df = pd.read_csv(path / 'texts.csv')
imdb_clas = DataBlock(blocks=(TextBlock.from_df('text', seq_len=72), CategoryBlock),
                      get_x=ColReader('text'), get_y=ColReader('label'), splitter=ColSplitter())
dls = imdb_clas.dataloaders(df, bs=64)

learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
learn.fine_tune(4, 1e-2)

# Should be a bit under 0.8.

learn.predict("I really liked that movie!")

# Should be a bit very nearly 1.

# ## Tabular

# +
path = untar_data(URLs.ADULT_SAMPLE)

dls = TabularDataLoaders.from_csv(path / 'adult.csv', path=path, y_names="salary",
                                  cat_names=['workclass', 'education', 'marital-status', 'occupation',
                                             'relationship', 'race'],
                                  cont_names=['age', 'fnlwgt', 'education-num'],
                                  procs=[Categorify, FillMissing, Normalize])

learn = tabular_learner(dls, metrics=accuracy)
learn.fit_one_cycle(3)
# -

# Should be around 0.83

# ## Collab filtering

path = untar_data(URLs.ML_SAMPLE)
dls = CollabDataLoaders.from_csv(path / 'ratings.csv')
learn = collab_learner(dls, y_range=(0.5, 5.5))

learn.fine_tune(6)

# Should be a bit over 0.7

learn.show_results(max_n=4)

# ## Keypoints

# +
path = untar_data(URLs.BIWI_HEAD_POSE)
img_files = get_image_files(path)


def img2pose(x): return Path(f'{str(x)[:-7]}pose.txt')


cal = np.genfromtxt(path / '01' / 'rgb.cal', skip_footer=6)


def get_ctr(f):
    ctr = np.genfromtxt(img2pose(f), skip_header=3)
    c1 = ctr[0] * cal[0][0] / ctr[2] + cal[0][2]
    c2 = ctr[1] * cal[1][1] / ctr[2] + cal[1][2]
    return tensor([c1, c2])


biwi = DataBlock(
    blocks=(ImageBlock, PointBlock),
    get_items=get_image_files,
    get_y=get_ctr,
    splitter=FuncSplitter(lambda o: o.parent.name == '13'),
    batch_tfms=[*aug_transforms(size=(240, 320)),
                Normalize.from_stats(*imagenet_stats)])

dls = biwi.dataloaders(path)
dls.show_batch(max_n=9, figsize=(8, 6))
# -

learn = cnn_learner(dls, resnet18, y_range=(-1, 1))

learn.lr_find()

learn.fine_tune(1, 1e-2)

# Should be around 0.0005

learn.show_results(ds_idx=1, nrows=3, figsize=(6, 8))

# Red dots should be close to noses.

# ## fin -
