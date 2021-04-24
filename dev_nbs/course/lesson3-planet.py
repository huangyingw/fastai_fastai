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

# ## Multi-label prediction with Planet Amazon dataset

# %matplotlib inline

from fastai.vision.all import *
from nbdev.showdoc import *

# ## Getting the data

# The planet dataset isn't available on the [fastai dataset page](https://course.fast.ai/datasets) due to copyright restrictions. You can download it from Kaggle however. Let's see how to do this by using the [Kaggle API](https://github.com/Kaggle/kaggle-api) as it's going to be pretty useful to you if you want to join a competition or use other Kaggle datasets later on.
#
# First, install the Kaggle API by uncommenting the following line and executing it, or by executing it in your terminal (depending on your platform you may need to modify this slightly to either add `source activate fastai` or similar, or prefix `pip` with a path. Have a look at how `conda install` is called for your platform in the appropriate *Returning to work* section of https://course.fast.ai/. (Depending on your environment, you may also need to append "--user" to the command.)

# +
# ! {sys.executable} -m pip install kaggle --upgrade
# -

# Then you need to upload your credentials from Kaggle on your instance. Login to kaggle and click on your profile picture on the top left corner, then 'My account'. Scroll down until you find a button named 'Create New API Token' and click on it. This will trigger the download of a file named 'kaggle.json'.
#
# Upload this file to the directory this notebook is running in, by clicking "Upload" on your main Jupyter page, then uncomment and execute the next two commands (or run them in a terminal). For Windows, uncomment the last two commands.

# +
# # ! mkdir -p ~/.kaggle/
# # ! mv kaggle.json ~/.kaggle/

# For Windows, uncomment these two commands
# # ! mkdir %userprofile%\.kaggle
# # ! move kaggle.json %userprofile%\.kaggle
# -

# You're all set to download the data from [planet competition](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space). You **first need to go to its main page and accept its rules**, and run the two cells below (uncomment the shell commands to download and unzip the data). If you get a `403 forbidden` error it means you haven't accepted the competition rules yet (you have to go to the competition page, click on *Rules* tab, and then scroll to the bottom to find the *accept* button).

path = Config().data / 'planet'
path.mkdir(parents=True, exist_ok=True)
path

# +
# #! kaggle competitions download -c planet-understanding-the-amazon-from-space -f train-jpg.tar.7z -p {path}
# #! kaggle competitions download -c planet-understanding-the-amazon-from-space -f train_v2.csv -p {path}
# #! unzip -q -n {path}/train_v2.csv.zip -d {path}
# -

# To extract the content of this file, we'll need 7zip, so uncomment the following line if you need to install it (or run `sudo apt install p7zip-full` in your terminal).

# +
# # ! conda install --yes --prefix {sys.prefix} -c haasad eidl7zip
# -

# And now we can unpack the data (uncomment to run - this might take a few minutes to complete).

# +
#! 7za -bd -y -so x {path}/train-jpg.tar.7z | tar xf - -C {path.as_posix()}
# -

# ## Multiclassification

# Contrary to the pets dataset studied in last lesson, here each picture can have multiple labels. If we take a look at the csv file containing the labels (in 'train_v2.csv' here) we see that each 'image_name' is associated to several tags separated by spaces.

df = pd.read_csv(path / 'train_v2.csv')
df.head()

# To put this in a `DataLoaders` while using the [data block API](https://docs.fast.ai/data_block.html), to do this we need to indicate:
# - the types of our inputs/targets (here image and multi-label categories) through a thing called blocks
# - how to get our xs and ys from the dataframe through a `ColReader`
# - how to split out data between training and validation

# Since we have satellite images, it makes sense to use all kinds of flip, we limit the amount of lighting/zoom and remove the warping.

tfms = aug_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0., size=128)

planet = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   get_x=ColReader(0, pref=str(path / "train-jpg") + "/", suff='.jpg'),
                   get_y=ColReader(1, label_delim=' '),
                   splitter=RandomSplitter(seed=42),
                   batch_tfms=tfms + [Normalize.from_stats(*imagenet_stats)])

# Since we have satellite images, it makes sense to use all kinds of flip, we limit the amount of lighting/zoom and remove the warping.

dls = planet.dataloaders(df, bs=64, path=path)

# `show_batch` still works, and show us the different labels separated by `;`.

dls.show_batch(max_n=9, figsize=(12, 9))

# To create a `Learner` we use the same function as in lesson 1. Our base architecture is resnet50 again, but the metrics are a little bit differeent: we use `accuracy_thresh` instead of `accuracy`. In lesson 1, we determined the predicition for a given class by picking the final activation that was the biggest, but here, each activation can be 0. or 1. `accuracy_thresh` selects the ones that are above a certain threshold (0.5 by default) and compares them to the ground truth.
#
# As for Fbeta, it's the metric that was used by Kaggle on this competition. See [here](https://en.wikipedia.org/wiki/F1_score) for more details.

arch = resnet50

acc_02 = partial(accuracy_multi, thresh=0.2)
f_score = FBetaMulti(2, thresh=0.2, average='samples')
learn = cnn_learner(dls, arch, metrics=[acc_02, f_score])

# We use the LR Finder to pick a good learning rate.

learn.lr_find()

# Then we can fit the head of our network.

lr = 0.01

learn.fit_one_cycle(5, slice(lr))

learn.save('stage-1-rn50')

# ...And fine-tune the whole model:

learn.unfreeze()

learn.lr_find()

learn.fit_one_cycle(5, slice(1e-5, lr / 5))

learn.save('stage-2-rn50')

tfms = aug_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0., size=256)

np.random.seed(42)
dls = planet.dataloaders(df, bs=64, path=path, batch_tfms=tfms + [Normalize.from_stats(*imagenet_stats)])

learn.dls = dls

learn.freeze()

learn.lr_find()

lr = 1e-2 / 2

learn.fit_one_cycle(5, slice(lr))

learn.save('stage-1-256-rn50')

learn.unfreeze()

learn.fit_one_cycle(5, slice(1e-5, lr / 5))

learn.recorder.plot_loss()

learn.save('stage-2-256-rn50')

# You won't really know how you're going until you submit to Kaggle, since the leaderboard isn't using the same subset as we have for training. But as a guide, 50th place (out of 938 teams) on the private leaderboard was a score of `0.930`.

# +
# learn.export()
# -

# ## Submitting to Kaggle

# +
# #! kaggle competitions download -c planet-understanding-the-amazon-from-space -f test-jpg.tar.7z -p {path}
#! 7za -bd -y -so x {path}/test-jpg.tar.7z | tar xf - -C {path}
# #! kaggle competitions download -c planet-understanding-the-amazon-from-space -f test-jpg-additional.tar.7z -p {path}
#! 7za -bd -y -so x {path}/test-jpg-additional.tar.7z | tar xf - -C {path}
# -

test_items = get_image_files(path / 'test-jpg') + get_image_files(path / 'test-jpg-additional')
len(test_items)

dl = learn.dls.test_dl(test_items, rm_type_tfms=1, bs=64)

preds, _ = learn.get_preds(dl=dl)

preds.shape

thresh = 0.2
labelled_preds = [' '.join([learn.dls.vocab[i] for i, p in enumerate(pred) if p > thresh]) for pred in preds.numpy()]

labelled_preds[:5]

fnames = [f.name[:-4] for f in test_items]

df = pd.DataFrame({'image_name': fnames, 'tags': labelled_preds}, columns=['image_name', 'tags'])

df.to_csv(path / 'submission.csv', index=False)

# ! kaggle competitions submit planet-understanding-the-amazon-from-space -f {path/'submission.csv'} -m "My submission"

# Private Leaderboard score: 0.9296 (around 80th)
