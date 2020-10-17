# -*- coding: utf-8 -*-
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
from fastai.vision.all import *
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# # Computer vision
#
# > Using the fastai library in computer vision.


# +
# all_slow
# -

# This tutorial highlights on how to quickly build a `Learner` and fine tune a pretrained model on most computer vision tasks.

# ## Single-label classification

# For this task, we will use the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) that contains images of cats and dogs of 37 different breeds. We will first show how to build a simple cat-vs-dog classifier, then a little bit more advanced model that can classify all breeds.
#
# The dataset can be downloaded and decompressed with this line of code:

path = untar_data(URLs.PETS)

# It will only do this download once, and return the location of the decompressed archive. We can check what is inside with the `.ls()` method.

path.ls()

# We will ignore the annotations folder for now, and focus on the images one. `get_image_files` is a fastai function that helps us grab all the image files (recursively) in one folder.

files = get_image_files(path / "images")
len(files)

# ### Cats vs dogs

# To label our data for the cats vs dogs problem, we need to know which filenames are of dog pictures and which ones are of cat pictures. There is an easy way to distinguish: the name of the file begins with a capital for cats, and a lowercased letter for dogs:

files[0], files[6]


# We can then define an easy label function:

def label_func(f): return f[0].isupper()


# To get our data ready for a model, we need to put it in a `DataLoaders` object. Here we have a function that labels using the file names, so we will use `ImageDataLoaders.from_name_func`. There are other factory methods of `ImageDataLoaders` that could be more suitable for your problem, so make sure to check them all in `vision.data`.

dls = ImageDataLoaders.from_name_func(path, files, label_func, item_tfms=Resize(224))

# We have passed to this function the directory we're working in, the `files` we grabbed, our `label_func` and one last piece as `item_tfms`: this is a `Transform` applied on all items of our dataset that will resize each image to 224 by 224, by using a random crop on the largest dimension to make it a square, then resizing to 224 by 224. If we didn't pass this, we would get an error later as it would be impossible to batch the items together.
#
# We can then check if everything looks okay with the `show_batch` method (`True` is for cat, `False` is for dog):

dls.show_batch()

# Then we can create a `Learner`, which is a fastai object that combines the data and a model for training, and uses transfer learning to fine tune a pretrained model in just two lines of code:

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)

# The first line downloaded a model called ResNet34, pretrained on [ImageNet](http://www.image-net.org/), and adapted it to our specific problem. It then fine tuned that model and in a relatively short time, we get a model with an error rate of 0.3%... amazing!
#
# If you want to make a prediction on a new image, you can use `learn.predict`:

learn.predict(files[0])

# The predict method returns three things: the decoded prediction (here `False` for dog), the index of the predicted class and the tensor of probabilities of all classes in the order of their indexed labels(in this case, the model is quite confifent about the being that of a dog). This method accepts a filename, a PIL image or a tensor directly in this case.
# We can also have a look at some predictions with the `show_results` method:

learn.show_results()

# Check out the other applications like text or tabular, or the other problems covered in this tutorial, and you will see they all share a consistent API for gathering the data and look at it, create a `Learner`, train the model and look at some predictions.

# ### Classifying breeds

# To label our data with the breed name, we will use a regular expression to extract it from the filename. Looking back at a filename, we have:

files[0].name

# so the class is everything before the last `_` followed by some digits. A regular expression that will catch the name is thus:

pat = r'^(.*)_\d+.jpg'

# Since it's pretty common to use regular expressions to label the data (often, labels are hidden in the file names), there is a factory method to do just that:

dls = ImageDataLoaders.from_name_re(path, files, pat, item_tfms=Resize(224))

# Like before, we can then use `show_batch` to have a look at our data:

dls.show_batch()

# Since classifying the exact breed of cats or dogs amongst 37 different breeds is a harder problem, we will slightly change the definition of our `DataLoaders` to use data augmentation:

dls = ImageDataLoaders.from_name_re(path, files, pat, item_tfms=Resize(460),
                                    batch_tfms=aug_transforms(size=224))

# This time we resized to a larger size before batching, and we added `batch_tfms`. `aug_transforms` is a function that provides a collection of data augmentation transforms with defaults we found that perform well on many datasets.  You can customize these transforms by passing appropriate arguments to `aug_transnforms`.

dls.show_batch()

# We can then create our `Learner` exactly as before and train our model.

learn = cnn_learner(dls, resnet34, metrics=error_rate)

# We used the default learning rate before, but we might want to find the best one possible. For this, we can use the learning rate finder:

learn.lr_find()

# It plots the graph of the learning rate finder and gives us two suggestions (minimum divided by 10 and steepest gradient). Let's use `3e-3` here. We will also do a bit more epochs:

learn.fine_tune(4, 3e-3)

# Again, we can have a look at some predictions with `show_results`:

learn.show_results()

# Another thing that is useful is an interpretation object, it can show us where the model made the worse predictions:

interp = Interpretation.from_learner(learn)

interp.plot_top_losses(9, figsize=(15, 10))

# ### Single-label classification - With the data block API

# We can also use the data block API to get our data in a `DataLoaders`. This is a bit more advanced, so fell free to skip this part if you are not comfortable with learning new API's just yet.
#
# A datablock is built by giving the fastai library a bunch of informations:
#
# - the types used, through an argument called `blocks`: here we have images and categories, so we pass `ImageBlock` and `CategoryBlock`.
# - how to get the raw items, here our function `get_image_files`.
# - how to label those items, here with the same regular expression as before.
# - how to split those items, here with a random splitter.
# - the `item_tfms` and `batch_tfms` like before.

pets = DataBlock(blocks=(ImageBlock, CategoryBlock),
                 get_items=get_image_files,
                 splitter=RandomSplitter(),
                 get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'), 'name'),
                 item_tfms=Resize(460),
                 batch_tfms=aug_transforms(size=224))

# The pets object by itself is empty: it only containes the functions that will help us gather the data. We have to call `dataloaders` method to get a `DataLoaders`. We pass it the source of the data:

dls = pets.dataloaders(untar_data(URLs.PETS) / "images")

# Then we can look at some of our pictures with `dls.show_batch()`

dls.show_batch(max_n=9)

# ## Multi-label classification

# For this task, we will use the [Pascal Dataset](http://host.robots.ox.ac.uk/pascal/VOC/) that contains images with different kinds of objects/persons. It's orginally a dataset for object detection, meaning the task is not only to detect if there is an instance of one class of an image, but to also draw a bounding box around it. Here we will just try to predict all the classes in one given image.
#
# Multi-label classification defers from before in the sense each image does not belong to one category. An image could have a person *and* a horse inside it for instance. Or have none of the categories we study.
#
# As before, we can download the dataset pretty easily:

path = untar_data(URLs.PASCAL_2007)
path.ls()

# The information about the labels of each image is in the file named `train.csv`. We load it using pandas:

df = pd.read_csv(path / 'train.csv')
df.head()

# ### Multi-label classification - Using the high-level API

# That's pretty straightforward: for each filename, we get the different labels (separated by space) and the last column tells if it's in the validation set or not. To get this in `DataLoaders` quickly, we have a factory method, `from_df`. We can specify the underlying path where all the images are, an additional folder to add between the base path and the filenames (here `train`), the `valid_col` to consider for the validation set (if we don't specify this, we take a random subset), a `label_delim` to split the labels and, as before, `item_tfms` and `batch_tfms`.
#
# Note that we don't have to specify the `fn_col` and the `label_col` because they default to the first and second column respectively.

dls = ImageDataLoaders.from_df(df, path, folder='train', valid_col='is_valid', label_delim=' ',
                               item_tfms=Resize(460), batch_tfms=aug_transforms(size=224))

# As before, we can then have a look at the data with the `show_batch` method.

dls.show_batch()

# Training a model is as easy as before: the same functions can be applied and the fastai library will automatically detect that we are in a multi-label problem, thus picking the right loss function. The only difference is in the metric we pass: `error_rate` will not work for a multi-label problem, but we can use `accuracy_thresh`.

learn = cnn_learner(dls, resnet50, metrics=partial(accuracy_multi, thresh=0.5))

# As before, we can use `learn.lr_find` to pick a good learning rate:

learn.lr_find()

# We can pick the suggested learning rate and fine-tune our pretrained model:

learn.fine_tune(4, 3e-2)

# Like before, we can easily have a look at the results:

learn.show_results()

# Or get the predictions on a given image:

learn.predict(path / 'train/000005.jpg')

# As for the single classification predictions, we get three things. The last one is the prediction of the model on each class (going from 0 to 1). The second to last cooresponds to a one-hot encoded targets (you get `True` for all predicted classes, the ones that get a probability > 0.5) and the first is the decoded, readable version.

# And like before, we can check where the model did its worse:

interp = Interpretation.from_learner(learn)
interp.plot_top_losses(9)

# ### Multi-label classification - With the data block API

# We can also use the data block API to get our data in a `DataLoaders`. Like we said before, feel free to skip this part if you are not comfortable with learning new APIs just yet.
#
# Remember how the data is structured in our dataframe:

df.head()

# In this case we build the data block by providing:
#
# - the types used: `ImageBlock` and `MultiCategoryBlock`.
# - how to get the input items from our dataframe: here we read the column `fname` and need to add path/train/ at the beginning to get proper filenames.
# - how to get the targets from our dataframe: here we read the column `labels` and need to split by space.
# - how to split the items, here by using the column `is_valid`.
# - the `item_tfms` and `batch_tfms` like before.

pascal = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   splitter=ColSplitter('is_valid'),
                   get_x=ColReader('fname', pref=str(path / 'train') + os.path.sep),
                   get_y=ColReader('labels', label_delim=' '),
                   item_tfms=Resize(460),
                   batch_tfms=aug_transforms(size=224))

# This block is slightly different than before: we don't need to pass a function to gather all our items as the dataframe we will give already has them all. However, we do need to preprocess the row of that dataframe to get out inputs, which is why we pass a `get_x`. It defaults to the fastai function `noop`, which is why we didn't need to pass it along before.
#
# Like before, `pascal` is just a blueprint. We need to pass it the source of our data to be able to get `DataLoaders`:

dls = pascal.dataloaders(df)

# Then we can look at some of our pictures with `dls.show_batch()`

dls.show_batch(max_n=9)

# ## Segmentation

# Segmentation is a problem where we have to predict a category for each pixel of the image. For this task, we will use the [Camvid dataset](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/), a dataset of screenshots from cameras in cars. Each pixel of the image has a label such as "road", "car" or "pedestrian".
#
# As usual, we can download the data with our `untar_data` function.

path = untar_data(URLs.CAMVID_TINY)
path.ls()

# The `images` folder contains the images, and the corresponding segmentation masks of labels are in the `labels` folder. The `codes` file contains the corresponding integer to class (the masks have an int value for each pixel).

codes = np.loadtxt(path / 'codes.txt', dtype=str)
codes

# ### Segmentation - Using the high-level API

# As before, the `get_image_files` function helps us grab all the image filenames:

fnames = get_image_files(path / "images")
fnames[0]

# Let's have a look in the labels folder:

(path / "labels").ls()[0]


# It seems the segmentation masks have the same base names as the images but with an extra `_P`, so we can define a label function:

def label_func(fn): return path / "labels" / f"{fn.stem}_P{fn.suffix}"


# We can then gather our data using `SegmentationDataLoaders`:

dls = SegmentationDataLoaders.from_label_func(
    path, bs=8, fnames=fnames, label_func=label_func, codes=codes
)

# We do not need to pass `item_tfms` to resize our images here because they already are all of the same size.
#
# As usual, we can have a look at our data with the `show_batch` method. In this instance, the fastai library is superimposing the masks with one specific color per pixel:

dls.show_batch(max_n=6)

# A traditional CNN won't work for segmentation, we have to use a special kind of model called a UNet, so we use `unet_learner` to define our `Learner`:

learn = unet_learner(dls, resnet34)
learn.fine_tune(8)

# And as before, we can get some idea of the predicted results with `show_results`

learn.show_results(max_n=6, figsize=(7, 8))

# ### Segmentation - With the data block API

# We can also use the data block API to get our data in a `DataLoaders`. Like it's been said before, feel free to skip this part if you are not comfortable with learning new APIs just yet.
#
# In this case we build the data block by providing:
#
# - the types used: `ImageBlock` and `MaskBlock`. We provide the `codes` to `MaskBlock` as there is no way to guess them from the data.
# - how to gather our items, here by using `get_image_files`.
# - how to get the targets from our items: by using `label_func`.
# - how to split the items, here randomly.
# - `batch_tfms` for data augmentation.

camvid = DataBlock(blocks=(ImageBlock, MaskBlock(codes)),
                   get_items=get_image_files,
                   get_y=label_func,
                   splitter=RandomSplitter(),
                   batch_tfms=aug_transforms(size=(120, 160)))

dls = camvid.dataloaders(path / "images", path=path, bs=8)

dls.show_batch(max_n=6)

# ## Points

# This section uses the data block API, so if you skipped it before, we recommend you skip this section as well.
#
# We will now look at a task where we want to predict points in a picture. For this, we will use the [Biwi Kinect Head Pose Dataset](https://data.vision.ee.ethz.ch/cvl/gfanelli/head_pose/head_forest.html#db). First thing first, let's begin by downloading the dataset as usual.

path = untar_data(URLs.BIWI_HEAD_POSE)

# Let's see what we've got!

path.ls()

# There are 24 directories numbered from 01 to 24 (they correspond to the different persons photographed) and a corresponding .obj file (we won't need them here). We'll take a look inside one of these directories:

(path / '01').ls()

# Inside the subdirectories, we have different frames, each of them come with an image (`\_rgb.jpg`) and a pose file (`\_pose.txt`). We can easily get all the image files recursively with `get_image_files`, then write a function that converts an image filename to its associated pose file.

img_files = get_image_files(path)
def img2pose(x): return Path(f'{str(x)[:-7]}pose.txt')
img2pose(img_files[0])

# We can have a look at our first image:

im = PILImage.create(img_files[0])
im.shape

im.to_thumb(160)

# The Biwi dataset web site explains the format of the pose text file associated with each image, which shows the location of the center of the head. The details of this aren't important for our purposes, so we'll just show the function we use to extract the head center point:

cal = np.genfromtxt(path / '01' / 'rgb.cal', skip_footer=6)
def get_ctr(f):
    ctr = np.genfromtxt(img2pose(f), skip_header=3)
    c1 = ctr[0] * cal[0][0] / ctr[2] + cal[0][2]
    c2 = ctr[1] * cal[1][1] / ctr[2] + cal[1][2]
    return tensor([c1, c2])


# This function returns the coordinates as a tensor of two items:

get_ctr(img_files[0])

# We can pass this function to `DataBlock` as `get_y`, since it is responsible for labeling each item. We'll resize the images to half their input size, just to speed up training a bit.
#
# One important point to note is that we should not just use a random splitter. The reason for this is that the same person appears in multiple images in this dataset — but we want to ensure that our model can generalise to people that it hasn't seen yet. Each folder in the dataset contains the images for one person. Therefore, we can create a splitter function which returns true for just one person, resulting in a validation set containing just that person's images.
#
# The only other difference to previous data block examples is that the second block is a `PointBlock`. This is necessary so that fastai knows that the labels represent coordinates; that way, it knows that when doing data augmentation, it should do the same augmentation to these coordinates as it does to the images.

biwi = DataBlock(
    blocks=(ImageBlock, PointBlock),
    get_items=get_image_files,
    get_y=get_ctr,
    splitter=FuncSplitter(lambda o: o.parent.name == '13'),
    batch_tfms=[*aug_transforms(size=(240, 320)),
                Normalize.from_stats(*imagenet_stats)]
)

dls = biwi.dataloaders(path)
dls.show_batch(max_n=9, figsize=(8, 6))

# Now that we have assembled our data, we can use the rest of the fastai API as usual. `cnn_learner` works perfectly in this case, and the library will infer the proper loss function from the data:

learn = cnn_learner(dls, resnet18, y_range=(-1, 1))

learn.lr_find()

# Then we can train our model:

learn.fine_tune(4, 5e-3)

# The loss is the mean squared error, so that means we make on average an error of

math.sqrt(0.0001)

# percent when predicting our points! And we can look at those results as usual:

learn.show_results()
