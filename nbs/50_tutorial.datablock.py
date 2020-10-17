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
from fastai.tabular.core import *
from fastai.text.all import *
from fastai.vision.all import *
from fastai.data.all import *
from nbdev.showdoc import show_doc
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# +
# all_slow
# -

# hide

# # Data block tutorial
#
# > Using the data block across all applications

# In this tutorial, we'll see how to use the data block API on a variety of tasks and how to debug data blocks. The data block API takes its name from the way it's designed: every bit needed to build the `DataLoaders` object (type of inputs, targets, how to label, split...) is encapsulated in a block, and you can mix and match those blocks

# ## Building a `DataBlock` from scratch

# The rest of this tutorial will give many examples, but let's first build a `DataBlock` from scratch on the dogs versus cats problem we saw in the [vision tutorial](http://docs.fast.ai/tutorial.vision). First we import everything needed in vision.


# The first step is to download and decompress our data (if it's not already done) and get its location:

path = untar_data(URLs.PETS)

# And as we saw, all the filenames are in the "images" folder. The `get_image_files` function helps get all the images in subfolders:

fnames = get_image_files(path / "images")

# Let's begin with an empty `DataBlock`.

dblock = DataBlock()

# By itself, a `DataBlock` is just a blue print on how to assemble your data. It does not do anything until you pass it a source. You can choose to then convert that source into a `Datasets` or a `DataLoaders` by using the `DataBlock.datasets` or `DataBlock.dataloaders` method. Since we haven't done anything to get our data ready for batches, the `dataloaders` method will fail here, but we can have a look at how it gets converted in `Datasets`. This is where we pass the source of our data, here all our filenames:

dsets = dblock.datasets(fnames)
dsets.train[0]

# By default, the data block API assumes we have an input and a target, which is why we see our filename repeated twice.
#
# The first thing we can do is use a `get_items` function to actually assemble our items inside the data block:

dblock = DataBlock(get_items=get_image_files)

# The difference is that you then pass as a source the folder with the images and not all the filenames:

dsets = dblock.datasets(path / "images")
dsets.train[0]


# Our inputs are ready to be processed as images (since images can be built from filenames), but our target is not. Since we are in a cat versus dog problem, we need to convert that filename to "cat" vs "dog" (or `True` vs `False`). Let's build a function for this:

def label_func(fname):
    return "cat" if fname.name[0].isupper() else "dog"


# We can then tell our data block to use it to label our target by passing it as `get_y`:

# +
dblock = DataBlock(get_items=get_image_files,
                   get_y=label_func)

dsets = dblock.datasets(path / "images")
dsets.train[0]
# -

# Now that our inputs and targets are ready, we can specify types to tell the data block API that our inputs are images and our targets are categories. Types are represented by blocks in the data block API, here we use `ImageBlock` and `CategoryBlock`:

# +
dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                   get_items=get_image_files,
                   get_y=label_func)

dsets = dblock.datasets(path / "images")
dsets.train[0]
# -

# We can see how the `DataBlock` automatically added the transforms necessary to open the image, or how it changed the name "cat" to an index (with a special tensor type). To do this, it created a mapping from categories to index called "vocab" that we can access this way:

dsets.vocab

# Note that you can mix and match any block for input and targets, which is why the API is named data block API. You can also have more than two blocks (if you have multiple inputs and/or targets), you would just need to pass `n_inp` to the `DataBlock` to tell the library how many inputs there are (the rest would be targets) and pass a list of functions to `get_x` and/or `get_y` (to explain how to process each item to be ready for his type). See the object detection below for such an example.
#
# The next step is to control how our validation set is created. We do this by passing a `splitter` to `DataBlock`. For instance, here is how to do a random split.

# +
dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                   get_items=get_image_files,
                   get_y=label_func,
                   splitter=RandomSplitter())

dsets = dblock.datasets(path / "images")
dsets.train[0]
# -

# The last step is to specify item transforms and batch transforms (the same way we do it in `ImageDataLoaders` factory methods):

dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                   get_items=get_image_files,
                   get_y=label_func,
                   splitter=RandomSplitter(),
                   item_tfms=Resize(224))

# With that resize, we are now able to batch items together and can finally call `dataloaders` to convert our `DataBlock` to a `DataLoaders` object:

dls = dblock.dataloaders(path / "images")
dls.show_batch()

# The way we usually build the data block in one go is by answering a list of questions:
#
# - what is the types of your inputs/targets? Here images and categories
# - where is your data? Here in filenames in subfolders
# - does something need to be applied to inputs? Here no
# - does something need to be applied to the target? Here the `label_func` function
# - how to split the data? Here randomly
# - do we need to apply something on formed items? Here a resize
# - do we need to apply something on formed batches? Here no
#
# This gives us this design:

dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                   get_items=get_image_files,
                   get_y=label_func,
                   splitter=RandomSplitter(),
                   item_tfms=Resize(224))

# For two questions that got a no, the corresponding arguments we would pass if the anwser was different would be `get_x` and `batch_tfms`.

# ## Image classification

# Let's begin with examples of image classification problems. There are two kinds of image classification problems: problems with single-label (each image has one given label) or multi-label (each image can have multiple or no labels at all). We will cover those two kinds here.


# ### MNIST (single label)

# [MNIST](http://yann.lecun.com/exdb/mnist/) is a dataset of hand-written digits from 0 to 9. We can very easily load it in the data block API by answering the following questions:
#
# - what are the types of our inputs and targets? Black and white images and labels.
# - where is the data? In subfolders.
# - how do we know if a sample is in the training or the validation set? By looking at the grandparent folder.
# - how do we know the label of an image? By looking at the parent folder.
#
# In terms of the API, those answers translate like this:

mnist = DataBlock(blocks=(ImageBlock(cls=PILImageBW), CategoryBlock),
                  get_items=get_image_files,
                  splitter=GrandparentSplitter(),
                  get_y=parent_label)

# Our types become blocks: one for images (using the black and white `PILImageBW` class) and one for categories. Searching subfolder for all image filenames is done by the `get_image_files` function. The split training/validation is done by using a `GrandparentSplitter`. And the function to get our targets (often called `y`) is `parent_label`.
#
# To get an idea of the objects the fastai library provides for reading, labelling or splitting, check the `data.transforms` module.
#
# In itself, a data block is just a blueprint. It does not do anything and does not check for errors. You have to feed it the source of the data to actually gather something. This is done with the `.dataloaders` method:

dls = mnist.dataloaders(untar_data(URLs.MNIST_TINY))
dls.show_batch(max_n=9, figsize=(4, 4))

# If something went wrong in the previous step, or if you're just curious about what happened under the hood, use the `summary` method. It will go verbosely step by step, and you will see at which point the process failed.

mnist.summary(untar_data(URLs.MNIST_TINY))

# Let's go over another example!

# ### Pets (single label)

# The [Oxford IIIT Pets dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) is a dataset of pictures of dogs and cats, with 37 different breeds. A slight (but very) important difference with MNIST is that images are now not all of the same size. In MNIST they were all 28 by 28 pixels, but here they have different aspect ratios or dimensions. Therefore, we will need to add something to make them all the same size to be able to assemble them together in a batch. We will also see how to add data augmentation.
#
# So let's go over the same questions as before and add two more:
#
# - what are the types of our inputs and targets? Images and labels.
# - where is the data? In subfolders.
# - how do we know if a sample is in the training or the validation set? We'll take a random split.
# - how do we know the label of an image? By looking at the parent folder.
# - do we want to apply a function to a given sample? Yes, we need to resize everything to a given size.
# - do we want to apply a function to a batch after it's created? Yes, we want data augmentation.

pets = DataBlock(blocks=(ImageBlock, CategoryBlock),
                 get_items=get_image_files,
                 splitter=RandomSplitter(),
                 get_y=Pipeline([attrgetter("name"), RegexLabeller(pat=r'^(.*)_\d+.jpg$')]),
                 item_tfms=Resize(128),
                 batch_tfms=aug_transforms())

# And like for MNIST, we can see how the answers to those questions directly translate in the API. Our types become blocks: one for images and one for categories. Searching subfolder for all image filenames is done by the `get_image_files` function. The split training/validation is done by using a `RandomSplitter`. The function to get our targets (often called `y`) is a composition of two transforms: we get the name attribute of our `Path` filenames, then apply a regular expression to get the class. To compose those two transforms into one, we use a `Pipeline`.
#
# Finally, We apply a resize at the item level and `aug_transforms()` at the batch level.

dls = pets.dataloaders(untar_data(URLs.PETS) / "images")
dls.show_batch(max_n=9)

# Now let's see how we can use the same API for a multi-label problem.

# ### Pascal (multi-label)

# The [Pascal dataset](http://host.robots.ox.ac.uk/pascal/VOC/) is originally an object detection dataset (we have to predict where some objects are in pictures). But it contains lots of pictures with various objects in them, so it gives a great example for a multi-label problem. Let's download it and have a look at the data:

pascal_source = untar_data(URLs.PASCAL_2007)
df = pd.read_csv(pascal_source / "train.csv")

df.head()

# So it looks like we have one column with filenames, one column with the labels (separated by space) and one column that tells us if the filename should go in the validation set or not.
#
# There are multiple ways to put this in a `DataBlock`, let's go over them, but first, let's answer our usual questionnaire:
#
# - what are the types of our inputs and targets? Images and multiple labels.
# - where is the data? In a dataframe.
# - how do we know if a sample is in the training or the validation set? A column of our dataframe.
# - how do we get an image? By looking at the column fname.
# - how do we know the label of an image? By looking at the column labels.
# - do we want to apply a function to a given sample? Yes, we need to resize everything to a given size.
# - do we want to apply a function to a batch after it's created? Yes, we want data augmentation.
#
# Notice how there is one more question compared to before: we wont have to use a `get_items` function here because we already have all our data in one place. But we will need to do something to the raw dataframe to get our inputs, read the first column and add the proper folder before the filename. This is what we pass as `get_x`.

pascal = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   splitter=ColSplitter(),
                   get_x=ColReader(0, pref=pascal_source / "train"),
                   get_y=ColReader(1, label_delim=' '),
                   item_tfms=Resize(224),
                   batch_tfms=aug_transforms())

# Again, we can see how the answers to the questions directly translate in the API. Our types become blocks: one for images and one for multi-categories. The split is done by a `ColSplitter` (it defaults to the column named `is_valid`). The function to get our inputs (often called `x`) is a `ColReader` on the first column with a prefix, the function to get our targets (often called `y`) is `ColReader` on the second column, with a space delimiter. We apply a resize at the item level and `aug_transforms()` at the batch level.

dls = pascal.dataloaders(df)
dls.show_batch()

# Another way to do this is by directly using functions for `get_x` and `get_y`:

# +
pascal = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   splitter=ColSplitter(),
                   get_x=lambda x: pascal_source / "train" / f'{x[0]}',
                   get_y=lambda x: x[1].split(' '),
                   item_tfms=Resize(224),
                   batch_tfms=aug_transforms())

dls = pascal.dataloaders(df)
dls.show_batch()
# -

# Alternatively, we can use the names of the columns as attributes (since rows of a dataframe are pandas series).

# +
pascal = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   splitter=ColSplitter(),
                   get_x=lambda o: f'{pascal_source}/train/' + o.fname,
                   get_y=lambda o: o.labels.split(),
                   item_tfms=Resize(224),
                   batch_tfms=aug_transforms())

dls = pascal.dataloaders(df)
dls.show_batch()


# -

# The most efficient way (to avoid iterating over the rows of the dataframe, which can take a long time) is to use the `from_columns` method. It will use `get_items` to convert the columns into numpy arrays. The drawback is that since we lose the dataframe after extracting the relevant columns, we can't use a `ColSplitter` anymore. Here we used an `IndexSplitter` after manually extracting the index of the validation set from the dataframe:

# +
def _pascal_items(x): return (
    f'{pascal_source}/train/' + x.fname, x.labels.str.split())
valid_idx = df[df['is_valid']].index.values

pascal = DataBlock.from_columns(blocks=(ImageBlock, MultiCategoryBlock),
                                get_items=_pascal_items,
                                splitter=IndexSplitter(valid_idx),
                                item_tfms=Resize(224),
                                batch_tfms=aug_transforms())
# -

dls = pascal.dataloaders(df)
dls.show_batch()

# ## Image localization

# There are various problems that fall in the image localization category: image segmentation (which is a task where you have to predict the class of each pixel of an image), coordinate predictions (predict one or several key points on an image) and object detection (draw a box around objects to detect).
#
# Let's see an example of each of those and how to use the data block API in each case.

# ### Segmentation

# We will use a small subset of the [CamVid dataset](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) for our example.

path = untar_data(URLs.CAMVID_TINY)

# Let's go over our usual questionnaire:
#
# - what are the types of our inputs and targets? Images and segmentation masks.
# - where is the data? In subfolders.
# - how do we know if a sample is in the training or the validation set? We'll take a random split.
# - how do we know the label of an image? By looking at a corresponding file in the "labels" folder.
# - do we want to apply a function to a batch after it's created? Yes, we want data augmentation.

camvid = DataBlock(blocks=(ImageBlock, MaskBlock(codes=np.loadtxt(path / 'codes.txt', dtype=str))),
                   get_items=get_image_files,
                   splitter=RandomSplitter(),
                   get_y=lambda o: path / 'labels' / f'{o.stem}_P{o.suffix}',
                   batch_tfms=aug_transforms())

# The `MaskBlock` is generated with the `codes` that give the correpondence between pixel value of the masks and the object they correspond to (like car, road, pedestrian...). The rest should look pretty familiar by now.

dls = camvid.dataloaders(path / "images")
dls.show_batch()

# ### Points

# For this example we will use a small sample of the [BiWi kinect head pose dataset](https://www.kaggle.com/kmader/biwi-kinect-head-pose-database). It contains pictures of people and the task is to predict where the center of their head is. We have saved this small dataet with a dictionary filename to center:

biwi_source = untar_data(URLs.BIWI_SAMPLE)
fn2ctr = (biwi_source / 'centers.pkl').load()

# Then we can go over our usual questions:
#
# - what are the types of our inputs and targets? Images and points.
# - where is the data? In subfolders.
# - how do we know if a sample is in the training or the validation set? We'll take a random split.
# - how do we know the label of an image? By using the `fn2ctr` dictionary.
# - do we want to apply a function to a batch after it's created? Yes, we want data augmentation.

biwi = DataBlock(blocks=(ImageBlock, PointBlock),
                 get_items=get_image_files,
                 splitter=RandomSplitter(),
                 get_y=lambda o: fn2ctr[o.name].flip(0),
                 batch_tfms=aug_transforms())

# And we can use it to create a `DataLoaders`:

dls = biwi.dataloaders(biwi_source)
dls.show_batch(max_n=9)

# ### Bounding boxes

# For this task, we will use a small subset of the [COCO dataset](http://cocodataset.org/#home). It contains pictures with day-to-day objects and the goal is to predict where the objects are by drawing a rectangle around them.
#
# The fastai library comes with a function called `get_annotations` that will interpret the content of `train.json` and give us a dictionary filename to (bounding boxes, labels).

coco_source = untar_data(URLs.COCO_TINY)
images, lbl_bbox = get_annotations(coco_source / 'train.json')
img2bbox = dict(zip(images, lbl_bbox))

# Then we can go over our usual questions:
#
# - what are the types of our inputs and targets? Images and bounding boxes.
# - where is the data? In subfolders.
# - how do we know if a sample is in the training or the validation set? We'll take a random split.
# - how do we know the label of an image? By using the `img2bbox` dictionary.
# - do we want to apply a function to a given sample? Yes, we need to resize everything to a given size.
# - do we want to apply a function to a batch after it's created? Yes, we want data augmentation.

coco = DataBlock(blocks=(ImageBlock, BBoxBlock, BBoxLblBlock),
                 get_items=get_image_files,
                 splitter=RandomSplitter(),
                 get_y=[lambda o: img2bbox[o.name][0], lambda o: img2bbox[o.name][1]],
                 item_tfms=Resize(128),
                 batch_tfms=aug_transforms(),
                 n_inp=1)

# Note that we provide three types, because we have two targets: the bounding boxes and the labels. That's why we pass `n_inp=1` at the end, to tell the library where the inputs stop and the targets begin.
#
# This is also why we pass a list to `get_y`: since we have two targets, we must tell the library how to label for each of them (you can use `noop` if you don't want to do anything for one).

dls = coco.dataloaders(coco_source)
dls.show_batch(max_n=9)

# ## Text

# We will show two examples: language modeling and text classification. Note that with the data block API, you can adapt the example before for multi-label to a problem where the inputs are texts.


# ### Language model

# We will use a dataset compose of movie reviews from IMDb. As usual, we can download it in one line of code with `untar_data`.

path = untar_data(URLs.IMDB_SAMPLE)
df = pd.read_csv(path / 'texts.csv')
df.head()

# We can see it's composed of (pretty long!) reviews labeled positive or negative. Let's go over our usual questions:
#
# - what are the types of our inputs and targets? Texts and we don't really have targets, since the targets is derived from the inputs.
# - where is the data? In a dataframe.
# - how do we know if a sample is in the training or the validation set? We have an `is_valid` column.
# - how do we get our inputs? In the `text` column.

imdb_lm = DataBlock(blocks=TextBlock.from_df('text', is_lm=True),
                    get_x=ColReader('text'),
                    splitter=ColSplitter())

# Since there are no targets here, we only have one block to specify. `TextBlock`s are a bit special compared to other `TransformBlock`s: to be able to efficiently tokenize all texts during setup, you need to use the class methods `from_folder` or `from_df`.
#
# Note: the `TestBlock` tokenization process puts tokenized inputs into a column called `text`. The `ColReader` for `get_x` will always reference `text`, even if the original text inputs were in a column with another name in the dataframe.
#
# We can then get our data into `DataLoaders` by passing the dataframe to the `dataloaders` method:

dls = imdb_lm.dataloaders(df, bs=64, seq_len=72)
dls.show_batch(max_n=6)

# ### Text classification

# For the text classification, let's go over our usual questions:
#
# - what are the types of our inputs and targets? Texts and categories.
# - where is the data? In a dataframe.
# - how do we know if a sample is in the training or the validation set? We have an `is_valid` column.
# - how do we get our inputs? In the `text` column.
# - how do we get our targets? In the `label` column.

imdb_clas = DataBlock(blocks=(TextBlock.from_df('text', seq_len=72, vocab=dls.vocab), CategoryBlock),
                      get_x=ColReader('text'),
                      get_y=ColReader('label'),
                      splitter=ColSplitter())

# Like in the previous example, we use a class method to build a `TextBlock`. We can pass it the vocabulary of our language model (very useful for the ULMFit approach). We also show the `seq_len` argument (which defaults to 72) just because you need to make sure to use the same here and also in your `text_classifier_learner`.

# > Warning: You need to make sure to use the same `seq_len` in `TextBlock` and the `Learner` you will define later on.

dls = imdb_clas.dataloaders(df, bs=64)
dls.show_batch()

# ## Tabular data

# Tabular data doesn't really use the data block API as it's relying on another API with `TabularPandas` for efficient preprocessing and batching (there will be some less efficient API that plays nicely with the data block API added in the near future). You can still use different blocks for the targets.


# For our example, we will look at a subset of the [adult dataset](https://archive.ics.uci.edu/ml/datasets/adult) which contains some census data and where the task is to predict if someone makes more than 50k or not.

adult_source = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(adult_source / 'adult.csv')
df.head()

# In a tabular problem, we need to split the columns between the ones that represent continuous variables (like the age) and the ones that represent categorical variables (like the education):

cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
cont_names = ['age', 'fnlwgt', 'education-num']

# Standard preprocessing in fastai, use those pre-processors:

procs = [Categorify, FillMissing, Normalize]

# `Categorify` will change the categorical columns into indices, `FillMissing` will fill the missing values in the continuous columns (if any) and add an na categorical column (if necessary). `Normalize` will normalize the continuous columns (subtract the mean and divide by the standard deviation).
#
# We can still use any splitter to create the splits as we'd like them:

splits = RandomSplitter()(range_of(df))

# And then everything goes in a `TabularPandas` object:

to = TabularPandas(df, procs, cat_names, cont_names, y_names="salary", splits=splits, y_block=CategoryBlock)

# We put `y_block=CategoryBlock` just to show you how to customize the block for the targets, but it's usually inferred from the data, so you don't need to pass it, normally.

dls = to.dataloaders()
dls.show_batch()
