# coding: utf-8
# # Computer vision data
get_ipython().run_line_magic('matplotlib', 'inline')
from fastai.gen_doc.nbdoc import *
from fastai.vision import *
# This module contains the classes that define datasets handling [`Image`](/vision.image.html#Image) objects and their tranformations. As usual, we'll start with a quick overview, before we get in to the detailed API docs.
# ## Quickly get your data ready for training
# To get you started as easily as possible, the fastai provides two helper functions to create a [`DataBunch`](/basic_data.html#DataBunch) object that you can directly use for training a classifier. To demonstrate them you'll first need to download and untar the file by executing the following cell. This will create a data folder containing an MNIST subset in `data/mnist_sample`.
path = untar_data(URLs.MNIST_SAMPLE); path
# There are a number of ways to create an [`ImageDataBunch`](/vision.data.html#ImageDataBunch). One common approach is to use *Imagenet-style folders* (see a ways down the page below for details) with [`ImageDataBunch.from_folder`](/vision.data.html#ImageDataBunch.from_folder):
tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_folder(path, ds_tfms=tfms, size=24)
# Here the datasets will be automatically created in the structure of *Imagenet-style folders*. The parameters specified:
# - the transforms to apply to the images in `ds_tfms` (here with `do_flip`=False because we don't want to flip numbers),
# - the target `size` of our pictures (here 24).
#
# As with all [`DataBunch`](/basic_data.html#DataBunch) usage,  a `train_dl` and a `valid_dl` are created that are of the type PyTorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).
#
# If you want to have a look at a few images inside a batch, you can use [`ImageDataBunch.show_batch`](/vision.data.html#ImageDataBunch.show_batch). The `rows` argument is the number of rows and columns to display.
data.show_batch(rows=3, figsize=(5, 5))
# The second way to define the data for a classifier requires a structure like this:
# ```
# path\
#   train\
#   test\
#   labels.csv
# ```
# where the labels.csv file defines the label(s) of each image in the training set. This is the format you will need to use when each image can have multiple labels. It also works with single labels:
pd.read_csv(path / 'labels.csv').head()
# You can then use [`ImageDataBunch.from_csv`](/vision.data.html#ImageDataBunch.from_csv):
data = ImageDataBunch.from_csv(path, ds_tfms=tfms, size=28)
data.show_batch(rows=3, figsize=(5, 5))
# An example of multiclassification can be downloaded with the following cell. It's a sample of the [planet dataset](https://www.google.com/search?q=kaggle+planet&rlz=1C1CHBF_enFR786FR786&oq=kaggle+planet&aqs=chrome..69i57j0.1563j0j7&sourceid=chrome&ie=UTF-8).
planet = untar_data(URLs.PLANET_SAMPLE)
# If we open the labels files, we seach that each image has one or more tags, separated by a space.
df = pd.read_csv(planet / 'labels.csv')
df.head()
data = ImageDataBunch.from_csv(planet, folder='train', size=128, suffix='.jpg', sep=' ',
    ds_tfms=get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.))
# The `show_batch`method will then print all the labels that correspond to each image.
data.show_batch(rows=3, figsize=(10, 8), ds_type=DatasetType.Valid)
# You can find more ways to build an [`ImageDataBunch`](/vision.data.html#ImageDataBunch) without the factory methods in [`data_block`](/data_block.html#data_block).
show_doc(ImageDataBunch)
# This is the same initilialization as a regular [`DataBunch`](/basic_data.html#DataBunch) so you probably don't want to use this directly, but one of the factory methods instead.
# ### Factory methods
# If you quickly want to get a [`ImageDataBunch`](/vision.data.html#ImageDataBunch) and train a model, you should process your data to have it in one of the formats the following functions handle.
show_doc(ImageDataBunch.from_folder)
# "*Imagenet-style*" datasets look something like this (note that the test folder is optional):
#
# ```
# path\
#   train\
#     clas1\
#     clas2\
#     ...
#   valid\
#     clas1\
#     clas2\
#     ...
#   test\
# ```
#
# For example:
data = ImageDataBunch.from_folder(path, ds_tfms=tfms, size=24)
# Note that this (and all factory methods in this section) pass any `kwargs` to [`ImageDataBunch.create`](/vision.data.html#ImageDataBunch.create).
show_doc(ImageDataBunch.from_csv)
# Create [`ImageDataBunch`](/vision.data.html#ImageDataBunch) from `path` by splitting the data in `folder` and labelled in a file `csv_labels` between a training and validation set. Use `valid_pct` to indicate the percentage of the total images for the validation set. An optional `test` folder contains unlabelled data and `suffix` contains an optional suffix to add to the filenames in `csv_labels` (such as '.jpg').
# For example:
data = ImageDataBunch.from_csv(path, ds_tfms=tfms, size=24);
show_doc(ImageDataBunch.from_df)
# Same as [`ImageDataBunch.from_csv`](/vision.data.html#ImageDataBunch.from_csv), but passing in a `DataFrame` instead of a csv file. E.gL
df = pd.read_csv(path / 'labels.csv', header='infer')
df.head()
data = ImageDataBunch.from_df(path, df, ds_tfms=tfms, size=24)
# Different datasets are labeled in many different ways. The following methods can help extract the labels from the dataset in a wide variety of situations. The way they are built in fastai is constructive: there are methods which do a lot for you but apply in specific circumstances and there are methods which do less for you but give you more flexibility.
#
# In this case the hierachy is:
#
# 1. [`ImageDataBunch.from_name_re`](/vision.data.html#ImageDataBunch.from_name_re): Gets the labels from the filenames using a regular expression
# 2. [`ImageDataBunch.from_name_func`](/vision.data.html#ImageDataBunch.from_name_func): Gets the labels from the filenames using any function
# 3. [`ImageDataBunch.from_lists`](/vision.data.html#ImageDataBunch.from_lists): Labels need to be provided as an input in a list
show_doc(ImageDataBunch.from_name_re)
# Creates an [`ImageDataBunch`](/vision.data.html#ImageDataBunch) from `fnames`, calling a regular expression (containing one *re group*) on the file names to get the labels, putting aside `valid_pct` for the validation. In the same way as [`ImageDataBunch.from_csv`](/vision.data.html#ImageDataBunch.from_csv), an optional `test` folder contains unlabelled data.
#
# Our previously created dataframe contains the labels in the filenames so we can leverage it to test this new method. [`ImageDataBunch.from_name_re`](/vision.data.html#ImageDataBunch.from_name_re) needs the exact path of each file so we will append the data path to each filename before creating our [`ImageDataBunch`](/vision.data.html#ImageDataBunch) object.
fn_paths = [path / name for name in df['name']]; fn_paths[:2]
pat = r"/(\d)/\d+\.png$"
data = ImageDataBunch.from_name_re(path, fn_paths, pat=pat, ds_tfms=tfms, size=24)
data.classes
show_doc(ImageDataBunch.from_name_func)
# Works in the same way as [`ImageDataBunch.from_name_re`](/vision.data.html#ImageDataBunch.from_name_re), but instead of a regular expression it expects a function that will determine how to extract the labels from the filenames. (Note that `from_name_re` uses this function in its implementation).
#
# To test it we could build a function with our previous regex. Let's try another, similar approach to show that the labels can be obtained in a different way.
def get_labels(file_path): return '3' if '/3/' in str(file_path) else '7'
data = ImageDataBunch.from_name_func(path, fn_paths, label_func=get_labels, ds_tfms=tfms, size=24)
data.classes
show_doc(ImageDataBunch.from_lists)
# The most flexible factory function; pass in a list of `labels` that correspond to each of the filenames in `fnames`.
#
# To show an example we have to build the labels list outside our [`ImageDataBunch`](/vision.data.html#ImageDataBunch) object and give it as an argument when we call `from_lists`. Let's use our previously created function to create our labels list.
labels_ls = list(map(get_labels, fn_paths))
data = ImageDataBunch.from_lists(path, fn_paths, labels=labels_ls, ds_tfms=tfms, size=24)
data.classes
show_doc(ImageDataBunch.create_from_ll)
# Use `bs`, `num_workers`, `collate_fn` and a potential `test` folder. `ds_tfms` is a tuple of two lists of transforms to be applied to the training and the validation (plus test optionally) set. `tfms` are the transforms to apply to the [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader). The `size` and the `kwargs` are passed to the transforms for data augmentation.
show_doc(ImageDataBunch.single_from_classes)
jekyll_note('This method is deprecated, you should use DataBunch.load_empty now.')
# ### Methods
# In the next two methods we will use a new dataset, CIFAR. This is because the second method will get the statistics for our dataset and we want to be able to show different statistics per channel. If we were to use MNIST, these statistics would be the same for every channel. White pixels are [255,255,255] and black pixels are [0,0,0] (or in normalized form [1,1,1] and [0,0,0]) so there is no variance between channels.
path = untar_data(URLs.CIFAR); path
show_doc(channel_view)
data = ImageDataBunch.from_folder(path, ds_tfms=tfms, valid='test', size=24)
def channel_view(x: Tensor) -> Tensor:
    "Make channel the first axis of `x` and flatten remaining axes"
    return x.transpose(0, 1).contiguous().view(x.shape[1], -1)
# This function takes a tensor and flattens all dimensions except the channels, which it keeps as the first axis. This function is used to feed [`ImageDataBunch.batch_stats`](/vision.data.html#ImageDataBunch.batch_stats) so that it can get the pixel statistics of a whole batch.
#
# Let's take as an example the dimensions our MNIST batches: 128, 3, 24, 24.
t = torch.Tensor(128, 3, 24, 24)
t.size()
tensor = channel_view(t)
tensor.size()
show_doc(ImageDataBunch.batch_stats)
data.batch_stats()
show_doc(ImageDataBunch.normalize)
# In the fast.ai library we have `imagenet_stats`, `cifar_stats` and `mnist_stats` so we can add normalization easily with any of these datasets. Let's see an example with our dataset of choice: MNIST.
data.normalize(cifar_stats)
data.batch_stats()
# ## Data normalization
# You may also want to normalize your data, which can be done by using the following functions.
show_doc(normalize)
show_doc(denormalize)
show_doc(normalize_funcs)
# On MNIST the mean and std are 0.1307 and 0.3081 respectively (looked on Google). If you're using a pretrained model, you'll need to use the normalization that was used to train the model. The imagenet norm and denorm functions are stored as constants inside the library named <code>imagenet_norm</code> and <code>imagenet_denorm</code>. If you're training a model on CIFAR-10, you can also use <code>cifar_norm</code> and <code>cifar_denorm</code>.
#
# You may sometimes see warnings about *clipping input data* when plotting normalized data. That's because even although it's denormalized when plotting automatically, sometimes floating point errors may make some values slightly out or the correct range. You can safely ignore these warnings in this case.
data = ImageDataBunch.from_folder(untar_data(URLs.MNIST_SAMPLE),
                                  ds_tfms=tfms, size=24)
data.normalize()
data.show_batch(rows=3, figsize=(6, 6))
show_doc(get_annotations)
# To use this dataset and collate samples into batches, you'll need to following function:
show_doc(bb_pad_collate)
# Finally, to apply transformations to [`Image`](/vision.image.html#Image) in a [`Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset), we use this last class.
# ## ItemList specific to vision
# The vision application adds a few subclasses of [`ItemList`](/data_block.html#ItemList) specific to images.
show_doc(ImageItemList, title_level=3)
# Create a [`ItemList`](/data_block.html#ItemList) in `path` from filenames in `items`. `create_func` will default to [`open_image`](/vision.image.html#open_image). `label_cls` can be specified for the labels, `xtra` contains any extra information (usually in the form of a dataframe) and `processor` is applied to the [`ItemList`](/data_block.html#ItemList) after splitting and labelling.
show_doc(ImageItemList.from_folder)
show_doc(ImageItemList.from_df)
show_doc(get_image_files)
show_doc(ImageItemList.open)
show_doc(ImageItemList.show_xys)
show_doc(ImageItemList.show_xyzs)
show_doc(ObjectCategoryList, title_level=3)
show_doc(ObjectItemList, title_level=3)
show_doc(SegmentationItemList, title_level=3)
show_doc(SegmentationLabelList, title_level=3)
show_doc(PointsLabelList)
show_doc(PointsItemList, title_level=3)
show_doc(ImageImageList, title_level=3)
# ## Building your own dataset
# This module also contains a few helper functions to allow you to build you own dataset for image classification.
show_doc(download_images)
show_doc(verify_images)
# It will try if every image in this folder can be opened and has `n_channels`. If `n_channels` is 3 – it'll try to convert image to RGB. If `delete=True`, it'll be removed it this fails. If `resume` – it will skip already existent images in `dest`.  If `max_size` is specifided, image is resized to the same ratio so that both sizes are less than `max_size`, using `interp`. Result is stored in `dest`, `ext` forces an extension type, `img_format` and `kwargs` are passed to PIL.Image.save. Use `max_workers` CPUs.
# ## Undocumented Methods - Methods moved below this line will intentionally be hidden
show_doc(PointsItemList.get)
show_doc(SegmentationLabelList.new)
show_doc(ImageItemList.from_csv)
show_doc(ObjectCategoryList.get)
show_doc(ImageItemList.get)
show_doc(SegmentationLabelList.reconstruct)
show_doc(ImageImageList.show_xys)
show_doc(ImageImageList.show_xyzs)
show_doc(ImageItemList.open)
show_doc(PointsItemList.analyze_pred)
show_doc(SegmentationLabelList.analyze_pred)
show_doc(PointsItemList.reconstruct)
show_doc(SegmentationLabelList.open)
show_doc(ImageItemList.reconstruct)
show_doc(resize_to)
show_doc(ObjectCategoryList.reconstruct)
show_doc(PointsLabelList.reconstruct)
show_doc(PointsLabelList.analyze_pred)
show_doc(PointsLabelList.get)
# ## New Methods - Please document or move to the undocumented section
