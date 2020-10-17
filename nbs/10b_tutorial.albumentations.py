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
from albumentations import ShiftScaleRotate
from fastai.vision.all import *
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# +
# all_slow
# -

# # Tutorial - Custom transforms
#
# > Using `Datasets`, `Pipeline`, `TfmdLists` and `Transform` in computer vision

# ## Overview


# ### Creating your own `Transform`

# Creating your own `Transform` is way easier than you think. In fact, each time you have passed a label function to the data block API or to `ImageDataLoaders.from_name_func`, you have created a `Transform` without knowing it. At its base, a `Transform` is just a function. Let's show how you can easily add a transform by implementing one that wraps a data augmentation from the [albumentations library](https://github.com/albumentations-team/albumentations).
#
# First things first, you will need to install the albumentations library. Uncomment the following cell to do so if needed:

# +
# -

# Then it's going to be easier to see the result of the transform on a color image bigger than the mnist one we had before, so let's load something from the PETS dataset.

source = untar_data(URLs.PETS)
items = get_image_files(source / "images")

# We can still open it with `PILIlmage.create`:

img = PILImage.create(items[0])
img

# We will show how to wrap one transform, but you can as easily wrap any set of transforms you wrapped in a `Compose` method. Here let's do some `ShiftScaleRotate`:


# The albumentations transform work on numpy images, so we just convert our `PILImage` to a numpy array before wrapping it back in `PILImage.create` (this function takes filenames as well as arrays or tensors).

aug = ShiftScaleRotate(p=1)
def aug_tfm(img):
    np_img = np.array(img)
    aug_img = aug(image=np_img)['image']
    return PILImage.create(aug_img)


aug_tfm(img)

# We can pass this function each time a `Transform` is expected and the fastai library will automatically do the conversion. That's because you can directly pass such a function to create a `Transform`:

tfm = Transform(aug_tfm)


# If you have some state in your transform, you might want to create a subclass of `Transform`. In that case, the function you want to apply should be written in the <code>encodes</code> method (the same way you implement `forward` for PyTorch module):

class AlbumentationsTransform(Transform):
    def __init__(self, aug): self.aug = aug
    def encodes(self, img: PILImage):
        aug_img = self.aug(image=np.array(img))['image']
        return PILImage.create(aug_img)


# We also added a type annotation: this will make sure this transform is only applied to `PILImage`s and their subclasses. For any other object, it won't do anything. You can also write as many <code>encodes</code> method you want with different type-annotations and the `Transform` will properly dispatch the objects it receives.
#
# This is because in practice, the transform is often applied as an `item_tfms` (or a `batch_tfms`) that you pass in the data block API. Those items are a tuple of objects of different types, and the transform may have different behaviors on each part of the tuple.
#
# Let's check here how this works:

tfm = AlbumentationsTransform(ShiftScaleRotate(p=1))
a, b = tfm((img, 'dog'))
show_image(a, title=b)

# The transform was applied over the tuple `(img, "dog")`. `img` is a `PILImage`, so it applied the <code>encodes</code> method we wrote. `"dog"` is a string, so the transform did nothing to it.
#
# Sometimes however, you need your transform to take your tuple as whole: for instance albumentations is applied simultaneously on images and segmentation masks. In this case you need to subclass `ItemTransfrom` instead of `Transform`. Let's see how this works:

cv_source = untar_data(URLs.CAMVID_TINY)
cv_items = get_image_files(cv_source / 'images')
img = PILImage.create(cv_items[0])
mask = PILMask.create(cv_source / 'labels' / f'{cv_items[0].stem}_P{cv_items[0].suffix}')
ax = img.show()
ax = mask.show(ctx=ax)


# We then write a subclass of `ItemTransform` that can wrap any albumentations augmentation transform, but only for a segmentation problem:

class SegmentationAlbumentationsTransform(ItemTransform):
    def __init__(self, aug): self.aug = aug
    def encodes(self, x):
        img, mask = x
        aug = self.aug(image=np.array(img), mask=np.array(mask))
        return PILImage.create(aug["image"]), PILMask.create(aug["mask"])


# And we can check how it gets applied on the tuple `(img, mask)`. This means you can pass it as an `item_tfms` in any segmentation problem.

tfm = SegmentationAlbumentationsTransform(ShiftScaleRotate(p=1))
a, b = tfm((img, mask))
ax = a.show()
ax = b.show(ctx=ax)

# ### Segmentation

# By using the same transforms in `after_item` but a different kind of targets (here segmentation masks), the targets are automatically processed as they should with the type-dispatch system.

cv_source = untar_data(URLs.CAMVID_TINY)
cv_items = get_image_files(cv_source / 'images')
cv_splitter = RandomSplitter(seed=42)
cv_split = cv_splitter(cv_items)
def cv_label(o): return cv_source / 'labels' / f'{o.stem}_P{o.suffix}'


class ImageResizer(Transform):
    order = 1
    "Resize image to `size` using `resample`"
    def __init__(self, size, resample=Image.BILINEAR):
        if not is_listy(size):
            size = (size, size)
        self.size, self.resample = (size[1], size[0]), resample

    def encodes(self, o: PILImage): return o.resize(size=self.size, resample=self.resample)
    def encodes(self, o: PILMask): return o.resize(size=self.size, resample=Image.NEAREST)


tfms = [[PILImage.create], [cv_label, PILMask.create]]
cv_dsets = Datasets(cv_items, tfms, splits=cv_split)
dls = cv_dsets.dataloaders(bs=64, after_item=[ImageResizer(128), ToTensor(), IntToFloatTensor()])


# If we want to use the augmentation transform we created before, we just need to add one thing to it: we want it to be applied on the training set only, not the validation set. To do this, we specify it should only be applied on a specific `idx` of our splits by adding `split_idx=0` (0 is for the training set, 1 for the validation set):

class SegmentationAlbumentationsTransform(ItemTransform):
    split_idx = 0
    def __init__(self, aug): self.aug = aug
    def encodes(self, x):
        img, mask = x
        aug = self.aug(image=np.array(img), mask=np.array(mask))
        return PILImage.create(aug["image"]), PILMask.create(aug["mask"])


# And we can check how it gets applied on the tuple `(img, mask)`. This means you can pass it as an `item_tfms` in any segmentation problem.

cv_dsets = Datasets(cv_items, tfms, splits=cv_split)
dls = cv_dsets.dataloaders(bs=64, after_item=[ImageResizer(128), ToTensor(), IntToFloatTensor(),
                                              SegmentationAlbumentationsTransform(ShiftScaleRotate(p=1))])

dls.show_batch(max_n=4)

# ## fin -
