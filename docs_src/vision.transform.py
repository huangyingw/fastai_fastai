# coding: utf-8
# # Image transforms
from fastai.gen_doc.nbdoc import *
from fastai.vision import *
# fastai provides a complete image transformation library written from scratch in PyTorch. Although the main purpose of the library is for data augmentation when training computer vision models, you can also use it for more general image transformation purposes. Before we get in to the detail of the full API, we'll look at a quick overview of the data augmentation pieces that you'll almost certainly need to use.
# ## Data augmentation
# Data augmentation is perhaps the most important regularization technique when training a model for Computer Vision: instead of feeding the model with the same pictures every time, we do small random transformations (a bit of rotation, zoom, translation, etc...) that don't change what's inside the image (for the human eye) but change its pixel values. Models trained with data augmentation will then generalize better.
#
# To get a set of transforms with default values that work pretty well in a wide range of tasks, it's often easiest to use [`get_transforms`](/vision.transform.html#get_transforms). Depending on the nature of the images in your data, you may want to adjust a few arguments, the most important being:
#
# - `do_flip`: if True the image is randomly flipped (default beheavior)
# - `flip_vert`: limit the flips to horizontal flips (when False) or to horizontal and vertical flips as well as 90-degrees rotations (when True)
#
# [`get_transforms`](/vision.transform.html#get_transforms) returns a tuple of two list of transforms: one for the training set and one for the validation set (we don't want to modify the pictures in the validation set, so the second list of transforms is limited to resizing the pictures). This can be then passed directly to define a [`DataBunch`](/basic_data.html#DataBunch) object (see below) which is then associated with a model to begin training.
#
# Note that the defaults got [`get_transforms`](/vision.transform.html#get_transforms) are generally pretty good for regular photos - although here we'll add a bit of extra rotation so it's easier to see the differences.
tfms = get_transforms(max_rotate=25)
len(tfms)
# We first define here a function to return a new image, since transformation functions modify their inputs. We also define a little helper function `plots_f` to let us output a grid of transformed images based on a function - the details of this function aren't important here.
def get_ex(): return open_image('imgs/cat_example.jpg')
def plots_f(rows, cols, width, height, **kwargs):
    [get_ex().apply_tfms(tfms[0], **kwargs).show(ax=ax) for i, ax in enumerate(plt.subplots(
        rows, cols, figsize=(width, height))[1].flatten())]
# If we want to have a look at what this transforms actually do, we need to use the [`apply_tfms`](/vision.image.html#apply_tfms) function. It will be in charge of picking the values of the random parameters and doing the transformation to the [`Image`](/vision.image.html#Image) object. This function has multiple arguments you can customize (see its documentation for details), we will highlight here the most useful. The first one we'll need to set, especially if our images are of different shapes, is the target `size`. It will ensure all the images are cropped or padded to the same size so we can then collate them into batches.
plots_f(2, 4, 12, 6, size=224)
# Note that the target `size` can be a rectangle if you specify a tuple of int (height by width).
plots_f(2, 4, 12, 8, size=(300, 200))
# The second argument that can be customized is how we treat missing pixels: when applying transforms (like a rotation), some of the pixels inside the square won't have values from the image. We can set missing pixels to one of the following:
# - black (`padding_mode`='zeros')
# - the value of the pixel at the nearest border (`padding_mode`='border')
# - the value of the pixel symmetric to the nearest border (`padding_mode`='reflection')
#
# `padding_mode`='reflection' is the default. Here is what `padding_mode`='zeros' looks like this:
plots_f(2, 4, 12, 6, size=224, padding_mode='zeros')
# And here is what `padding_mode`='border' looks like this:
plots_f(2, 4, 12, 6, size=224, padding_mode='border')
# The third argument that might be useful to change is `resize_method`. Images are often rectangles of different ratios, so to get them to the target `size`, we can either/ By default, the library resize the image while keeping its original ratio so that the smaller size corresponds to the given size, then takes a crop (<code>ResizeMethod.CROP</code>). You can choose to resize the image while keeping its original ratio so that the bigger size corresponds to the given size, then take a pad (<code>ResizeeMethod.PAD</code>). Another way is to just squish the image to the given size (<code>ResizeeMethod.SQUISH</code>).
_, axs = plt.subplots(1, 3, figsize=(9, 3))
for rsz, ax in zip([ResizeMethod.CROP, ResizeMethod.PAD, ResizeMethod.SQUISH], axs):
    get_ex().apply_tfms([crop_pad()], size=224, resize_method=rsz, padding_mode='zeros').show(ax=ax, title=rsz.name.lower())
# ## Data augmentation details
# If you want to quickly get a set of random transforms that have proved to work well in a wide range of tasks, you should use the [`get_transforms`](/vision.transform.html#get_transforms) function. The most important parameters to adjust are *do\_flip* and *flip\_vert*, depending on the type of images you have.
show_doc(get_transforms, arg_comments={
    'do_flip': 'if True, a random flip is applied with probability 0.5',
    'flip_vert': 'requires do_flip=True. If True, the image can be flipped vertically or rotated of 90 degrees, otherwise only an horizontal flip is applied',
    'max_rotate': 'if not None, a random rotation between -max\_rotate and max\_rotate degrees is applied with probability p\_affine',
    'max_zoom': 'if not 1. or less, a random zoom betweem 1. and max\_zoom is applied with probability p\_affine',
    'max_lighting': 'if not None, a random lightning and contrast change controlled by max\_lighting is applied with probability p\_lighting',
    'max_warp': 'if not None, a random symmetric warp of magnitude between -max\_warp and maw\_warp is applied with probability p\_affine',
    'p_affine': 'the probability that each affine transform and symmetric warp is applied',
    'p_lighting': 'the probability that each lighting transform is applied',
    'xtra_tfms': 'a list of additional transforms you would like to be applied'
})
# This function returns a tuple of two list of transforms, one for the training set and the other for the validation set (which is limited to a center crop by default.
tfms = get_transforms(max_rotate=25); len(tfms)
# Let's see how [`get_transforms`](/vision.transform.html#get_transforms) changes this little kitten now.
plots_f(2, 4, 12, 6, size=224)
# Another useful function that gives basic transforms is [`zoom_crop`](/vision.transform.html#zoom_crop):
show_doc(zoom_crop, arg_comments={
    'scale': 'Decimal or range of decimals to zoom the image',
    'do_rand': "If true, transform is randomized, otherwise it's a `zoom` of `scale` and a center crop",
    'p': 'Probability to apply the zoom'
})
# `scale` should be a given float if `do_rand` is false, otherwise it can be a range of floats (and the zoom will have a random value inbetween). Again, here is a sense of what this can give us.
tfms = zoom_crop(scale=(0.75, 2), do_rand=True)
plots_f(2, 4, 12, 6, size=224)
show_doc(rand_resize_crop, ignore_warn=True, arg_comments={
    'size': 'Final size of the image',
    'max_scale': 'Zooms the image to a random scale up to this',
    'ratios': 'Range of ratios in which a new one will be randomly picked'
})
# This transform is an implementation of the main approach used for nearly all winning Imagenet entries since 2013, based on Andrew Howard's [Some Improvements on Deep Convolutional Neural Network Based Image Classification](https://arxiv.org/abs/1312.5402). It determines a new width and height of the image after the random scale and squish to the new ratio are applied. Those are switched with probability 0.5. Then we return the part of the image with the width and height computed, centered in `row_pct`, `col_pct` if width and height are both less than the corresponding size of the image. Otherwise we try again with new random parameters.
tfms = [rand_resize_crop(224)]
plots_f(2, 4, 12, 6, size=224)
# ## Randomness
# The functions that define each transform, such as [`rotate`](/vision.transform.html#_rotate)or [`flip_lr`](/vision.transform.html#_flip_lr) are deterministic. The fastai library will then randomize them in two different ways:
# - each transform can be defined with an argument named `p` representing the probability for it to be applied
# - each argument that is type-annoted with a random function (like [`uniform`](/torch_core.html#uniform) or [<code>rand_bool</code>](http://docs.fast.ai/vision.image.html#rand_bool)) can be replaced by a tuple of arguments accepted by this function, and on each call of the transform, the argument that is passed inside the function will be picked randomly using that random function.
#
# If we look at the function [`rotate`](/vision.transform.html#_rotate) for instance, we see it had an argument `degrees` that is type-annotated as uniform.
#
# **First level of randomness:** We can define a transform using [`rotate`](/vision.transform.html#_rotate) with `degrees` fixed to a value, but by passing an argument `p`. The rotation will then be executed with a probability of `p` but always with the same value of `degrees`.
tfm = [rotate(degrees=30, p=0.5)]
fig, axs = plt.subplots(1, 5, figsize=(12, 4))
for ax in axs:
    img = get_ex().apply_tfms(tfm)
    title = 'Done' if tfm[0].do_run else 'Not done'
    img.show(ax=ax, title=title)
# **Second level of randomness**: We can define a transform using [`rotate`](/vision.transform.html#_rotate) with `degrees` defined as a range, without an argument `p`. The rotation will then always be executed with a random value picked uniformly between the two floats we put in `degrees`.
tfm = [rotate(degrees=(-30, 30))]
fig, axs = plt.subplots(1, 5, figsize=(12, 4))
for ax in axs:
    img = get_ex().apply_tfms(tfm)
    title = f"deg={tfm[0].resolved['degrees']:.1f}"
    img.show(ax=ax, title=title)
# **All combined**: We can define a transform using [`rotate`](/vision.transform.html#_rotate) with `degrees` defined as a range, and an argument `p`. The rotation will then always be executed with a probability `p` and a random value picked uniformly between the two floats we put in `degrees`.
tfm = [rotate(degrees=(-30, 30), p=0.75)]
fig, axs = plt.subplots(1, 5, figsize=(12, 4))
for ax in axs:
    img = get_ex().apply_tfms(tfm)
    title = f"Done, deg={tfm[0].resolved['degrees']:.1f}" if tfm[0].do_run else f'Not done'
    img.show(ax=ax, title=title)
# ## List of transforms
# Here is the list of all the deterministic functions on which the transforms are built. As explained before, each of those can have a probability `p` of being executed, and any time an argument is type-annotated with a random function, it's possible to randomize it via that function.
show_doc(brightness)
# This transform adjusts the brightness of the image depending on the value in `change`. A `change` of 0 will transform the image to black and a `change` of 1 will transform the image to white. `change`=0.5 doesn't do adjust the brightness.
fig, axs = plt.subplots(1, 5, figsize=(12, 4))
for change, ax in zip(np.linspace(0.1, 0.9, 5), axs):
    brightness(get_ex(), change).show(ax=ax, title=f'change={change:.1f}')
show_doc(contrast)
# `scale` adjusts the contrast. A `scale` of 0 will transform the image to grey and a `scale` over 1 will transform the picture to super-contrast. `scale` = 1. doesn't adjust the contrast.
fig, axs = plt.subplots(1, 5, figsize=(12, 4))
for scale, ax in zip(np.exp(np.linspace(log(0.5), log(2), 5)), axs):
    contrast(get_ex(), scale).show(ax=ax, title=f'scale={scale:.2f}')
show_doc(crop)
# This transform takes a crop of the image to return one of the given size. The position is given by `(col_pct, row_pct)`, with `col_pct` and `row_pct` being normalized between 0. and 1.
fig, axs = plt.subplots(1, 5, figsize=(12, 4))
for center, ax in zip([[0., 0.], [0., 1.], [0.5, 0.5], [1., 0.], [1., 1.]], axs):
    crop(get_ex(), 300, *center).show(ax=ax, title=f'center=({center[0]}, {center[1]})')
show_doc(crop_pad, ignore_warn=True, arg_comments={
    'x': 'Image to transform',
    'size': "Size of the crop, if it's an int, the crop will be square",
    'padding_mode': "How to pad the output image ('zeros', 'border' or 'reflection')",
    'row_pct': 'Between 0. and 1., position of the center on the y axis (0. is top, 1. is bottom, 0.5 is center)',
    'col_pct': 'Between 0. and 1., position of the center on the x axis (0. is left, 1. is right, 0.5 is center)'
})
# This works like [`crop`](/vision.transform.html#_crop) but if the target size is bigger than the size of the image (on either dimension), padding is applied according to `padding_mode` (see [`pad`](/vision.transform.html#_pad) for an example of all the options) and the position of center is ignored on that dimension.
fig, axs = plt.subplots(1, 5, figsize=(12, 4))
for size, ax in zip(np.linspace(200, 600, 5), axs):
    crop_pad(get_ex(), int(size), 'zeros', 0., 0.).show(ax=ax, title=f'size = {int(size)}')
show_doc(dihedral)
# This transform applies combines a flip (horizontal or vertical) and a rotation of a multiple of 90 degrees.
fig, axs = plt.subplots(2, 4, figsize=(12, 8))
for k, ax in enumerate(axs.flatten()):
    dihedral(get_ex(), k).show(ax=ax, title=f'k={k}')
plt.tight_layout()
show_doc(dihedral_affine)
# This is an affine implementation of [`dihedral`](/vision.transform.html#_dihedral) that should be used if the target is an [`ImagePoints`](/vision.image.html#ImagePoints) or an [`ImageBBox`](/vision.image.html#ImageBBox).
show_doc(flip_lr)
# This transform horizontally flips the image. [`flip_lr`](/vision.transform.html#_flip_lr) mirrors the image.
fig, axs = plt.subplots(1, 2, figsize=(6, 4))
get_ex().show(ax=axs[0], title=f'no flip')
flip_lr(get_ex()).show(ax=axs[1], title=f'flip')
show_doc(flip_affine)
# This is an affine implementation of [`flip_lr`](/vision.transform.html#_flip_lr) that should be used if the target is an [`ImagePoints`](/vision.image.html#ImagePoints) or an [`ImageBBox`](/vision.image.html#ImageBBox).
show_doc(jitter, doc_string=False)
# This transform changes the pixels of the image by randomly replacing them with pixels from the neighborhood (how far the neighborhood extends is controlled by the value of `magnitude`).
fig, axs = plt.subplots(1, 5, figsize=(12, 4))
for magnitude, ax in zip(np.linspace(-0.05, 0.05, 5), axs):
    tfm = jitter(magnitude=magnitude)
    get_ex().jitter(magnitude).show(ax=ax, title=f'magnitude={magnitude:.2f}')
show_doc(pad)
# Pad the image by adding `padding` pixel on each side of the picture accordin to `mode`:
# - `mode=zeros`:  pads with zeros,
# - `mode=border`: repeats the pixels at the border.
# - `mode=reflection`: pads by taking the pixels symmetric to the border.
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
for mode, ax in zip(['zeros', 'border', 'reflection'], axs):
    pad(get_ex(), 50, mode).show(ax=ax, title=f'mode={mode}')
show_doc(perspective_warp)
# Perspective warping is a deformation of the image as seen in a different plane of the 3D-plane. The new plane is determined by telling where we want each of the four corners of the image (from -1 to 1, -1 being left/top, 1 being right/bottom).
fig, axs = plt.subplots(2, 4, figsize=(12, 8))
for i, ax in enumerate(axs.flatten()):
    magnitudes = torch.tensor(np.zeros(8))
    magnitudes[i] = 0.5
    perspective_warp(get_ex(), magnitudes).show(ax=ax, title=f'coord {i}')
show_doc(rotate)
fig, axs = plt.subplots(1, 5, figsize=(12, 4))
for deg, ax in zip(np.linspace(-60, 60, 5), axs):
    get_ex().rotate(degrees=deg).show(ax=ax, title=f'degrees={deg}')
show_doc(skew)
fig, axs = plt.subplots(2, 4, figsize=(12, 8))
for i, ax in enumerate(axs.flatten()):
    get_ex().skew(i, 0.2).show(ax=ax, title=f'direction={i}')
show_doc(squish)
fig, axs = plt.subplots(1, 5, figsize=(12, 4))
for scale, ax in zip(np.linspace(0.66, 1.33, 5), axs):
    get_ex().squish(scale=scale).show(ax=ax, title=f'scale={scale:.2f}')
show_doc(symmetric_warp)
# Apply the four tilts at the same time, each with a strength given in the vector `magnitude`. See [`tilt`](/vision.transform.html#_tilt) just below for the effect of each individual tilt.
tfm = symmetric_warp(magnitude=(-0.2, 0.2))
_, axs = plt.subplots(2, 4, figsize=(12, 6))
for ax in axs.flatten():
    img = get_ex().apply_tfms(tfm, padding_mode='zeros')
    img.show(ax=ax)
show_doc(tilt)
# `direction` is a number (0: left, 1: right, 2: top, 3: bottom). A positive `magnitude` is a tilt forward (toward the person looking at the picture), a negative `magnitude` a tilt backward.
fig, axs = plt.subplots(2, 4, figsize=(12, 8))
for i in range(4):
    get_ex().tilt(i, 0.4).show(ax=axs[0, i], title=f'direction={i}, fwd')
    get_ex().tilt(i, -0.4).show(ax=axs[1, i], title=f'direction={i}, bwd')
show_doc(zoom)
fig, axs = plt.subplots(1, 5, figsize=(12, 4))
for scale, ax in zip(np.linspace(1., 1.5, 5), axs):
    get_ex().zoom(scale=scale).show(ax=ax, title=f'scale={scale:.2f}')
# ## Convenience functions
# These functions simplify creating random versions of [`crop_pad`](/vision.transform.html#_crop_pad) and [`zoom`](/vision.transform.html#_zoom).
show_doc(rand_crop)
tfm = rand_crop()
_, axs = plt.subplots(2, 4, figsize=(12, 6))
for ax in axs.flatten():
    img = get_ex().apply_tfms(tfm, size=224)
    img.show(ax=ax)
show_doc(rand_pad)
tfm = rand_pad(4, 224)
_, axs = plt.subplots(2, 4, figsize=(12, 6))
for ax in axs.flatten():
    img = get_ex().apply_tfms(tfm, size=224)
    img.show(ax=ax)
show_doc(rand_zoom)
tfm = rand_zoom(scale=(1., 1.5))
_, axs = plt.subplots(2, 4, figsize=(12, 6))
for ax in axs.flatten():
    img = get_ex().apply_tfms(tfm)
    img.show(ax=ax)
# ## Undocumented Methods - Methods moved below this line will intentionally be hidden
# ## New Methods - Please document or move to the undocumented section
