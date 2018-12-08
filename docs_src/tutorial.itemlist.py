
# coding: utf-8

# # Customizing datasets in fastai

from fastai import *
from fastai.gen_doc.nbdoc import *
from fastai.vision import *


# In this tutorial, we'll see how to create custom subclasses of [`ItemBase`](/core.html#ItemBase) or [`ItemList`](/data_block.html#ItemList) while retaining everything the fastai library has to offer. To allow basic functions to work consistently across various applications, the fastai library delegates several tasks to one of those specific objets, and we'll see here which methods you have to implement to be able to have everything work properly. But first let's see take a step back to see where you'll use your end result.

# ## Links with the data block API

# The data block API works by allowing you to pick a class that is responsible to get your items and another class that is charged with getting your targets. Combined together, they create a pytorch [`Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) that is then wrapped inside a [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader). The training set, validation set and maybe test set are then all put in a [`DataBunch`](/basic_data.html#DataBunch).
#
# The data block API allows you to mix and match what class your inputs have, what clas you target have, how to do the split between train and validation set, then how to create the [`DataBunch`](/basic_data.html#DataBunch), but if you have a very specific kind of input/target, the fastai classes might no be sufficient to you. This tutorial is there to explain what is needed to create a new class of items and what methods are important to implement or override.
#
# It goes in two phases: first we focus on what you need to create a custom [`ItemBase`](/core.html#ItemBase) class (which the type of your inputs/targets) then on how to create your custom [`ItemList`](/data_block.html#ItemList) (which is basically a set of [`ItemBase`](/core.html#ItemBase)) while highlining which methods are called by the library.

# ## Creating a custom [`ItemBase`](/core.html#ItemBase) subclass

# The fastai library contains three basic type of [`ItemBase`](/core.html#ItemBase) that you might want to subclass:
# - [`Image`](/vision.image.html#Image) for vision applications
# - [`Text`](/text.data.html#Text) for text applications
# - [`TabularLine`](/tabular.data.html#TabularLine) for tabular applications
#
# Whether you decide to create your own item class or to subclass one of the above, here is what you need to implement:

# ### Basic attributes

# Those are the more importants attribute your custom [`ItemBase`](/core.html#ItemBase) needs as they're used everywhere in the fastai library:
# - `ItemBase.data` is the thing that is passed to pytorch when you want to create a [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader). This is what needs to be fed to your model. Note that it might be different from the representation of your item since you might want something that is more understandable.
# - `ItemBase.obj` is the thing that truly represents the underlying object behind your item. It should be sufficient to create a copy of your item. For instance, when creating the test set, the basic label is the `obj` attribute of the first label (or y) in the training set.
# - `__str__` representation: if applicable, this is what will be displayed when the fastai library has to show your item.
#
# If we take the example of a [`MultiCategory`](/core.html#MultiCategory) object `o` for instance:
# - `o.obj` is the list of tags that object has
# - `o.data` is a tensor where the tags are one-hot encoded
# - `str(o)` returns the tags separated by ;
#
# If you want to code the way data augmentation should be applied to your custom `Item`, you should write an `apply_tfms` method. This is what will be called if you apply a [`transform`](/vision.transform.html#vision.transform) block in the data block API.

# ### Advanced show methods

# If you want to use methods such a `data.show_batch()` or `learn.show_results()` with a brand new kind of [`ItemBase`](/core.html#ItemBase) you will need to implement two other methods. In both cases, the generic function will grab the tensors of inputs, targets and predictions (if applicable), reconstruct the corespoding [`ItemBase`](/core.html#ItemBase) (see below) but it will delegate to the [`ItemBase`](/core.html#ItemBase) the way to display the results.
#
# ``` python
# def show_xys(self, xs, ys, **kwargs)->None:
#
# def show_xyzs(self, xs, ys, zs, **kwargs)->None:
# ```
# In both cases `xs` and `ys` represent the inputs and the targets, in the second case `zs` represent the predictions. They are lists of the same length that depend on the `rows` argument you passed. The kwargs are passed from `data.show_batch()` / `learn.show_results()`. As an example, here is the source code of those methods in [`Image`](/vision.image.html#Image):
#
# ``` python
# def show_xys(self, xs, ys, figsize:Tuple[int,int]=(9,10), **kwargs):
#     "Show the `xs` and `ys` on a figure of `figsize`. `kwargs` are passed to the show method."
#     rows = int(math.sqrt(len(xs)))
#     fig, axs = plt.subplots(rows,rows,figsize=figsize)
#     for i, ax in enumerate(axs.flatten() if rows > 1 else [axs]):
#         xs[i].show(ax=ax, y=ys[i], **kwargs)
#     plt.tight_layout()
#
# def show_xyzs(self, xs, ys, zs, figsize:Tuple[int,int]=None, **kwargs):
#     """Show `xs` (inputs), `ys` (targets) and `zs` (predictions) on a figure of `figsize`.
#     `kwargs` are passed to the show method."""
#     figsize = ifnone(figsize, (6,3*len(xs)))
#     fig,axs = plt.subplots(len(xs), 2, figsize=figsize)
#     fig.suptitle('Ground truth / Predictions', weight='bold', size=14)
#     for i,(x,y,z) in enumerate(zip(xs,ys,zs)):
#         x.show(ax=axs[i,0], y=y, **kwargs)
#         x.show(ax=axs[i,1], y=z, **kwargs)
# ```

# ### Example: ImageTuple

# For cycleGANs, we need to create a custom type of items since we feed the model tuples of images. Let's look at how to code this. The basis is to code the `obj` and [`data`](/vision.data.html#vision.data) attributes. We do that in the init. The object is the tuple of images and the data their underlying tensors normalized between -1 and 1.

class ImageTuple(ItemBase):
    def __init__(self, img1, img2):
        self.img1, self.img2 = img1, img2
        self.obj, self.data = (img1, img2), [-1 + 2 * img1.data, -1 + 2 * img2.data]


# Then we want to apply data augmentation to our tuple of images. That's done by writing and `apply_tfms` method as we saw before. Here we just pass that call to the two underlying images then update the data.

def apply_tfms(self, tfms, **kwargs):
    self.img1 = self.img1.apply_tfms(tfms, **kwargs)
    self.img2 = self.img2.apply_tfms(tfms, **kwargs)
    self.data = [-1 + 2 * self.img1.data, -1 + 2 * self.img2.data]
    return self


# We define a last method to stack the two images next ot each other, which we will use later for a customized `show_batch`/ `show_results` behavior.

def to_one(self): return Image(0.5 + torch.cat(self.data, 2) / 2)


# This is all your need to create your custom [`ItemBase`](/core.html#ItemBase). You won't be able to use it until you have put it inside your custom [`ItemList`](/data_block.html#ItemList) though, so you should continue reading the next section.

# ## Creating a custom [`ItemList`](/data_block.html#ItemList) subclass

# This is the main class that allows you to group your inputs or your targets in the data block API. You can then use any of the splitting or labelling methods before creating a [`DataBunch`](/basic_data.html#DataBunch). To make sure everything is properly working, her eis what you need to know.

# ### Class variables

# Whether you're directly subclassing [`ItemList`](/data_block.html#ItemList) or one of the particular fastai ones, make sure to know the content of the following three variables as you may need to adjust them:
# - `_bunch` contains the name of the class that will be used to create a [`DataBunch`](/basic_data.html#DataBunch)
# - `_processor` contains a class (or a list of classes) of [`PreProcessor`](/data_block.html#PreProcessor) that will then be used as the default to create processor for this [`ItemList`](/data_block.html#ItemList)
# - `_label_cls` contains the class that will be used to create the labels by default
#
# `_label_cls` is the first to be used in the data block API, in the labelling function. If this variable is set to `None`, the label class will be guessed between [`CategoryList`](/data_block.html#CategoryList), [`MultiCategoryList`](/data_block.html#MultiCategoryList) and [`FloatList`](/data_block.html#FloatList) depending on the type of the first item. The default can be overriden by passing a `label_cls` in the kwargs of the labelling function.
#
# `_processor` is the second to be used. The processors are called at the end of the labelling to apply some kind of function on your items. The default processor of the inputs can be overriden by passing a `processor` in the kwargs when creating the [`ItemList`](/data_block.html#ItemList), the default processor of the targets can be overriden by passing a `processor` in the kwargs of the labelling function.
#
# Processors are useful for pre-processing some data, but you also need to put in their state any variable you want to save for the call of `data.export()` before creating a [`Learner`](/basic_train.html#Learner) object for inference: the state of the [`ItemList`](/data_block.html#ItemList) isn't saved there, only their processors. For instance `SegmentationProcessor` only reason to exist is to save the dataset classes, and during the process call, it doesn't do anything apart from setting the `classes` and `c` attributes to its dataset.
# ``` python
# class SegmentationProcessor(PreProcessor):
#     def __init__(self, ds:ItemList): self.classes = ds.classes
#     def process(self, ds:ItemList):  ds.classes,ds.c = self.classes,len(self.classes)
# ```
#
# `_bunch` is the last class variable usd in the data block. When you type the final `databunch()`, the data block API calls the `_bunch.create` method with the `_bunch` of the inputs.

# ### Keeping \_\_init\_\_ arguments

# If you pass additional arguments in your `__init__` call that you save in the state of your [`ItemList`](/data_block.html#ItemList), be wary to also pass them along in the `new` method as this one is used to create your training and validation set when splitting. The basic scheme is:
# ``` python
# class MyCustomItemList(ItemList):
#     def __init__(self, items, my_arg, **kwargs):
#         self.my_arg = my_arg
#         super().__init__(items, **kwargs)
#
#     def new(self, items, **kwargs):
#         return super().new(items, self.my_arg, **kwargs)
# ```
# Be sure to keep the kwargs as is, as they contain all the additional stuff you can pass to an [`ItemList`](/data_block.html#ItemList).

# ### Important methods

# #### - get

# The most important method you have to implement is `get`: this one will explain your custom [`ItemList`](/data_block.html#ItemList) how to general an [`ItemBase`](/core.html#ItemBase) from the thign stored in its `items` array. For instance an [`ImageItemList`](/vision.data.html#ImageItemList) has the following `get` method:
# ``` python
# def get(self, i):
#     fn = super().get(i)
#     res = self.open(fn)
#     self.sizes[i] = res.size
#     return res
# ```
# The first line basically looks at `self.items[i]` (which is a filename). The second line opens it since the `open`method is just
# ``` python
# def open(self, fn): return open_image(fn)
# ```
# The third line is there for [`ImagePoints`](/vision.image.html#ImagePoints) or [`ImageBBox`](/vision.image.html#ImageBBox) targets that require the size of the input [`Image`](/vision.image.html#Image) to be created. Note that if you are building a custom target class and you need the size of an image, you should call `self.x.size[i]`.

jekyll_note("""If you just want to customize the way an `Image` is opened, subclass `Image` and just change the
`open` method.""")


# #### - reconstruct

# This is the method that is called in `data.show_batch()`, `learn.predict()` or `learn.show_results()` to transform a pytorch tensor back in an [`ItemBase`](/core.html#ItemBase). In a way, it does the opposite of calling `ItemBase.data`. It should take a tensor `t` and return the same king of thing as the `get` method.
#
# In some situations ([`ImagePoints`](/vision.image.html#ImagePoints), [`ImageBBox`](/vision.image.html#ImageBBox) for instance) you need to have a look at the corresponding input to rebuild your item. In this case, you should have a second argument called `x` (don't change that name). For instance, here is the `reconstruct` method of [`PointsItemList`](/vision.data.html#PointsItemList):
# ```python
# def reconstruct(self, t, x): return ImagePoints(FlowField(x.size, t), scale=False)
# ```

# #### - analyze_pred

# This is the method that is called in `learn.predict()` or `learn.show_results()` to transform predictions in an output tensor suitable for `reconstruct`. For instance we may need to take the maximum argument (for [`Category`](/core.html#Category)) or the predictions greater than a certain threshold (for [`MultiCategory`](/core.html#MultiCategory)). It should take a tensor, along with optional kwargs and return a tensor.
#
# For instance, here is the `anaylze_pred` method of [`MultiCategoryList`](/data_block.html#MultiCategoryList):
# ```python
# def analyze_pred(self, pred, thresh:float=0.5): return (pred >= thresh).float()
# ```
# `thresh` can then be passed as kwarg during the calls to `learn.predict()` or `learn.show_results()`.

# ### Advanced show methods

# If you want to use methods such a `data.show_batch()` or `learn.show_results()` with a brand new kind of [`ItemBase`](/core.html#ItemBase) you will need to implement two other methods. In both cases, the generic function will grab the tensors of inputs, targets and predictions (if applicable), reconstruct the coresponding (as seen before) but it will delegate to the [`ItemList`](/data_block.html#ItemList) the way to display the results.
#
# ``` python
# def show_xys(self, xs, ys, **kwargs)->None:
#
# def show_xyzs(self, xs, ys, zs, **kwargs)->None:
# ```
# In both cases `xs` and `ys` represent the inputs and the targets, in the second case `zs` represent the predictions. They are lists of the same length that depend on the `rows` argument you passed. The kwargs are passed from `data.show_batch()` / `learn.show_results()`. As an example, here is the source code of those methods in [`ImageItemList`](/vision.data.html#ImageItemList):
#
# ``` python
# def show_xys(self, xs, ys, figsize:Tuple[int,int]=(9,10), **kwargs):
#     "Show the `xs` and `ys` on a figure of `figsize`. `kwargs` are passed to the show method."
#     rows = int(math.sqrt(len(xs)))
#     fig, axs = plt.subplots(rows,rows,figsize=figsize)
#     for i, ax in enumerate(axs.flatten() if rows > 1 else [axs]):
#         xs[i].show(ax=ax, y=ys[i], **kwargs)
#     plt.tight_layout()
#
# def show_xyzs(self, xs, ys, zs, figsize:Tuple[int,int]=None, **kwargs):
#     """Show `xs` (inputs), `ys` (targets) and `zs` (predictions) on a figure of `figsize`.
#     `kwargs` are passed to the show method."""
#     figsize = ifnone(figsize, (6,3*len(xs)))
#     fig,axs = plt.subplots(len(xs), 2, figsize=figsize)
#     fig.suptitle('Ground truth / Predictions', weight='bold', size=14)
#     for i,(x,y,z) in enumerate(zip(xs,ys,zs)):
#         x.show(ax=axs[i,0], y=y, **kwargs)
#         x.show(ax=axs[i,1], y=z, **kwargs)
# ```
#
# Linked to this method is the class variable `_show_square` of an [`ItemList`](/data_block.html#ItemList). It defaults to `False` but if it's `True`, the `show_batch` method will send `rows * rows` `xs` and `ys` to `show_xys` (so that it shows a square of inputs/targets), like here for iamges.

# ### Example: ImageTupleList

# Continuing our custom item example, we create a custom [`ItemList`](/data_block.html#ItemList) class that will wrap those `ImageTuple` properly. The first thing is to write a custom `__init__` method (since we need to list of filenames here) which means we also have to change the `new` method.

class ImageTupleList(ImageItemList):
    def __init__(self, items, itemsB=None, **kwargs):
        self.itemsB = itemsB
        super().__init__(items, **kwargs)
    
    def new(self, items, **kwargs):
        return super().new(items, itemsB=self.itemsB, **kwargs)


# We then specify how to get one item. Here we pass the image in the first list of items, and pick one randomly in the second list.

def get(self, i):
    img1 = super().get(i)
    fn = self.itemsB[random.randint(0, len(self.itemsB) - 1)]
    return ImageTuple(img1, open_image(fn))


# We also add a custom factory method to directly create an `ImageTupleList` from two folders.

@classmethod
def from_folders(cls, path, folderA, folderB, **kwargs):
    itemsB = ImageItemList.from_folder(path / folderB).items
    res = super().from_folder(path / folderA, itemsB=itemsB, **kwargs)
    res.path = path
    return res


# Finally, we have to specify how to reconstruct the `ImageTuple` from tensors if we want `show_batch` to work. We recreate the images and denormalize.

def reconstruct(self, t: Tensor):
    return ImageTuple(Image(t[0] / 2 + 0.5), Image(t[1] / 2 + 0.5))


# There is no need to write a `analyze_preds` method since the default behavior (returning the output tensor) is what we need here. However `show_results` won't work properly unless the target (which we don't really care about here) has the right `reconstruct` method: the fastai library uses the `reconstruct` method of the target on the outputs. That's why we create another custom [`ItemList`](/data_block.html#ItemList) with just that `reconstruct` method. The first line is to reconstruct our dummy targets, and the second one is the same as in `ImageTupleList`.

class TargetTupleList(ItemList):
    def reconstruct(self, t: Tensor):
        if len(t.size()) == 0: return t
        return ImageTuple(Image(t[0] / 2 + 0.5), Image(t[1] / 2 + 0.5))


# To make sure our `ImageTupleList` uses that for labelling, we pass it in `_label_cls` and this is what the result looks like.

class ImageTupleList(ImageItemList):
    _label_cls = TargetTupleList

    def __init__(self, items, itemsB=None, **kwargs):
        self.itemsB = itemsB
        super().__init__(items, **kwargs)
    
    def new(self, items, **kwargs):
        return super().new(items, itemsB=self.itemsB, **kwargs)
    
    def get(self, i):
        img1 = super().get(i)
        fn = self.itemsB[random.randint(0, len(self.itemsB) - 1)]
        return ImageTuple(img1, open_image(fn))
    
    def reconstruct(self, t: Tensor):
        return ImageTuple(Image(t[0] / 2 + 0.5), Image(t[1] / 2 + 0.5))
    
    @classmethod
    def from_folders(cls, path, folderA, folderB, **kwargs):
        itemsB = ImageItemList.from_folder(path / folderB).items
        res = super().from_folder(path / folderA, itemsB=itemsB, **kwargs)
        res.path = path
        return res


# Lastly, we want to customize the behavior of `show_batch` and `show_results`. Remember the `to_one` method just puts the two images next to each other.

def show_xys(self, xs, ys, figsize: Tuple[int, int]=(12, 6), **kwargs):
    "Show the `xs` and `ys` on a figure of `figsize`. `kwargs` are passed to the show method."
    rows = int(math.sqrt(len(xs)))
    fig, axs = plt.subplots(rows, rows, figsize=figsize)
    for i, ax in enumerate(axs.flatten() if rows > 1 else [axs]):
        xs[i].to_one().show(ax=ax, **kwargs)
    plt.tight_layout()

def show_xyzs(self, xs, ys, zs, figsize: Tuple[int, int]=None, **kwargs):
    """Show `xs` (inputs), `ys` (targets) and `zs` (predictions) on a figure of `figsize`.
    `kwargs` are passed to the show method."""
    figsize = ifnone(figsize, (12, 3 * len(xs)))
    fig, axs = plt.subplots(len(xs), 2, figsize=figsize)
    fig.suptitle('Ground truth / Predictions', weight='bold', size=14)
    for i, (x, z) in enumerate(zip(xs, zs)):
        x.to_one().show(ax=axs[i, 0], **kwargs)
        z.to_one().show(ax=axs[i, 1], **kwargs)
