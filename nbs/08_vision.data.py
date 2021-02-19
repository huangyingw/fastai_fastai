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
from nbdev.export import notebook2script
from nbdev.showdoc import *
import types
from fastai.vision.core import *
from fastai.data.all import *
from fastai.torch_basics import *
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# +
# default_exp vision.data
# -

# export

# hide
# from fastai.vision.augment import *

# # Vision data
#
# > Helper functions to get data in a `DataLoaders` in the vision application and higher class `ImageDataLoaders`

# The main classes defined in this module are `ImageDataLoaders` and `SegmentationDataLoaders`, so you probably want to jump to their definitions. They provide factory methods that are a great way to quickly get your data ready for training, see the [vision tutorial](http://docs.fast.ai/tutorial.vision) for examples.

# ## Helper functions

# export
@delegates(subplots)
def get_grid(n, nrows=None, ncols=None, add_vert=0, figsize=None, double=False, title=None, return_fig=False,
             flatten=True, **kwargs):
    "Return a grid of `n` axes, `rows` by `cols`"
    if nrows:
        ncols = ncols or int(np.ceil(n / nrows))
    elif ncols:
        nrows = nrows or int(np.ceil(n / ncols))
    else:
        nrows = int(math.sqrt(n))
        ncols = int(np.ceil(n / nrows))
    if double:
        ncols *= 2
        n *= 2
    fig, axs = subplots(nrows, ncols, figsize=figsize, **kwargs)
    if flatten:
        axs = [ax if i < n else ax.set_axis_off() for i, ax in enumerate(axs.flatten())][:n]
    if title is not None:
        fig.suptitle(title, weight='bold', size=14)
    return (fig, axs) if return_fig else axs


# This is used by the type-dispatched versions of `show_batch` and `show_results` for the vision application. By default, there will be `int(math.sqrt(n))` rows and `ceil(n/rows)` columns. `double` will double the number of columns and `n`. The default `figsize` is `(cols*imsize, rows*imsize+add_vert)`. If a `title` is passed it is set to the figure. `sharex`, `sharey`, `squeeze`, `subplot_kw` and `gridspec_kw` are all passed down to `plt.subplots`. If `return_fig` is `True`, returns `fig,axs`, otherwise just `axs`. `flatten` will flatten the matplot axes such that they can be iterated over with a single loop.

# export
def clip_remove_empty(bbox, label):
    "Clip bounding boxes with image border and label background the empty ones"
    bbox = torch.clamp(bbox, -1, 1)
    empty = ((bbox[..., 2] - bbox[..., 0]) * (bbox[..., 3] - bbox[..., 1]) <= 0.)
    return (bbox[~empty], label[~empty])


bb = tensor([[-2, -0.5, 0.5, 1.5], [-0.5, -0.5, 0.5, 0.5], [1, 0.5, 0.5, 0.75], [-0.5, -0.5, 0.5, 0.5], [-2, -0.5, -1.5, 0.5]])
bb, lbl = clip_remove_empty(bb, tensor([1, 2, 3, 2, 5]))
test_eq(bb, tensor([[-1, -0.5, 0.5, 1.], [-0.5, -0.5, 0.5, 0.5], [-0.5, -0.5, 0.5, 0.5]]))
test_eq(lbl, tensor([1, 2, 2]))


# export
def bb_pad(samples, pad_idx=0):
    "Function that collect `samples` of labelled bboxes and adds padding with `pad_idx`."
    samples = [(s[0], *clip_remove_empty(*s[1:])) for s in samples]
    max_len = max([len(s[2]) for s in samples])

    def _f(img, bbox, lbl):
        bbox = torch.cat([bbox, bbox.new_zeros(max_len - bbox.shape[0], 4)])
        lbl = torch.cat([lbl, lbl .new_zeros(max_len - lbl .shape[0]) + pad_idx])
        return img, bbox, lbl
    return [_f(*s) for s in samples]


img1, img2 = TensorImage(torch.randn(16, 16, 3)), TensorImage(torch.randn(16, 16, 3))
bb1 = tensor([[-2, -0.5, 0.5, 1.5], [-0.5, -0.5, 0.5, 0.5], [1, 0.5, 0.5, 0.75], [-0.5, -0.5, 0.5, 0.5]])
lbl1 = tensor([1, 2, 3, 2])
bb2 = tensor([[-0.5, -0.5, 0.5, 0.5], [-0.5, -0.5, 0.5, 0.5]])
lbl2 = tensor([2, 2])
samples = [(img1, bb1, lbl1), (img2, bb2, lbl2)]
res = bb_pad(samples)
non_empty = tensor([True, True, False, True])
test_eq(res[0][0], img1)
test_eq(res[0][1], tensor([[-1, -0.5, 0.5, 1.], [-0.5, -0.5, 0.5, 0.5], [-0.5, -0.5, 0.5, 0.5]]))
test_eq(res[0][2], tensor([1, 2, 2]))
test_eq(res[1][0], img2)
test_eq(res[1][1], tensor([[-0.5, -0.5, 0.5, 0.5], [-0.5, -0.5, 0.5, 0.5], [0, 0, 0, 0]]))
test_eq(res[1][2], tensor([2, 2, 0]))


# ## Show methods -

# export
@typedispatch
def show_batch(x: TensorImage, y, samples, ctxs=None, max_n=10, nrows=None, ncols=None, figsize=None, **kwargs):
    if ctxs is None:
        ctxs = get_grid(min(len(samples), max_n), nrows=nrows, ncols=ncols, figsize=figsize)
    ctxs = show_batch[object](x, y, samples, ctxs=ctxs, max_n=max_n, **kwargs)
    return ctxs


# export
@typedispatch
def show_batch(x: TensorImage, y: TensorImage, samples, ctxs=None, max_n=10, nrows=None, ncols=None, figsize=None, **kwargs):
    if ctxs is None:
        ctxs = get_grid(min(len(samples), max_n), nrows=nrows, ncols=ncols, add_vert=1, figsize=figsize, double=True)
    for i in range(2):
        ctxs[i::2] = [b.show(ctx=c, **kwargs) for b, c, _ in zip(samples.itemgot(i), ctxs[i::2], range(max_n))]
    return ctxs


# ## `TransformBlock`s for vision

# These are the blocks the vision application provide for the [data block API](http://docs.fast.ai/data.block).

# export
def ImageBlock(cls=PILImage):
    "A `TransformBlock` for images of `cls`"
    return TransformBlock(type_tfms=cls.create, batch_tfms=IntToFloatTensor)


# export
def MaskBlock(codes=None):
    "A `TransformBlock` for segmentation masks, potentially with `codes`"
    return TransformBlock(type_tfms=PILMask.create, item_tfms=AddMaskCodes(codes=codes), batch_tfms=IntToFloatTensor)


# +
# export
PointBlock = TransformBlock(type_tfms=TensorPoint.create, item_tfms=PointScaler)
BBoxBlock = TransformBlock(type_tfms=TensorBBox.create, item_tfms=PointScaler, dls_kwargs={'before_batch': bb_pad})

PointBlock.__doc__ = "A `TransformBlock` for points in an image"
BBoxBlock.__doc__ = "A `TransformBlock` for bounding boxes in an image"
# -

show_doc(PointBlock, name='PointBlock')

show_doc(BBoxBlock, name='BBoxBlock')


# export
def BBoxLblBlock(vocab=None, add_na=True):
    "A `TransformBlock` for labeled bounding boxes, potentially with `vocab`"
    return TransformBlock(type_tfms=MultiCategorize(vocab=vocab, add_na=add_na), item_tfms=BBoxLabeler)


# If `add_na` is `True`, a new category is added for NaN (that will represent the background class).

# ## ImageDataLoaders -

# +
# export
class ImageDataLoaders(DataLoaders):
    "Basic wrapper around several `DataLoader`s with factory methods for computer vision problems"
    @classmethod
    @delegates(DataLoaders.from_dblock)
    def from_folder(cls, path, train='train', valid='valid', valid_pct=None, seed=None, vocab=None, item_tfms=None,
                    batch_tfms=None, **kwargs):
        "Create from imagenet style dataset in `path` with `train` and `valid` subfolders (or provide `valid_pct`)"
        splitter = GrandparentSplitter(train_name=train, valid_name=valid) if valid_pct is None else RandomSplitter(valid_pct, seed=seed)
        get_items = get_image_files if valid_pct else partial(get_image_files, folders=[train, valid])
        dblock = DataBlock(blocks=(ImageBlock, CategoryBlock(vocab=vocab)),
                           get_items=get_items,
                           splitter=splitter,
                           get_y=parent_label,
                           item_tfms=item_tfms,
                           batch_tfms=batch_tfms)
        return cls.from_dblock(dblock, path, path=path, **kwargs)

    @classmethod
    @delegates(DataLoaders.from_dblock)
    def from_path_func(cls, path, fnames, label_func, valid_pct=0.2, seed=None, item_tfms=None, batch_tfms=None, **kwargs):
        "Create from list of `fnames` in `path`s with `label_func`"
        dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                           splitter=RandomSplitter(valid_pct, seed=seed),
                           get_y=label_func,
                           item_tfms=item_tfms,
                           batch_tfms=batch_tfms)
        return cls.from_dblock(dblock, fnames, path=path, **kwargs)

    @classmethod
    def from_name_func(cls, path, fnames, label_func, **kwargs):
        "Create from the name attrs of `fnames` in `path`s with `label_func`"
        if sys.platform == 'win32' and isinstance(label_func, types.LambdaType) and label_func.__name__ == '<lambda>':
            # https://medium.com/@jwnx/multiprocessing-serialization-in-python-with-pickle-9844f6fa1812
            raise ValueError("label_func couldn't be lambda function on Windows")
        f = using_attr(label_func, 'name')
        return cls.from_path_func(path, fnames, f, **kwargs)

    @classmethod
    def from_path_re(cls, path, fnames, pat, **kwargs):
        "Create from list of `fnames` in `path`s with re expression `pat`"
        return cls.from_path_func(path, fnames, RegexLabeller(pat), **kwargs)

    @classmethod
    @delegates(DataLoaders.from_dblock)
    def from_name_re(cls, path, fnames, pat, **kwargs):
        "Create from the name attrs of `fnames` in `path`s with re expression `pat`"
        return cls.from_name_func(path, fnames, RegexLabeller(pat), **kwargs)

    @classmethod
    @delegates(DataLoaders.from_dblock)
    def from_df(cls, df, path='.', valid_pct=0.2, seed=None, fn_col=0, folder=None, suff='', label_col=1, label_delim=None,
                y_block=None, valid_col=None, item_tfms=None, batch_tfms=None, **kwargs):
        "Create from `df` using `fn_col` and `label_col`"
        pref = f'{Path(path) if folder is None else Path(path)/folder}{os.path.sep}'
        if y_block is None:
            is_multi = (is_listy(label_col) and len(label_col) > 1) or label_delim is not None
            y_block = MultiCategoryBlock if is_multi else CategoryBlock
        splitter = RandomSplitter(valid_pct, seed=seed) if valid_col is None else ColSplitter(valid_col)
        dblock = DataBlock(blocks=(ImageBlock, y_block),
                           get_x=ColReader(fn_col, pref=pref, suff=suff),
                           get_y=ColReader(label_col, label_delim=label_delim),
                           splitter=splitter,
                           item_tfms=item_tfms,
                           batch_tfms=batch_tfms)
        return cls.from_dblock(dblock, df, path=path, **kwargs)

    @classmethod
    def from_csv(cls, path, csv_fname='labels.csv', header='infer', delimiter=None, **kwargs):
        "Create from `path/csv_fname` using `fn_col` and `label_col`"
        df = pd.read_csv(Path(path) / csv_fname, header=header, delimiter=delimiter)
        return cls.from_df(df, path=path, **kwargs)

    @classmethod
    @delegates(DataLoaders.from_dblock)
    def from_lists(cls, path, fnames, labels, valid_pct=0.2, seed: int=None, y_block=None, item_tfms=None, batch_tfms=None,
                   **kwargs):
        "Create from list of `fnames` and `labels` in `path`"
        if y_block is None:
            y_block = MultiCategoryBlock if is_listy(labels[0]) and len(labels[0]) > 1 else (
                RegressionBlock if isinstance(labels[0], float) else CategoryBlock)
        dblock = DataBlock.from_columns(blocks=(ImageBlock, y_block),
                                        splitter=RandomSplitter(valid_pct, seed=seed),
                                        item_tfms=item_tfms,
                                        batch_tfms=batch_tfms)
        return cls.from_dblock(dblock, (fnames, labels), path=path, **kwargs)


ImageDataLoaders.from_csv = delegates(to=ImageDataLoaders.from_df)(ImageDataLoaders.from_csv)
ImageDataLoaders.from_name_func = delegates(to=ImageDataLoaders.from_path_func)(ImageDataLoaders.from_name_func)
ImageDataLoaders.from_path_re = delegates(to=ImageDataLoaders.from_path_func)(ImageDataLoaders.from_path_re)
ImageDataLoaders.from_name_re = delegates(to=ImageDataLoaders.from_name_func)(ImageDataLoaders.from_name_re)
# -

# This class should not be used directly, one of the factory methods should be preferred instead. All those factory methods accept as arguments:
#
# - `item_tfms`: one or several transforms applied to the items before batching them
# - `batch_tfms`: one or several transforms applied to the batches once they are formed
# - `bs`: the batch size
# - `val_bs`: the batch size for the validation `DataLoader` (defaults to `bs`)
# - `shuffle_train`: if we shuffle the training `DataLoader` or not
# - `device`: the PyTorch device to use (defaults to `default_device()`)

show_doc(ImageDataLoaders.from_folder)

# If `valid_pct` is provided, a random split is performed (with an optional `seed`) by setting aside that percentage of the data for the validation set (instead of looking at the grandparents folder). If a `vocab` is passed, only the folders with names in `vocab` are kept.
#
# Here is an example loading a subsample of MNIST:

path = untar_data(URLs.MNIST_TINY)
dls = ImageDataLoaders.from_folder(path)

# Passing `valid_pct` will ignore the valid/train folders and do a new random split:

dls = ImageDataLoaders.from_folder(path, valid_pct=0.2)
dls.valid_ds.items[:3]

show_doc(ImageDataLoaders.from_path_func)

# The validation set is a random `subset` of `valid_pct`, optionally created with `seed` for reproducibility.
#
# Here is how to create the same `DataLoaders` on the MNIST dataset as the previous example with a `label_func`:

fnames = get_image_files(path)


def label_func(x): return x.parent.name


dls = ImageDataLoaders.from_path_func(path, fnames, label_func)

# Here is another example on the pets dataset. Here filenames are all in an "images" folder and their names have the form `class_name_123.jpg`. One way to properly label them is thus to throw away everything after the last `_`:

show_doc(ImageDataLoaders.from_path_re)

# The validation set is a random subset of `valid_pct`, optionally created with `seed` for reproducibility.
#
# Here is how to create the same `DataLoaders` on the MNIST dataset as the previous example (you will need to change the initial two / by a \ on Windows):

pat = r'/([^/]*)/\d+.png$'
dls = ImageDataLoaders.from_path_re(path, fnames, pat)

show_doc(ImageDataLoaders.from_name_func)

# The validation set is a random subset of `valid_pct`, optionally created with `seed` for reproducibility. This method does the same as `ImageDataLoaders.from_path_func` except `label_func` is applied to the name of each filenames, and not the full path.

show_doc(ImageDataLoaders.from_name_re)

# The validation set is a random subset of `valid_pct`, optionally created with `seed` for reproducibility. This method does the same as `ImageDataLoaders.from_path_re` except `pat` is applied to the name of each filenames, and not the full path.

show_doc(ImageDataLoaders.from_df)

# The validation set is a random subset of `valid_pct`, optionally created with `seed` for reproducibility. Alternatively, if your `df` contains a `valid_col`, give its name or its index to that argument (the column should have `True` for the elements going to the validation set).
#
# You can add an additional `folder` to the filenames in `df` if they should not be concatenated directly to `path`. If they do not contain the proper extensions, you can add `suff`. If your label column contains multiple labels on each row, you can use `label_delim` to warn the library you have a multi-label problem.
#
# `y_block` should be passed when the task automatically picked by the library is wrong, you should then give `CategoryBlock`, `MultiCategoryBlock` or `RegressionBlock`. For more advanced uses, you should use the data block API.
#
# The tiny mnist example from before also contains a version in a dataframe:

path = untar_data(URLs.MNIST_TINY)
df = pd.read_csv(path / 'labels.csv')
df.head()

# Here is how to load it using `ImageDataLoaders.from_df`:

dls = ImageDataLoaders.from_df(df, path)

# Here is another example with a multi-label problem:

path = untar_data(URLs.PASCAL_2007)
df = pd.read_csv(path / 'train.csv')
df.head()

dls = ImageDataLoaders.from_df(df, path, folder='train', valid_col='is_valid')

# Note that can also pass `2` to valid_col (the index, starting with 0).

show_doc(ImageDataLoaders.from_csv)

# Same as `ImageDataLoaders.from_df` after loading the file with `header` and `delimiter`.
#
# Here is how to load the same dataset as before with this method:

dls = ImageDataLoaders.from_csv(path, 'train.csv', folder='train', valid_col='is_valid')

show_doc(ImageDataLoaders.from_lists)

# The validation set is a random subset of `valid_pct`, optionally created with `seed` for reproducibility. `y_block` can be passed to specify the type of the targets.

path = untar_data(URLs.PETS)
fnames = get_image_files(path / "images")
labels = ['_'.join(x.name.split('_')[:-1]) for x in fnames]
dls = ImageDataLoaders.from_lists(path, fnames, labels)


# export
class SegmentationDataLoaders(DataLoaders):
    "Basic wrapper around several `DataLoader`s with factory methods for segmentation problems"
    @classmethod
    @delegates(DataLoaders.from_dblock)
    def from_label_func(cls, path, fnames, label_func, valid_pct=0.2, seed=None, codes=None, item_tfms=None, batch_tfms=None, **kwargs):
        "Create from list of `fnames` in `path`s with `label_func`."
        dblock = DataBlock(blocks=(ImageBlock, MaskBlock(codes=codes)),
                           splitter=RandomSplitter(valid_pct, seed=seed),
                           get_y=label_func,
                           item_tfms=item_tfms,
                           batch_tfms=batch_tfms)
        res = cls.from_dblock(dblock, fnames, path=path, **kwargs)
        return res


show_doc(SegmentationDataLoaders.from_label_func)

# The validation set is a random subset of `valid_pct`, optionally created with `seed` for reproducibility. `codes` contain the mapping index to label.

# +
path = untar_data(URLs.CAMVID_TINY)
fnames = get_image_files(path / 'images')


def label_func(x): return path / 'labels' / f'{x.stem}_P{x.suffix}'


codes = np.loadtxt(path / 'codes.txt', dtype=str)

dls = SegmentationDataLoaders.from_label_func(path, fnames, label_func, codes=codes)
# -

# # Export -

# hide
notebook2script()
