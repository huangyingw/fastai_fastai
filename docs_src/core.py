
# coding: utf-8

# # Basic core

# This module contains all the basic functions we need in other modules of the fastai library (split with [`torch_core`](/torch_core.html#torch_core) that contains the ones requiring pytorch). Its documentation can easily be skipped at a first read, unless you want to know what a given function does.

from fastai.gen_doc.nbdoc import *
from fastai.core import *


# ## Global constants

# `default_cpus = min(16, num_cpus())` <div style="text-align: right"><a href="https://github.com/fastai/fastai/blob/master/fastai/core.py#L45">[source]</a></div>

# ## Check functions

show_doc(has_arg)


# Check if `func` accepts `arg` as an argument.

show_doc(ifnone)


show_doc(is1d)


show_doc(is_listy)


# Check if `x` is a `Collection`.

show_doc(is_tuple)


# Check if `x` is a `tuple`.

show_doc(try_int)


# ## Collection related functions

show_doc(arange_of)


# Return the numpy array of the range of the same length as `x`.

show_doc(array)


show_doc(arrays_split)


show_doc(df_names_to_idx)


show_doc(extract_kwargs)


show_doc(idx_dict)


# Create a dictionary value to index from `a`.

idx_dict(['a', 'b', 'c'])


show_doc(index_row)


show_doc(listify)


show_doc(random_split)


show_doc(range_of)


# Make a `range` of the same size as `x`.

show_doc(series2cat)


show_doc(split_kwargs_by_func)


show_doc(to_int)


show_doc(uniqueify)


# ## Files management and downloads

show_doc(download_url)


show_doc(find_classes)


show_doc(join_path)


show_doc(join_paths)


show_doc(loadtxt_str)


show_doc(save_texts)


# ## Multiprocessing

show_doc(num_cpus)


show_doc(parallel)


show_doc(partition)


show_doc(partition_by_cores)


# ## Data block API

show_doc(ItemBase, title_level=3, doc_string=False)


# The base clase for all items. Must have a [`data`](/tabular.data.html#tabular.data) field that will be used when collating in mini-batches.

show_doc(ItemBase.apply_tfms)


# Subclass this method if you want to apply data augmentation to this [`ItemBase`](/core.html#ItemBase).

show_doc(ItemBase.show)


# Subclass this method if you want to customize the way this [`ItemBase`](/core.html#ItemBase) is shown on `ax` (default is to set the title to the string representation of this object).

show_doc(Category, doc_string=False, title_level=3)


# Create a [`Category`](/core.html#Category) with an `obj` of index [`data`](/tabular.data.html#tabular.data) in a certain classes list.

show_doc(EmptyLabel, title_level=3, doc_string=False)


# For dummy targets.

show_doc(MultiCategory, doc_string=False, title_level=3)


# Create a [`MultiCategory`](/core.html#MultiCategory) with an `obj` that is a collection of labels. [`data`](/tabular.data.html#tabular.data) corresponds to the one-hot encoded labels and `raw` is a list of associated string.

# ## Others

show_doc(camel2snake)


# Format `name` by removing capital letters from a class-style name and separates the subwords with underscores.

camel2snake('DeviceDataLoader')


show_doc(even_mults)


show_doc(func_args)


show_doc(noop)


# Return `x`.

show_doc(one_hot)


show_doc(text2html_table)


# ## Undocumented Methods - Methods moved below this line will intentionally be hidden

# ## New Methods - Please document or move to the undocumented section
