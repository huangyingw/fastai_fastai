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
# ## Collection related functions
show_doc(arange_of)
show_doc(array)
show_doc(arrays_split)
show_doc(chunks)
# You can transform a `Collection` into an `Iterable` of 'n' sized chunks by calling [`chunks`](/core.html#chunks):
ls = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
for chunk in chunks(ls, 2):
    print(chunk)
show_doc(df_names_to_idx)
show_doc(extract_kwargs)
show_doc(idx_dict)
idx_dict(['a', 'b', 'c'])
show_doc(index_row)
show_doc(listify)
show_doc(random_split)
show_doc(range_of)
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
show_doc(ItemBase, title_level=3)
# All items used in fastai should subclass this. Must have a [`data`](/data.html#data) field that will be used when collating in mini-batches.
show_doc(ItemBase.apply_tfms)
show_doc(ItemBase.show)
# The default behavior is to set the string representation of this object as title of `ax`.
show_doc(Category, title_level=3)
# Create a [`Category`](/core.html#Category) with an `obj` of index [`data`](/data.html#data) in a certain classes list.
show_doc(EmptyLabel, title_level=3)
show_doc(MultiCategory, title_level=3)
# Create a [`MultiCategory`](/core.html#MultiCategory) with an `obj` that is a collection of labels. [`data`](/data.html#data) corresponds to the one-hot encoded labels and `raw` is a list of associated string.
# ## Others
show_doc(camel2snake)
camel2snake('DeviceDataLoader')
show_doc(even_mults)
show_doc(func_args)
show_doc(noop)
# Return `x`.
show_doc(one_hot)
show_doc(subplots)
show_doc(text2html_table)
# ## Undocumented Methods - Methods moved below this line will intentionally be hidden
# ## New Methods - Please document or move to the undocumented section
