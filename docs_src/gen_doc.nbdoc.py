
# coding: utf-8

# # Documentation notebook functions

from fastai.gen_doc.nbdoc import *


# [`nbdoc`](/gen_doc.nbdoc.html#gen_doc.nbdoc) contains the functions for documentation notebooks. The most important is [`show_doc`](/gen_doc.nbdoc.html#show_doc):

# ## Show the documentation of a function

show_doc(show_doc, doc_string=False)


# Show the documentation of an `elt` (function, class or enum). `doc_string` decices if we show the doc string of the element or not, `full_name` will override the name shown, `arg_comments` is a dictionary that will then show list the arguments with comments. `title_level` is the level of the corresponding cell in the TOC, `alt_doc_string` is a text that can replace the `doc_string`. `ignore_warn` will ignore warnings if you pass arguments in `arg_comments` that don't appear to belong to this function and `markdown` decides if the return is a Markdown cell or plain text.
#
# Plenty of examples of uses of this cell can been seen through the documentation, and you will want to *hide input* those cells for a clean final result.

# ## Convenience functions

show_doc(get_source_link)


show_doc(show_video)


show_doc(show_video_from_youtube)


# ## Functions for internal fastai library use

show_doc(get_exports)


# Get the exports of `mod`.

show_doc(get_fn_link)


show_doc(get_ft_names)


show_doc(is_enum)


# Check if something is an enumerator.

show_doc(import_mod)


show_doc(link_docstring)


# ## Undocumented Methods - Methods moved below this line will intentionally be hidden

# ## New Methods - Please document or move to the undocumented section

show_doc(jekyll_important)


show_doc(jekyll_warn)


show_doc(jekyll_note)


show_doc(doc)
