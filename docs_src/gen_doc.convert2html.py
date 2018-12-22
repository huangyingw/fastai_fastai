# coding: utf-8
# # Conversion notebook to HTML
# This is the module to convert a jupyter notebook to an html page. It will normally render everything the way it was in the notebooks, including suppressing the input of the code cells with input cells masked, and converting the links between notebooks in links between the html pages.
from fastai.gen_doc.nbdoc import *
from fastai.gen_doc.convert2html import *
# ## Functions
show_doc(convert_nb)
show_doc(convert_all)
# Here's an example to convert all the docs in this folder:
#
# ``` python
# convert_all('.', '../docs')
# ```
# ## Undocumented Methods - Methods moved below this line will intentionally be hidden
# ## New Methods - Please document or move to the undocumented section
show_doc(read_nb)
