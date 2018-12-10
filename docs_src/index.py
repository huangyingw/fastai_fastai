
# coding: utf-8

# # Welcome to fastai

from fastai import *
from fastai.vision import *
from fastai.gen_doc.nbdoc import *
from fastai.core import *
from fastai.basic_train import *


# The fastai library simplifies training fast and accurate neural nets using modern best practices. It's based on research in to deep learning best practices undertaken at [fast.ai](http://www.fast.ai), including "out of the box" support for [`vision`](/vision.html#vision), [`text`](/text.html#text), [`tabular`](/tabular.html#tabular), and [`collab`](/collab.html#collab) (collaborative filtering) models. If you're looking for the source code, head over to the [fastai repo](https://github.com/fastai/fastai) on GitHub. For brief examples, see the [examples](https://github.com/fastai/fastai/tree/master/examples) folder; detailed examples are provided in the full documentation (see the sidebar). For example, here's how to train an MNIST model using [resnet18](https://arxiv.org/abs/1512.03385) (from the [vision example](https://github.com/fastai/fastai/blob/master/examples/vision.ipynb)):

os.chdir(os.path.dirname(os.path.realpath(__file__)))
path = untar_data(URLs.MNIST_SAMPLE)
data = ImageDataBunch.from_folder(path)
learn = create_cnn(data, models.resnet18, metrics=accuracy)
learn.fit(1, saved_model_name='index_1')


jekyll_note("""This documentation is all built from notebooks;
that means that you can try any of the code you see in any notebook yourself!
You'll find the notebooks in the <a href="https://github.com/fastai/fastai/tree/master/docs_src">docs_src</a> folder of the
<a href="https://github.com/fastai/fastai">fastai</a> repo. For instance,
<a href="https://nbviewer.jupyter.org/github/fastai/fastai/blob/master/docs_src/index.ipynb">here</a>
is the notebook source of what you're reading now.""")


# ## Installation and updating

# To install or update fastai, we recommend `conda`:
#
# ```
# conda install -c pytorch -c fastai fastai pytorch
# ```
# For troubleshooting, and alternative installations (including pip and CPU-only options) see the [fastai readme](https://github.com/fastai/fastai/blob/master/README.md).

# ## Reading the docs

# To get started quickly, click *Applications* on the sidebar, and then choose the application you're interested in. That will take you to a walk-through of training a model of that type. You can then either explore the various links from there, or dive more deeply into the various fastai modules.
#
# We've provided below a quick summary of the key modules in this library. For details on each one, use the sidebar to find the module you're interested in. Each module includes an overview and example of how to use it, along with documentation for every class, function, and method. API documentation looks, for example, like this:
#
# ### An example function

show_doc(rotate)


# ---
#
# Types for each parameter, and the return type, are displayed following standard Python [type hint syntax](https://www.python.org/dev/peps/pep-0484/). Sometimes for compound types we use [type variables](/fastai_typing.html). Types that are defined by fastai or Pytorch link directly to more information about that type; try clicking *Image* in the function above for an example. The docstring for the symbol is shown immediately after the signature, along with a link to the source code for the symbol in GitHub. After the basic signature and docstring you'll find examples and additional details (not shown in this example). As you'll see at the top of the page, all symbols documented like this also appear in the table of contents.
#
# For inherited classes and some types of decorated function, the base class or decorator type will also be shown at the end of the signature, delimited by `::`. For `vision.transforms`, the random number generator used for data augmentation is shown instead of the type, for randomly generated parameters.

# ## Module structure

# ### Imports

# fastai is designed to support both interactive computing as well as traditional software development. For interactive computing, where convenience and speed of experimentation is a priority, data scientists often prefer to grab all the symbols they need, with `import *`. Therefore, fastai is designed to support this approach, without compromising on maintainability and understanding.
#
# In order to do so, the module dependencies are carefully managed (see next section), with each exporting a carefully chosen set of symbols when using `import *`. In general, for interactive computing, you'll want to import from both `fastai`, and from one of the *applications*, such as:

from fastai import *
from fastai.vision import *


# That will give you all the standard external modules you'll need, in their customary namespaces (e.g. `pandas as pd`, `numpy as np`, `matplotlib.pyplot as plt`), plus the core fastai libraries. In addition, the main classes and functions for your application ([`fastai.vision`](/vision.html#vision), in this case), e.g. creating a [`DataBunch`](/basic_data.html#DataBunch) from an image folder and training a convolutional neural network (with [`create_cnn`](/vision.learner.html#create_cnn)), are also imported. Alternatively, use `from fastai import vision` to import the vision
#
# If you wish to see where a symbol is imported from, either just type the symbol name (in a REPL such as Jupyter Notebook or IPython), or (in most editors) wave your mouse over the symbol to see the definition. For instance:

Learner


# ### Dependencies

# At the base of everything are the two modules [`core`](/core.html#core) and [`torch_core`](/torch_core.html#torch_core) (we're not including the `fastai.` prefix when naming modules in these docs). They define the basic functions we use in the library; [`core`](/core.html#core) only relies on general modules, whereas [`torch_core`](/torch_core.html#torch_core) requires pytorch. Most type-hinting shortcuts are defined there too (at least the one that don't depend on fastai classes defined later). Nearly all modules below import [`torch_core`](/torch_core.html#torch_core).
#
# Then, there are three modules directly on top of [`torch_core`](/torch_core.html#torch_core):
# - [`data`](/vision.data.html#vision.data), which contains the class that will take a [`Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) or pytorch [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) to wrap it in a [`DeviceDataLoader`](/basic_data.html#DeviceDataLoader) (a class that sits on top of a [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) and is in charge of putting the data on the right device as well as applying transforms such as normalization) and regroup then in a [`DataBunch`](/basic_data.html#DataBunch).
# - [`layers`](/layers.html#layers), which contains basic functions to define custom layers or groups of layers
# - [`metrics`](/metrics.html#metrics), which contains all the metrics
#
# This takes care of the basics, then we regroup a model with some data in a [`Learner`](/basic_train.html#Learner) object to take care of training. More specifically:
# - [`callback`](/callback.html#callback) (depends on [`data`](/vision.data.html#vision.data)) defines the basis of callbacks and the [`CallbackHandler`](/callback.html#CallbackHandler). Those are functions that will be called every step of the way of the training loop and can allow us to customize what is happening there;
# - [`basic_train`](/basic_train.html#basic_train) (depends on [`callback`](/callback.html#callback)) defines [`Learner`](/basic_train.html#Learner) and [`Recorder`](/basic_train.html#Recorder) (which is a callback that records training stats) and has the training loop;
# - [`callbacks`](/callbacks.html#callbacks) (depends on [`basic_train`](/basic_train.html#basic_train)) is a submodule defining various callbacks, such as for mixed precision training or 1cycle annealing;
# - `learn` (depends on [`callbacks`](/callbacks.html#callbacks)) defines helper functions to invoke the callbacks more easily.
#
# From [`data`](/vision.data.html#vision.data) we can split on one of the four main *applications*, which each has their own module: [`vision`](/vision.html#vision), [`text`](/text.html#text) [`collab`](/collab.html#collab), or [`tabular`](/tabular.html#tabular). Each of those submodules is built in the same way with:
# - a submodule named <code>transform</code> that handles the transformations of our data (data augmentation for computer vision, numericalizing and tokenizing for text and preprocessing for tabular)
# - a submodule named <code>data</code> that contains the class that will create datasets specific to this application and the helper functions to create [`DataBunch`](/basic_data.html#DataBunch) objects.
# - a submodule named <code>models</code> that contains the models specific to this application.
# - optionally, a submodule named <code>learn</code> that will contain [`Learner`](/basic_train.html#Learner) speficic to the application.
#
# Here is a graph of the key module dependencies:

# ![Modules overview](imgs/dependencies.svg)
