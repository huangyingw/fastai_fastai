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

# + [markdown] hide_input=false
# # Creating your own dataset from Google Images
#
# *by: Francisco Ingham and Jeremy Howard. Inspired by [Adrian Rosebrock](https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/)*

# + [markdown] hide_input=true
# In this tutorial we will see how to easily create an image dataset through Google Images. **Note**: You will have to repeat these steps for any new category you want to Google (e.g once for dogs and once for cats).

# + hide_input=false
from fastai.callback.all import *
from fastai.basics import *
from fastai.vision.all import *
from nbdev.showdoc import *
# -

# ## Get a list of URLs

# ### Search and scroll

# Go to [Google Images](http://images.google.com) and search for the images you are interested in. The more specific you are in your Google Search, the better the results and the less manual pruning you will have to do.
#
# Scroll down until you've seen all the images you want to download, or until you see a button that says 'Show more results'. All the images you scrolled past are now available to download. To get more, click on the button, and continue scrolling. The maximum number of images Google Images shows is 700.
#
# It is a good idea to put things you want to exclude into the search query, for instance if you are searching for the Eurasian wolf, "canis lupus lupus", it might be a good idea to exclude other variants:
#
#     "canis lupus lupus" -dog -arctos -familiaris -baileyi -occidentalis
#
# You can also limit your results to show only photos by clicking on Tools and selecting Photos from the Type dropdown.

# ### Download into file

# Now you must run some Javascript code in your browser which will save the URLs of all the images you want for you dataset.
#
# Press <kbd>Ctrl</kbd><kbd>Shift</kbd><kbd>J</kbd> in Windows/Linux and <kbd>Cmd</kbd><kbd>Opt</kbd><kbd>J</kbd> in Mac, and a small window the javascript 'Console' will appear. That is where you will paste the JavaScript commands.
#
# You will need to get the urls of each of the images. Before running the following commands, you may want to disable ad blocking extensions (uBlock, AdBlockPlus etc.) in Chrome. Otherwise the window.open() command doesn't work. Then you can run the following commands:
#
# ```javascript
# urls = Array.from(document.querySelectorAll('.rg_di .rg_meta')).map(el=>JSON.parse(el.textContent).ou);
# window.open('data:text/csv;charset=utf-8,' + escape(urls.join('\n')));
# ```

# ### Create directory and upload urls file into your server

# Choose an appropriate name for your labeled images. You can run these steps multiple times to create different labels.

path = Config().data / 'bears'
path.mkdir(parents=True, exist_ok=True)

path.ls()

# Finally, upload your urls file. You just need to press 'Upload' in your working directory and select your file, then click 'Upload' for each of the displayed files.

# ## Download images

# Now you will need to download your images from their respective urls.
#
# fast.ai has a function that allows you to do just that. You just have to specify the urls filename as well as the destination folder and this function will download and save all images that can be opened. If they have some problem in being opened, they will not be saved.
#
# Let's download our images! Notice you can choose a maximum number of images to be downloaded. In this case we will not download all the urls.

classes = ['teddy', 'grizzly', 'black']

for c in classes:
    print(c)
    file = f'urls_{c}.csv'
    download_images(path / c, path / file, max_pics=200)

# +
# If you have problems download, try with `max_workers=0` to see exceptions:
#download_images(path/file, dest, max_pics=20, max_workers=0)
# -

# Then we can remove any images that can't be opened:

for c in classes:
    print(c)
    verify_images(path / c, delete=True, max_size=500)

# ## View data

np.random.seed(42)
dls = ImageDataLoaders.from_folder(path, train=".", valid_pct=0.2, item_tfms=RandomResizedCrop(460, min_scale=0.75),
                                   bs=64, batch_tfms=[*aug_transforms(size=224, max_warp=0), Normalize.from_stats(*imagenet_stats)])

# +
# If you already cleaned your data, run this cell instead of the one before
# np.random.seed(42)
# dls = ImageDataLoaders.from_csv(path, folder=".", valid_pct=0.2, csv_labels='cleaned.csv',
#         item_tfms=RandomResizedCrop(460, min_scale=0.75), bs=64,
#         batch_tfms=[*aug_transforms(size=224, max_warp=0), Normalize.from_stats(*imagenet_stats)])
# -

# Good! Let's take a look at some of our pictures then.

dls.vocab

dls.show_batch(rows=3, figsize=(7, 8))

dls.vocab, dls.c, len(dls.train_ds), len(dls.valid_ds)

# ## Train model

learn = cnn_learner(dls, resnet34, metrics=error_rate)

learn.fit_one_cycle(4)

learn.save('stage-1')

learn.unfreeze()

# If the plot is not showing try to give a start and end learning rate:
#
# `learn.lr_find(start_lr=1e-5, end_lr=1e-1)`

learn.lr_find()

learn.load('stage-1')

learn.fit_one_cycle(2, lr_max=slice(3e-5, 3e-4))

learn.save('stage-2')

# ## Interpretation

learn.load('stage-2')

interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()

# ## Putting your model in production

# First thing first, let's export the content of our `Learner` object for production:

learn.export()

# This will create a file named 'export.pkl' in the directory where we were working that contains everything we need to deploy our model (the model, the weights but also some metadata like the classes or the transforms/normalization used).

# You probably want to use CPU for inference, except at massive scale (and you almost certainly don't need to train in real-time). If you don't have a GPU that happens automatically. You can test your model on CPU like so:

defaults.device = torch.device('cpu')

img = Image.open(path / 'black' / '00000021.jpg')
img

# We create our `Learner` in production environment like this, just make sure that `path` contains the file 'export.pkl' from before.

learn = torch.load(path / 'export.pkl')

pred_class, pred_idx, outputs = learn.predict(path / 'black' / '00000021.jpg')
pred_class

# So you might create a route something like this ([thanks](https://github.com/simonw/cougar-or-not) to Simon Willison for the structure of this code):
#
# ```python
# @app.route("/classify-url", methods=["GET"])
# async def classify_url(request):
#     bytes = await get_bytes(request.query_params["url"])
#     img = PILImage.create(bytes)
#     _,_,probs = learner.predict(img)
#     return JSONResponse({
#         "predictions": sorted(
#             zip(cat_learner.dls.vocab, map(float, probs)),
#             key=lambda p: p[1],
#             reverse=True
#         )
#     })
# ```
#
# (This example is for the [Starlette](https://www.starlette.io/) web app toolkit.)

# ## Things that can go wrong

# - Most of the time things will train fine with the defaults
# - There's not much you really need to tune (despite what you've heard!)
# - Most likely are
#   - Learning rate
#   - Number of epochs

# ### Learning rate (LR) too high

learn = cnn_learner(dls, resnet34, metrics=error_rate)

learn.fit_one_cycle(1, lr_max=0.5)

# ### Learning rate (LR) too low

learn = cnn_learner(dls, resnet34, metrics=error_rate)

# Previously we had this result:
#
# ```
# Total time: 00:57
# epoch  train_loss  valid_loss  error_rate
# 1      1.030236    0.179226    0.028369    (00:14)
# 2      0.561508    0.055464    0.014184    (00:13)
# 3      0.396103    0.053801    0.014184    (00:13)
# 4      0.316883    0.050197    0.021277    (00:15)
# ```

learn.fit_one_cycle(5, lr_max=1e-5)

learn.recorder.plot_loss()

# As well as taking a really long time, it's getting too many looks at each image, so may overfit.

# ### Too few epochs

learn = cnn_learner(dls, resnet34, metrics=error_rate, pretrained=False)

learn.fit_one_cycle(1)

# ### Too many epochs


path = Config().data / 'bears'

np.random.seed(42)
dls = ImageDataLoaders.from_folder(path, train=".", valid_pct=0.8, item_tfms=RandomResizedCrop(460, min_scale=0.75),
                                   bs=32, batch_tfms=[AffineCoordTfm(size=224), Normalize.from_stats(*imagenet_stats)])

learn = cnn_learner(dls, resnet50, metrics=error_rate, config=cnn_config(ps=0))
learn.unfreeze()

learn.fit_one_cycle(40, slice(1e-6, 1e-4), wd=0)
