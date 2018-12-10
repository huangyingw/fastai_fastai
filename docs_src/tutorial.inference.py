
# coding: utf-8

# # Create a Learner for inference

from fastai import *
from fastai.gen_doc.nbdoc import *
os.chdir(os.path.dirname(os.path.realpath(__file__)))


# In this tutorial, we'll see how the same API allows you to create an empty [`DataBunch`](/basic_data.html#DataBunch) for a [`Learner`](/basic_train.html#Learner) at inference time (once you have trained your model) and how to call the `predict` method to get the predictions on a single item.

jekyll_note("""As usual, this page is generated from a notebook that you can find in the <code>docs_src</code> folder of the
<a href="https://github.com/fastai/fastai">fastai repo</a>. We use the saved models from <a href="/tutorial.data.html">this tutorial</a> to
have this notebook run quickly.""")


# ## Vision

# To quickly get acces to all the vision functionality inside fastai, we use the usual import statements.

from fastai import *
from fastai.vision import *


# ### A classification problem

# Let's begin with our sample of the MNIST dataset.

mnist = untar_data(URLs.MNIST_TINY)
tfms = get_transforms(do_flip=False)


# It's set up with an imagenet structure so we use it to split our training and validation set, then labelling.

data = (ImageItemList.from_folder(mnist)
        .split_by_folder()
        .label_from_folder()
        .transform(tfms, size=32)
        .databunch()
        .normalize(imagenet_stats))


# Now that our data has been properly set up, we can train a model. Once the time comes to deploy it for inference, we'll need to save the information this [`DataBunch`](/basic_data.html#DataBunch) contains (classes for instance), to do this, we call `data.export()`. This will create an `export.pkl` file that you'll need to copy with your model file if you want to deploy it on another device.

data.export()


# To create the [`DataBunch`](/basic_data.html#DataBunch) for inference, you'll need to use the `load_empty` method. Note that for now, transforms and normalization aren't saved inside the export file. This is going to be integrated in a future version of the library. For now, we pass the transforms we applied on the validation set, along with all relevant `kwargs`, and we normalize with the same statistics as during training.

sd = LabelLists.load_empty(mnist / 'export.pkl', tfms=tfms, size=32)
empty_data = sd.databunch().normalize(imagenet_stats)


# Then, we use it to create a [`Learner`](/basic_train.html#Learner) and load the model we trained before.

learn = create_cnn(empty_data, models.resnet18).load('mini_train')


# You can now get the predictions on any image via `learn.predict`.

img = data.train_ds[0][0]
learn.predict(img)


# It returns a tuple of three things: the object predicted (with the class in this instance), the underlying data (here the corresponding index) and the raw probabilities. You can also do inference on a larger set of data by adding a *test set*. Simply use the exact same steps as before, but add a test set to your [`LabelLists`](/data_block.html#LabelLists):

sd = LabelLists.load_empty(mnist / 'export.pkl', tfms=tfms, size=32).add_test_folder('test')
empty_data = sd.databunch().normalize(imagenet_stats)


# Now you can use [`Learner.get_preds`](/basic_train.html#Learner.get_preds) in the usual way.

learn = create_cnn(empty_data, models.resnet18).load('mini_train')


preds, y = learn.get_preds(ds_type=DatasetType.Test)
preds[:5]


# ### A multi-label problem

# Now let's try these on the planet dataset, which is a little bit different in the sense that each image can have multiple tags (and not just one label).

planet = untar_data(URLs.PLANET_TINY)
planet_tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)


# Here each images is labelled in a file named `labels.csv`. We have to add [`train`](/train.html#train) as a prefix to the filenames, `.jpg` as a suffix and indicate that the labels are separated by spaces.

data = (ImageItemList.from_csv(planet, 'labels.csv', folder='train', suffix='.jpg')
        .random_split_by_pct()
        .label_from_df(sep=' ')
        .transform(planet_tfms, size=128)
        .databunch()
        .normalize(imagenet_stats))


# Again, we call `data.export()` to export our data object properties.

data.export()


# We can then create the [`DataBunch`](/basic_data.html#DataBunch) for inference, by using the `load_empty` method as before.

empty_data = ImageDataBunch.load_empty(planet, tfms=tfms, size=32).normalize(imagenet_stats)
learn = create_cnn(empty_data, models.resnet18)
learn.load('mini_train');


# And we get the predictions on any image via `learn.predict`.

img = data.train_ds[0][0]
learn.predict(img)


# Here we can specify a particular threshold to consider the predictions to be correct or not. The default is `0.5`, but we can change it.

learn.predict(img, thresh=0.3)


# ### A regression example

# For the next example, we are going to use the [BIWI head pose](https://data.vision.ee.ethz.ch/cvl/gfanelli/head_pose/head_forest.html#db) dataset. On pictures of persons, we have to find the center of their face. For the fastai docs, we have built a small subsample of the dataset (200 images) and prepared a dictionary for the correspondance fielname to center.

biwi = untar_data(URLs.BIWI_SAMPLE)
fn2ctr = pickle.load(open(biwi / 'centers.pkl', 'rb'))


# To grab our data, we use this dictionary to label our items. We also use the [`PointsItemList`](/vision.data.html#PointsItemList) class to have the targets be of type [`ImagePoints`](/vision.image.html#ImagePoints) (which will make sure the data augmentation is properly applied to them). When calling [`transform`](/tabular.transform.html#tabular.transform) we make sure to set `tfm_y=True`.

data = (ImageItemList.from_folder(biwi)
        .random_split_by_pct(seed=42)
        .label_from_func(lambda o: fn2ctr[o.name], label_cls=PointsItemList)
        .transform(get_transforms(), tfm_y=True, size=(120, 160))
        .databunch()
        .normalize(imagenet_stats))


# As before, the road to inference is pretty straightforward: export the data, then load an empty [`DataBunch`](/basic_data.html#DataBunch).

data.export()


empty_data = ImageDataBunch.load_empty(biwi, tfms=get_transforms(), tfm_y=True, size=(120, 60)).normalize(imagenet_stats)
learn = create_cnn(empty_data, models.resnet18, lin_ftrs=[100], ps=0.05)
learn.load('mini_train');


# And now we can a prediction on an image.

img = data.valid_ds[0][0]
learn.predict(img)


# To visualize the predictions, we can use the [`Image.show`](/vision.image.html#Image.show) method.

img.show(y=learn.predict(img)[0])


# ### A segmentation example

# Now we are going to look at the [camvid dataset](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) (at least a small sample of it), where we have to predict the class of each pixel in an image. Each image in the 'images' subfolder as an equivalent in 'labels' that is its segmentations mask.

camvid = untar_data(URLs.CAMVID_TINY)
path_lbl = camvid / 'labels'
path_img = camvid / 'images'


# We read the classes in 'codes.txt' and the function maps each image filename with its corresponding mask filename.

codes = np.loadtxt(camvid / 'codes.txt', dtype=str)
get_y_fn = lambda x: path_lbl / f'{x.stem}_P{x.suffix}'


# The data block API allows us to uickly get everything in a [`DataBunch`](/basic_data.html#DataBunch) and then we can have a look with `show_batch`.

data = (SegmentationItemList.from_folder(path_img)
        .random_split_by_pct()
        .label_from_func(get_y_fn, classes=codes)
        .transform(get_transforms(), tfm_y=True, size=128)
        .databunch(bs=16, path=camvid)
        .normalize(imagenet_stats))


# As before, we export the data then create an empty [`DataBunch`](/basic_data.html#DataBunch) that we pass to a [`Learner`](/basic_train.html#Learner).

data.export()


empty_data = ImageDataBunch.load_empty(camvid, tfms=get_transforms(), tfm_y=True, size=128).normalize(imagenet_stats)
learn = unet_learner(empty_data, models.resnet18)
learn.load('mini_train');


# And now we can a prediction on an image.

img = data.train_ds[0][0]
learn.predict(img);


# To visualize the predictions, we can use the [`Image.show`](/vision.image.html#Image.show) method.

img.show(y=learn.predict(img)[0])


# ## Text

# Next application is text, so let's start by importing everything we'll need.

from fastai import *
from fastai.text import *


# ### Language modelling

# First let's look a how to get a language model ready for inference. Since we'll load the model trained in the [visualize data tutorial](/tutorial.data.html), we load the vocabulary used there.

imdb = untar_data(URLs.IMDB_SAMPLE)


vocab = Vocab(pickle.load(open(imdb / 'tmp' / 'itos.pkl', 'rb')))
data_lm = (TextList.from_csv(imdb, 'texts.csv', cols='text', vocab=vocab)
                   .random_split_by_pct()
                   .label_for_lm()
                   .databunch())


# Like in vision, we just have to type `data_lm.export()` to save all the information inside the [`DataBunch`](/basic_data.html#DataBunch) we'll need. In this case, this includes all the vocabulary we created.

data_lm.export()


# Now let's define a language model learner from an empty data object.

empty_data = TextLMDataBunch.load_empty(imdb)
learn = language_model_learner(empty_data)
learn.unfreeze()
learn.load('mini_train_lm', with_opt=False);


# Then we can predict with the usual method, here we can specify how many words we want the model to predict.

learn.predict('This is a simple test of', n_words=20)


# ### Classification

# Now let's see a classification example. We have to use the same vocabulary as for the language model if we want to be able to use the encoder we saved.

data_clas = (TextList.from_csv(imdb, 'texts.csv', cols='text', vocab=vocab)
                   .split_from_df(col='is_valid')
                   .label_from_df(cols='label')
                   .databunch(bs=42))


# Again we export the data.

data_clas.export()


# Now let's define a text classifier from an empty data object.

empty_data = TextClasDataBunch.load_empty(imdb)
learn = text_classifier_learner(empty_data)
learn.load('mini_train_clas', with_opt=False);


# Then we can predict with the usual method.

learn.predict('I really loved that movie!')


# ## Tabular

# Last application brings us to tabular data. First let's import everything we'll need.

from fastai import *
from fastai.tabular import *


# We'll use a sample of the [adult dataset](https://archive.ics.uci.edu/ml/datasets/adult) here. Once we read the csv file, we'll need to specify the dependant variable, the categorical variables, the continuous variables and the processors we want to use.

adult = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(adult / 'adult.csv')
dep_var = '>=50k'
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
cont_names = ['education-num', 'hours-per-week', 'age', 'capital-loss', 'fnlwgt', 'capital-gain']
procs = [FillMissing, Categorify, Normalize]


# Then we can use the data block API to grab everything together before using `data.show_batch()`

data = (TabularList.from_df(df, path=adult, cat_names=cat_names, cont_names=cont_names, procs=procs)
                           .split_by_idx(valid_idx=range(800, 1000))
                           .label_from_df(cols=dep_var)
                           .databunch())


# We define a [`Learner`](/basic_train.html#Learner) object that we fit and then save the model.

learn = tabular_learner(data, layers=[200, 100], metrics=accuracy)
learn.fit(1, 1e-2, saved_model_name='tutorial.inference_1')
learn.save('mini_train')


# As in the other applications, we just have to type `data.export()` to save everything we'll need for inference (here the inner state of each processor).

data.export()


# Then we create an empty data object and a learner from it like before.

data = TabularDataBunch.load_empty(adult)
learn = tabular_learner(data, layers=[200, 100])
learn.load('mini_train');


# And we can predict on a row of dataframe that has the right `cat_names` and `cont_names`.

learn.predict(df.iloc[0])
