# coding: utf-8
# # Viewing inputs and outputs
from fastai.basics import *
from fastai.gen_doc.nbdoc import *
# In this tutorial, we'll see how the same API allows you to get a look at the inputs and outputs of your model, whether in the vision, text or tabular application. We'll go over a lot of different tasks and each time, grab some data in a [`DataBunch`](/basic_data.html#DataBunch) with the [data block API](/data_block.html), see how to get a look at a few inputs with the `show_batch` method, train an appropriate [`Learner`](/basic_train.html#Learner) then use the `show_results` method to see what the outputs of our model actually look like.
jekyll_note("""As usual, this page is generated from a notebook that you can find in the docs_srs folder of the
[fastai repo](https://github.com/fastai/fastai). The examples are all designed to run fast, which is why we use
samples of the dataset, a resnet18 as a backbone and don't train for very long. You can change all of those parameters
to run your own experiments!
""")
# ## Vision
# To quickly get access to all the vision functions inside fastai, we use the usual import statements.
from fastai.vision import *
# ### A classification problem
# Let's begin with our sample of the MNIST dataset.
mnist = untar_data(URLs.MNIST_TINY)
tfms = get_transforms(do_flip=False)
# It's set up with an imagenet structure so we use it to load our training and validation datasets, then label, transform, convert them into ImageDataBunch and finally, normalize them.
data = (ImageItemList.from_folder(mnist)
        .split_by_folder()
        .label_from_folder()
        .transform(tfms, size=32)
        .databunch()
        .normalize(imagenet_stats))
# Once your data is properly set up in a [`DataBunch`](/basic_data.html#DataBunch), we can call `data.show_batch()` to see what a sample of a batch looks like.
data.show_batch()
# Note that the images were automatically de-normalized before being showed with their labels (inferred from the names of the folder). We can specify a number of rows if the default of 5 is too big, and we can also limit the size of the figure.
data.show_batch(rows=3, figsize=(4, 4))
# Now let's create a [`Learner`](/basic_train.html#Learner) object to train a classifier.
learn = create_cnn(data, models.resnet18, metrics=accuracy)
learn.fit_one_cycle(1, 1e-2)
learn.save('mini_train')
# Our model has quickly reache around 89% accuracy, now let's see its predictions on a sample of the validation set. For this, we use the `show_results` method.
learn.show_results()
# Since the validation set is usually sorted, we get only images belonging to the same class. We can then again specify a number of rows, a figure size, but also the dataset on which we want to make predictions.
learn.show_results(ds_type=DatasetType.Train, rows=4, figsize=(8, 10))
# ### A multilabel problem
# Now let's try these on the planet dataset, which is a little bit different in the sense that each image can have multiple tags (and not jsut one label).
planet = untar_data(URLs.PLANET_TINY)
planet_tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
# Here each images is labelled in a file named 'labels.csv'. We have to add 'train' as a prefix to the filenames, '.jpg' as a suffix and he labels are separated by spaces.
data = (ImageItemList.from_csv(planet, 'labels.csv', folder='train', suffix='.jpg')
        .random_split_by_pct()
        .label_from_df(sep=' ')
        .transform(planet_tfms, size=128)
        .databunch()
        .normalize(imagenet_stats))
# And we can have look at our data with `data.show_batch`.
data.show_batch(rows=2, figsize=(9, 7))
# Then we can then create a [`Learner`](/basic_train.html#Learner) object pretty easily and train it for a little bit.
learn = create_cnn(data, models.resnet18)
learn.fit_one_cycle(5, 1e-2)
learn.save('mini_train')
# And to see actual predictions, we just have to run `learn.show_results()`.
learn.show_results(rows=3, figsize=(12, 15))
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
data = (PointsItemList.from_folder(biwi)
        .random_split_by_pct(seed=42)
        .label_from_func(lambda o: fn2ctr[o.name])
        .transform(get_transforms(), tfm_y=True, size=(120, 160))
        .databunch()
        .normalize(imagenet_stats))
# Then we can have a first look at our data with `data.show_batch()`.
data.show_batch(rows=3, figsize=(9, 6))
# We train our model for a little bit before using `learn.show_results()`.
learn = create_cnn(data, models.resnet18, lin_ftrs=[100], ps=0.05)
learn.fit_one_cycle(5, 5e-2)
learn.save('mini_train')
learn.show_results(rows=3)
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
data.show_batch(rows=2, figsize=(7, 5))
# Then we train a Unet for a few epochs.
jekyll_warn("This training is fairly unstable, you should use more epochs and the full dataset to get better results.")
learn = unet_learner(data, models.resnet18)
learn.fit_one_cycle(3, 1e-2)
learn.save('mini_train')
learn.show_results()
# ## Text
# Next application is text, so let's start by importing everything we'll need.
from fastai.text import *
# ### Language modelling
# First we'll fine-tune a pretrained language model on our subset of imdb.
imdb = untar_data(URLs.IMDB_SAMPLE)
data_lm = (TextList.from_csv(imdb, 'texts.csv', cols='text')
                   .random_split_by_pct()
                   .label_for_lm()
                   .databunch())
data_lm.save()
# `data.show_batch()` will work here as well. For a language model, it shows us the beginning of each sequence of text along the batch dimension (the target being to guess the next word).
data_lm.show_batch()
# Now let's define a language model learner
learn = language_model_learner(data_lm, pretrained_model=URLs.WT103_1)
learn.fit_one_cycle(2, 1e-2)
learn.save('mini_train_lm')
learn.save_encoder('mini_train_encoder')
# Then we can have a look at the results. It shows a certain amount of words (default 20), then the next 20 target words and the ones that were predicted.
learn.show_results()
# ### Classification
# Now let's see a classification example. We have to use the same vocabulary as for the language model if we want to be able to use the encoder we saved.
data_clas = (TextList.from_csv(imdb, 'texts.csv', cols='text', vocab=data_lm.vocab)
                   .split_from_df(col='is_valid')
                   .label_from_df(cols='label')
                   .databunch(bs=42))
# Here show_batch shows the beginning of each review with its target.
data_clas.show_batch()
# And we can train a classifier that uses our previous encoder.
learn = text_classifier_learner(data_clas)
learn.load_encoder('mini_train_encoder')
learn.fit_one_cycle(2, slice(1e-3, 1e-2))
learn.save('mini_train_clas')
learn.show_results()
# ## Tabular
# Last application brings us to tabular data. First let's import everything we'll need.
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
data.show_batch()
# Here we grab a [`tabular_learner`](/tabular.data.html#tabular_learner) that we train for a little bit.
learn = tabular_learner(data, layers=[200, 100], metrics=accuracy)
learn.fit(1, 1e-2)
learn.save('mini_train')
# And we can use `learn.show_results()`.
learn.show_results()
