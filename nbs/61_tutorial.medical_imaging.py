# -*- coding: utf-8 -*-
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
from nbdev.showdoc import *
import pandas as pd
import pydicom
from fastai.medical.imaging import *
from fastai.vision.all import *
from fastai.callback.all import *
from fastai.basics import *
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# +
# hide
# all_slow
# -

# To use `fastai.medical.imaging` you'll need to:
#
# ```bash
# conda install pyarrow
# pip install pydicom kornia opencv-python scikit-image
# ```

# To run this tutorial on Google Colab, you'll need to uncomment the following two lines and run the cell:

# +
# #!conda install pyarrow

# +


# -

# hide

# # Tutorial - Binary classification of chest X-rays
#
# > In this tutorial we will build a classifier that distinguishes between chest X-rays with pneumothorax and chest X-rays without pneumothorax. The image data is loaded directly from the DICOM source files, so no prior DICOM data handling is needed. This tutorial also goes through what DICOM images are and review at a high level how to evaluate the results of the classifier.

# ## Download and import of X-ray DICOM files

# First, we will use the `untar_data` function to download the _siim_small_ folder containing a subset (250 DICOM files, \~30MB) of the [SIIM-ACR Pneumothorax Segmentation](https://doi.org/10.1007/s10278-019-00299-9) \[1\] dataset.
# The downloaded _siim_small_ folder will be stored in your _\~/.fastai/data/_ directory. The variable `pneumothorax-source` will store the absolute path to the _siim_small_ folder as soon as the download is complete.

pneumothorax_source = untar_data(URLs.SIIM_SMALL)

# The _siim_small_ folder has the following directory/file structure:

# ![siim_folder_structure.jpg](images/siim_folder_structure.jpeg)

# ## What are DICOMs?

# **DICOM**(**D**igital **I**maging and **CO**mmunications in **M**edicine) is the de-facto standard that establishes rules that allow medical images(X-Ray, MRI, CT) and associated information to be exchanged between imaging equipment from different vendors, computers, and hospitals. The DICOM format provides a suitable means that meets health infomation exchange (HIE) standards for transmision of health related data among facilites and HL7 standards which is the messaging standard that enables clinical applications to exchange data
#
# DICOM files typically have a `.dcm` extension and provides a means of storing data in separate ‘tags’ such as patient information as well as image/pixel data. A DICOM file consists of a header and image data sets packed into a single file. By extracting data from these tags one can access important information regarding the patient demographics, study parameters, etc.
#
# 16 bit DICOM images have values ranging from `-32768` to `32768` while 8-bit greyscale images store values from `0` to `255`. The value ranges in DICOM images are useful as they correlate with the [Hounsfield Scale](https://en.wikipedia.org/wiki/Hounsfield_scale) which is a quantitative scale for describing radiodensity

# ### Plotting the DICOM data

# To analyze our dataset, we load the paths to the DICOM files with the `get_dicom_files` function. When calling the function, we append _train/_ to the `pneumothorax_source` path to choose the folder where the DICOM files are located. We store the path to each DICOM file in the `items` list.

items = get_dicom_files(pneumothorax_source / f"train/")

# Next, we split the `items` list into a train `trn` and validation `val` list using the `RandomSplitter` function:

trn, val = RandomSplitter()(items)

# Pydicom is a python package for parsing DICOM files, making it easier to access the `header` of the DICOM as well as coverting the raw `pixel_data` into pythonic structures for easier manipulation. `fastai.medical.imaging` uses `pydicom.dcmread` to load the DICOM file.
#
# To plot an X-ray, we can select an entry in the `items` list and load the DICOM file with `dcmread`.

patient = 7
xray_sample = items[patient].dcmread()

# To view the `header`

xray_sample

# Explanation of each element is beyond the scope of this tutorial but [this](http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.3.html#sect_C.7.6.3.1.4) site has some excellent information about each of the entries

# Some key pointers on the tag information above:
#
# - **Pixel Data** (7fe0 0010) - This is where the raw pixel data is stored. The order of pixels encoded for each image plane is left to right, top to bottom, i.e., the upper left pixel (labeled 1,1) is encoded first
# - **Photometric Interpretation** (0028, 0004) - also known as color space. In this case it is `MONOCHROME2` where pixel data is represented as a single monochrome image plane where low values=dark, high values=bright. If the colorspace was `MONOCHROME` then the low values=bright and high values=dark info.
# - **Samples per Pixel** (0028, 0002) - This should be 1 as this image is monochrome. This value would be 3 if the color space was RGB for example
# - **Bits Stored** (0028 0101) - Number of bits stored for each pixel sample. Typical 8 bit images have a pixel range between `0` and `255`
# - **Pixel Represenation**(0028 0103) - can either be unsigned(0) or signed(1)
# - **Lossy Image Compression** (0028 2110) - `00` image has not been subjected to lossy compression. `01` image has been subjected to lossy compression.
# - **Lossy Image Compression Method** (0028 2114) - states the type of lossy compression used (in this case `ISO_10918_1` represents JPEG Lossy Compression)
# - **Pixel Data** (7fe0, 0010) - Array of 161452 elements represents the image pixel data that pydicom uses to convert the pixel data into an image.

# What does `PixelData` look like?

xray_sample.PixelData[:200]

# Because of the complexity in interpreting `PixelData`, pydicom provides an easy way to get it in a convenient form: `pixel_array` which returns a `numpy.ndarray` containing the pixel data:

xray_sample.pixel_array, xray_sample.pixel_array.shape

# You can then use the `show` function to view the image

xray_sample.show()

# You can also conveniently create a dataframe with all the `tag` information as columns for all the images in a dataset by using `from_dicoms`

dicom_dataframe = pd.DataFrame.from_dicoms(items)
dicom_dataframe[:5]

# Next, we need to load the labels for the dataset. We import the _labels.csv_ file using pandas and print the first five entries. The **file** column shows the relative path to the _.dcm_ file and the **label** column indicates whether the chest x-ray has a pneumothorax or not.

df = pd.read_csv(pneumothorax_source / f"labels.csv")
df.head()

# Now, we use the `DataBlock` class to prepare the DICOM data for training.

# As we are dealing with DICOM images, we need to use `PILDicom` as the `ImageBlock` category.  This is so the `DataBlock` will know how to open the DICOM images.  As this is a binary classification task we will use `CategoryBlock`

# +
pneumothorax = DataBlock(blocks=(ImageBlock(cls=PILDicom), CategoryBlock),
                         get_x=lambda x: pneumothorax_source / f"{x[0]}",
                         get_y=lambda x: x[1],
                         batch_tfms=aug_transforms(size=224))

dls = pneumothorax.dataloaders(df.values, num_workers=0)
# -

# Additionally, we plot a first batch with the specified transformations:

dls = pneumothorax.dataloaders(df.values)
dls.show_batch(max_n=16)

# ## Training

# We can then use the `cnn_learner` function and initiate the training.

learn = cnn_learner(dls, resnet34, metrics=accuracy)

# Note that if you do not select a loss or optimizer function, fastai will try to choose the best selection for the task.  You can check the loss function by calling `loss_func`

learn.loss_func

# And you can do the same for the optimizer by calling `opt_func`

learn.opt_func

# Use `lr_find` to try to find the best learning rate

learn.lr_find()

learn.fit_one_cycle(1)

learn.predict(pneumothorax_source / f"train/Pneumothorax/000004.dcm")

# When predicting on an image `learn.predict` returns a tuple (class, class tensor and [probabilities of each class]).In this dataset there are only 2 classes `No Pneumothorax` and `Pneumothorax` hence the reason why each probability has 2 values, the first value is the probability whether the image belongs to `class 0` or `No Pneumothorax` and the second value is the probability whether the image belongs to `class 1` or `Pneumothorax`

tta = learn.tta(use_max=True)

learn.show_results(max_n=16)

interp = Interpretation.from_learner(learn)

interp.plot_top_losses(2)

# ## Result Evaluation

# Medical models are predominantly high impact so it is important to know how good a model is at detecting a certain condition.
#
# This model has an accuracy of 56%. Accuracy can be defined as the number of correctly predicted data points out of all the data points. However in this context we can define accuracy as the probability that the model is correct and the patient has the condition **PLUS** the probability that the model is correct and the patient does not have the condition

# There are some other key terms that need to be used when evaluating medical models:

# **False Positive & False Negative**

# - **False Positive** is an error in which a test result improperly indicates presence of a condition, such as a disease (the result is positive), when in reality it is not present
#
#
# - **False Negative** is an error in which a test result improperly indicates no presence of a condition (the result is negative), when in reality it is present

# **Sensitivity & Specificity**

# - **Sensitivity or True Positive Rate** is where the model classifies a patient has the disease given the patient actually does have the disease. Sensitivity quantifies the avoidance of false negatives
#
#
# Example: A new test was tested on 10,000 patients, if the new test has a sensitivity of 90% the test will correctly detect 9,000 (True Positive) patients but will miss 1000 (False Negative) patients that have the condition but were tested as not having the condition
#
# - **Specificity or True Negative Rate** is where the model classifies a patient as not having the disease given the patient actually does not have the disease. Specificity quantifies the avoidance of false positives

# [Understanding and using sensitivity, specificity and predictive values](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2636062/) is a great paper if you are interested in learning more about understanding sensitivity, specificity and predictive values.

# **PPV and NPV**

# Most medical testing is evaluated via **PPV** (Positive Predictive Value) or **NPV** (Negative Predictive Value).
#
# **PPV** - if the model predicts a patient has a condition what is the probability that the patient actually has the condition
#
# **NPV** - if the model predicts a patient does not have a condition what is the probability that the patient actually does not have the condition
#
# The ideal value of the PPV, with a perfect test, is 1 (100%), and the worst possible value would be zero
#
# The ideal value of the NPV, with a perfect test, is 1 (100%), and the worst possible value would be zero

# **Confusion Matrix**

# The confusion matrix is plotted against the `valid` dataset

interp = ClassificationInterpretation.from_learner(learn)
losses, idxs = interp.top_losses()
len(dls.valid_ds) == len(losses) == len(idxs)
interp.plot_confusion_matrix(figsize=(7, 7))

# You can also reproduce the results interpreted from plot_confusion_matrix like so:

upp, low = interp.confusion_matrix()
tn, fp = upp[0], upp[1]
fn, tp = low[0], low[1]
print(tn, fp, fn, tp)

# Note that **Sensitivity = True Positive/(True Positive + False Negative)**

sensitivity = tp / (tp + fn)
sensitivity

# In this case the model has a sensitivity of 40% and hence is only capable of correctly detecting 40% True Positives (i.e. who have Pneumothorax) but will miss 60% of False Negatives (patients that actually have Pneumothorax but were told they did not! Not a good situation to be in).
#
# This is also know as a **Type II error**

# **Specificity = True Negative/(False Positive + True Negative)**

specificity = tn / (fp + tn)
specificity

# The model has a specificity of 63% and hence can correctly detect 63% of the time that a patient does **not** have Pneumothorax but will incorrectly classify that 37% of the patients have Pneumothorax (False Postive) but actually do not.
#
# This is also known as a **Type I error**

# **Positive Predictive Value (PPV)**

ppv = tp / (tp + fp)
ppv

# In this case the model performs poorly in correctly predicting patients with Pneumothorax

# **Negative Predictive Value (NPV)**

npv = tn / (tn + fn)
npv

# This model is better at predicting patients with No Pneumothorax

# **Calculating Accuracy**

# The accuracy of this model as mentioned before was 56% but how was this calculated? We can consider accuracy as:

# **accuracy = sensitivity x prevalence + specificity * (1 - prevalence)**

# Where **prevalence** is a statistical concept referring to the number of cases of a disease that are present in a particular population at a given time. The prevalence in this case is how many patients in the valid dataset have the condition compared to the total number.
#
# To view the files in the valid dataset you call `dls.valid_ds.cat`

val = dls.valid_ds.cat
# val[0]

# There are 15 Pneumothorax images in the valid set (which has a total of 50 images and can be checked by using `len(dls.valid_ds)`) so the prevalence here is 15/50 = 0.3

prevalence = 15 / 50
prevalence

accuracy = (sensitivity * prevalence) + (specificity * (1 - prevalence))
accuracy

# _**Citations:**_
#
# \[1\] _Filice R et al. Crowdsourcing pneumothorax annotations using machine learning annotations on the NIH chest X-ray dataset.  J Digit Imaging (2019). https://doi.org/10.1007/s10278-019-00299-9_
