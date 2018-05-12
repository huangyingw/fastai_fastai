from fastai.conv_learner import ConvLearner
from fastai.dataset import ImageClassifierData
from fastai.transforms import tfms_from_model, transforms_side_on
from torchvision.models import resnet34
import matplotlib.pyplot as plt
import os
import os.path
import subprocess
import torch

os.chdir(os.path.dirname(os.path.realpath(__file__)))
PATH = "data/smallset/"
sz = 10
torch.cuda.is_available()
torch.backends.cudnn.enabled

command = "ls %svalid/cats | head" % (PATH)
files = subprocess.getoutput(command).split()

file_name = "%svalid/cats/%s" % (PATH, files[0])
img = plt.imread(file_name)
# plt.imshow(img)

# Here is how the raw data looks like
img.shape
img[:4, :4]

# Uncomment the below if you need to reset your precomputed activations
command = "rm -rf %stmp" % (PATH)
subprocess.getoutput(command)

arch = resnet34
data = ImageClassifierData.from_paths(PATH, tfms=tfms_from_model(arch, sz))


tfms = tfms_from_model(
    resnet34, sz, aug_tfms=transforms_side_on, max_zoom=1.1)

def learn1():
    data = ImageClassifierData.from_paths(PATH, tfms=tfms)
    learn = ConvLearner.pretrained(arch, data, precompute=True)
    learn.fit(1e-2, 1, saved_model_name='lesson1_1e-2')
    learn.sched.plot_lr()

learn1()
