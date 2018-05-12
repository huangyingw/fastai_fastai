from PIL import Image
from fastai.conv_learner import ConvLearner
from fastai.dataset import ImageClassifierData
from fastai.transforms import tfms_from_model
from torchvision.models import resnet34
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path
import pprint
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


def plots(ims, figsize=(12, 6), rows=1, titles=None):
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims) // rows, i + 1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
            plt.imshow(ims[i])
    plt.show()


def plot_val_with_title(idxs, title, probs):
    imgs = np.stack([data.val_ds[x][0] for x in idxs])
    title_probs = [probs[x] for x in idxs]
    print(title)
    return plots(data.val_ds.denorm(imgs), rows=1, titles=title_probs)


def most_by_mask(mask, mult, probs):
    idxs = np.where(mask)[0]
    return idxs[np.argsort(mult * probs[idxs])[:4]]


def most_by_correct(y, is_correct, preds, probs):
    mult = -1 if (y == 1) == is_correct else 1
    return most_by_mask(((preds == data.val_y) == is_correct)
                        & (data.val_y == y), mult, probs)


def learn1():

    learn = ConvLearner.pretrained(arch, data, precompute=True)
    learn.lr_find()
    learn.sched.plot_lr()
    learn.sched.plot()

    pp = pprint.PrettyPrinter(indent=4)
    print('learn --> ')
    pp.pprint(learn)

    print('data.val_y --> ', data.val_y)

    print('data.classes --> ', data.classes)

    log_preds = learn.predict()
    print('log_preds.shape --> ', log_preds.shape)
    log_preds.shape

    print('log_preds[:10] --> ', log_preds[:10])

    preds = np.argmax(log_preds, axis=1)  # from log probabilities to 0 or 1
    probs = np.exp(log_preds[:, 1])        # pr(dog)


learn1()
