from PIL import Image
from fastai.conv_learner import ConvLearner
from fastai.dataset import ImageClassifierData
from fastai.metrics import accuracy_np
from fastai.plots import plot_confusion_matrix
from fastai.torch_imports import resnext50
from fastai.transforms import tfms_from_model, transforms_side_on
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path

os.chdir(os.path.dirname(os.path.realpath(__file__)))
PATH = "data/dogscats/"
sz = 299
arch = resnext50
bs = 28

tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
data = ImageClassifierData.from_paths(PATH, tfms=tfms, bs=bs, num_workers=4)
learn = ConvLearner.pretrained(arch, data, precompute=True, ps=0.5)

learn.fit(1e-2, 1, saved_model_name='lesson1-rxt50_1')

learn.precompute = False
learn.fit(1e-2, 2, saved_model_name='lesson1-rxt50_2', cycle_len=1)

learn.unfreeze()
lr = np.array([1e-4, 1e-3, 1e-2])
learn.fit(lr, 3, saved_model_name='224_all_50', cycle_len=1)

log_preds, y = learn.TTA()
probs = np.mean(np.exp(log_preds), 0)
accuracy_np(probs, y)

# ## Analyzing results
preds = np.argmax(probs, axis=1)
probs = probs[:, 1]
cm = confusion_matrix(y, preds)
plot_confusion_matrix(cm, data.classes)


def rand_by_mask(mask): return np.random.choice(np.where(mask)[0], 4, replace=False)


def rand_by_correct(is_correct): return rand_by_mask((preds == data.val_y) == is_correct)


def plot_val_with_title(idxs, title):
    imgs = np.stack([data.val_ds[x][0] for x in idxs])
    title_probs = [probs[x] for x in idxs]
    print(title)
    return plots(data.val_ds.denorm(imgs), rows=1, titles=title_probs)


def plots(ims, figsize=(12, 6), rows=1, titles=None):
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims) // rows, i + 1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i])
    plt.show()


def load_img_id(ds, idx): return np.array(Image.open(PATH + ds.fnames[idx]))
# def plot_val_with_title(idxs, title):
#    imgs = [load_img_id(data.val_ds, x) for x in idxs]
#    title_probs = [probs[x] for x in idxs]
#    print(title)
#    return plots(imgs, rows=1, titles=title_probs, figsize=(16, 8))


def most_by_mask(mask, mult):
    idxs = np.where(mask)[0]
    return idxs[np.argsort(mult * probs[idxs])[:4]]


def most_by_correct(y, is_correct):
    mult = -1 if (y == 1) == is_correct else 1
    return most_by_mask((preds == data.val_y) == is_correct & (data.val_y == y), mult)


plot_val_with_title(most_by_correct(0, False), "Most incorrect cats")
plot_val_with_title(most_by_correct(1, False), "Most incorrect dogs")
