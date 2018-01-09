
# coding: utf-8

# ## Multi-label classification

get_ipython().magic(u'reload_ext autoreload')
get_ipython().magic(u'autoreload 2')
get_ipython().magic(u'matplotlib inline')


from fastai.conv_learner import *


PATH = 'data/planet/'


# Data preparation steps if you are using Crestle:

os.makedirs('data/planet/models', exist_ok=True)
os.makedirs('/cache/planet/tmp', exist_ok=True)

get_ipython().system(
    u'ln -s /datasets/kaggle/planet-understanding-the-amazon-from-space/train-jpg {PATH}')
get_ipython().system(
    u'ln -s /datasets/kaggle/planet-understanding-the-amazon-from-space/train_v2.csv {PATH}')
get_ipython().system(u'ln -s /cache/planet/tmp {PATH}')


ls {PATH}


# ## Multi-label versus single-label classification

from fastai.plots import *


def get_1st(path): return glob(f'{path}/*.*')[0]


dc_path = "data/dogscats/valid/"
list_paths = [get_1st(f"{dc_path}cats"), get_1st(f"{dc_path}dogs")]
plots_from_files(
    list_paths,
    titles=[
        "cat",
        "dog"],
    maintitle="Single-label classification")


# In single-label classification each sample belongs to one class. In the
# previous example, each image is either a *dog* or a *cat*.

list_paths = [f"{PATH}train-jpg/train_0.jpg", f"{PATH}train-jpg/train_1.jpg"]
titles = ["haze primary", "agriculture clear primary water"]
plots_from_files(
    list_paths,
    titles=titles,
    maintitle="Multi-label classification")


# In multi-label classification each sample can belong to one or more
# clases. In the previous example, the first images belongs to two clases:
# *haze* and *primary*. The second image belongs to four clases:
# *agriculture*, *clear*, *primary* and  *water*.

# ## Multi-label models for Planet dataset

from planet import f2

metrics = [f2]
f_model = resnet34


label_csv = f'{PATH}train_v2.csv'
n = len(list(open(label_csv))) - 1
val_idxs = get_cv_idxs(n)


# We use a different set of data augmentations for this dataset - we also
# allow vertical flips, since we don't expect vertical orientation of
# satellite images to change our classifications.

def get_data(sz):
    tfms = tfms_from_model(
        f_model,
        sz,
        aug_tfms=transforms_top_down,
        max_zoom=1.05)
    return ImageClassifierData.from_csv(PATH, 'train-jpg', label_csv, tfms=tfms,
                                        suffix='.jpg', val_idxs=val_idxs, test_name='test-jpg')


data = get_data(256)


x, y = next(iter(data.val_dl))


y


list(zip(data.classes, y[0]))


plt.imshow(data.val_ds.denorm(to_np(x))[0] * 1.4)


sz = 64


data = get_data(sz)


data = data.resize(int(sz * 1.3), 'tmp')


learn = ConvLearner.pretrained(f_model, data, metrics=metrics)


lrf = learn.lr_find()
learn.sched.plot()


lr = 0.2


learn.fit(lr, 3, cycle_len=1, cycle_mult=2)


lrs = np.array([lr / 9, lr / 3, lr])


learn.unfreeze()
learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)


learn.save(f'{sz}')


learn.sched.plot_loss()


sz = 128


learn.set_data(get_data(sz))
learn.freeze()
learn.fit(lr, 3, cycle_len=1, cycle_mult=2)


learn.unfreeze()
learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)
learn.save(f'{sz}')


sz = 256


learn.set_data(get_data(sz))
learn.freeze()
learn.fit(lr, 3, cycle_len=1, cycle_mult=2)


learn.unfreeze()
learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)
learn.save(f'{sz}')


log_preds, y = learn.TTA()
preds = np.mean(np.exp(log_preds), 0)


f2(preds, y)


# ### End
