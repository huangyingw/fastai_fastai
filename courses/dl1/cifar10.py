
# coding: utf-8

# ## CIFAR 10

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


from fastai.conv_learner import *
PATH = "data/cifar10/"
os.makedirs(PATH, exist_ok=True)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
stats = (np.array([0.4914, 0.48216, 0.44653]), np.array([0.24703, 0.24349, 0.26159]))


def get_data(sz, bs):
    tfms = tfms_from_stats(stats, sz, aug_tfms=[RandomFlip()], pad=sz // 8)
    return ImageClassifierData.from_paths(PATH, val_name='test', tfms=tfms, bs=bs)


bs = 128


# ### Look at data

data = get_data(32, 4)


x, y = next(iter(data.trn_dl))


plt.imshow(data.trn_ds.denorm(x)[0]);


plt.imshow(data.trn_ds.denorm(x)[1]);


# ## Initial model

from fastai.models.cifar10.resnext import resnext29_8_64

m = resnext29_8_64()
bm = BasicModel(m.cuda(), name='cifar10_rn29_8_64')


data = get_data(8, bs * 4)


learn = ConvLearner(data, bm)
learn.unfreeze()


lr = 1e-2; wd = 5e-4


learn.lr_find()


learn.sched.plot()


get_ipython().run_line_magic('time', 'learn.fit(lr, 1)')


learn.fit(lr, 2, cycle_len=1)


learn.fit(lr, 3, cycle_len=1, cycle_mult=2, wds=wd)


learn.save('8x8_8')


# ## 16x16

learn.load('8x8_8')


learn.set_data(get_data(16, bs * 2))


get_ipython().run_line_magic('time', 'learn.fit(1e-3, 1, wds=wd)')


learn.unfreeze()


learn.lr_find()


learn.sched.plot()


lr = 1e-2


learn.fit(lr, 2, cycle_len=1, wds=wd)


learn.fit(lr, 3, cycle_len=1, cycle_mult=2, wds=wd)


learn.save('16x16_8')


# ## 24x24

learn.load('16x16_8')


learn.set_data(get_data(24, bs))


learn.fit(1e-2, 1, wds=wd)


learn.unfreeze()


learn.fit(lr, 1, cycle_len=1, wds=wd)


learn.fit(lr, 3, cycle_len=1, cycle_mult=2, wds=wd)


learn.save('24x24_8')


log_preds, y = learn.TTA()
preds = np.mean(np.exp(log_preds), 0), metrics.log_loss(y, preds), accuracy_np(preds, y)


# ## 32x32

learn.load('24x24_8')


learn.set_data(get_data(32, bs))


learn.fit(1e-2, 1, wds=wd)


learn.unfreeze()


learn.fit(lr, 3, cycle_len=1, cycle_mult=2, wds=wd)


learn.fit(lr, 3, cycle_len=4, wds=wd)


log_preds, y = learn.TTA()
metrics.log_loss(y, np.exp(log_preds)), accuracy_np(log_preds, y)


learn.save('32x32_8')
