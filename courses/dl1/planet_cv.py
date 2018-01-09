
# coding: utf-8

# - leaky relu / elu

# ## Planet Kaggle competition

get_ipython().magic(u'reload_ext autoreload')
get_ipython().magic(u'autoreload 2')
get_ipython().magic(u'matplotlib inline')


from fast_gen import *
from learner import *
from pt_models import *
from dataset_pt import *
from sgdr_pt import *

from planet import *

bs = 64
f_model = resnet34
path = "/data/jhoward/fast/planet/"
torch.cuda.set_device(1)


n = len(list(open(f'{path}train_v2.csv'))) - 1


data = get_data_pad(f_model, path, 256, 64, n, 0)


learn = Learner.pretrained_convnet(f_model, data, metrics=[f2])


# ### Train

learn.fit(0.2, 1, cycle_len=1)


learn.sched.plot_lr()


learn.unfreeze()


learn.fit([0.01, 0.05, 0.2], 12, cycle_len=4)


learn.fit([1e-4, 1e-3, 0.01], 4)


# ### Evaluate

name = '170809'


def load_cycle_cv(cv, cycle):
    data = get_data_zoom(f_model, path, 256, 64, n, cv)
    learn.set_data(data)
    learn.load_cycle(f'{name}_{cv}', cycle)
    return data


data = load_cycle_cv(0, 1)


val = learn.predict()


f2(val, data.val_y)


f2(learn.TTA(), data.val_y)


f2(val, data.val_y)


f2(learn.TTA(), data.val_y)


def get_labels(a): return [data.classes[o] for o in a.nonzero()[0]]


lbls = test > 0.2
idx = 9
print(get_labels(lbls[idx]))
PIL.Image.open(path + data.test_dl.dataset.fnames[idx]).convert('RGB')


res = [get_labels(o) for o in lbls]
data.test_dl.dataset.fnames[:5]


outp = pd.DataFrame({'image_name': [f[9:-4] for f in data.test_dl.dataset.fnames],
                     'tags': [' '.join(l) for l in res]})
outp.head()


outp.to_csv('tmp/subm.gz', compression='gzip', index=None)


from IPython.display import FileLink


FileLink('tmp/subm.gz')


def cycle_preds(name, cycle, n_tta=4, is_test=False):
    learn.load_cycle(name, cycle)
    return learn.TTA(n_tta, is_test=is_test)


def cycle_cv_preds(cv, n_tta=4, is_test=False):
    data = get_data_pad(f_model, path, 256, 64, n, cv)
    learn.set_data(data)
    return [cycle_preds(f'{name}_{cv}', i, is_test=is_test) for i in range(5)]


# - check dogs and cats
# - get resize working again with new path structure

get_ipython().run_cell_magic(
    u'time',
    u'',
    u'preds_arr=[]\nfor i in range(5):\n    print(i)\n    preds_arr.append(cycle_cv_preds(i, is_test=True))')


def all_cycle_cv_preds(end_cycle, start_cycle=0, n_tta=4, is_test=False):
    return [cycle_cv_preds(i, is_test=is_test)
            for i in range(start_cycle, end_cycle)]


np.savez_compressed(f'{path}tmp/test_preds', preds_arr)


preds_avg = [np.mean(o, 0) for o in preds_arr]
test = np.mean(preds_avg, 0)


get_ipython().magic(u'time preds_arr = all_cycle_cv_preds(5)')


[f2(preds_arr[0][o], data.val_y) for o in range(5)]


preds_avg = [np.mean(o, 0) for o in preds_arr]


ys = [get_data_zoom(f_model, path, 256, 64, n, cv).val_y for cv in range(5)]


f2s = [f2(o, y) for o, y in zip(preds_avg, ys)]
f2s


ots = [opt_th(o, y) for o, y in zip(preds_avg, ys)]
ots


np.mean(ots)


np.mean(f2s, 0)


# ### End
