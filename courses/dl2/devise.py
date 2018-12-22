# coding: utf-8
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
# ## Devise
from fastai.conv_learner import *
torch.backends.cudnn.benchmark = True
import fastText as ft
import torchvision.transforms as transforms
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
tfms = transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
fname = 'valid/n01440764/ILSVRC2012_val_00007197.JPEG'
PATH = Path('data/imagenet/')
TMP_PATH = PATH / 'tmp'
TRANS_PATH = Path('data/translate/')
PATH_TRN = PATH / 'train'
img = Image.open(PATH / fname)
import fastai
fastai.dataloader.DataLoader
arch = resnet50
ttfms, vtfms = tfms_from_model(arch, 224, transforms_side_on, max_zoom=1.1)
def to_array(x, y): return np.array(x).astype(np.float32) / 255, None
def TT(x, y): return torch.from_numpy(x), None
ttfms.tfms = [to_array] + ttfms.tfms# + [TT]
ttfms(img)
ft_vecs = ft.load_model(str((TRANS_PATH / 'wiki.en.bin')))
ft_vecs.get_word_vector('king')
np.corrcoef(ft_vecs.get_word_vector('jeremy'), ft_vecs.get_word_vector('Jeremy'))
np.corrcoef(ft_vecs.get_word_vector('banana'), ft_vecs.get_word_vector('Jeremy'))
# ### Map imagenet classes to word vectors
ft_words = ft_vecs.get_words(include_freq=True)
ft_word_dict = {k: v for k, v in zip(*ft_words)}
ft_words = sorted(ft_word_dict.keys(), key=lambda x: ft_word_dict[x])
len(ft_words)
from fastai.io import get_data
CLASSES_FN = 'imagenet_class_index.json'
get_data(f'http://files.fast.ai/models/{CLASSES_FN}', TMP_PATH / CLASSES_FN)
WORDS_FN = 'classids.txt'
get_data(f'http://files.fast.ai/data/{WORDS_FN}', PATH / WORDS_FN)
class_dict = json.load((TMP_PATH / CLASSES_FN).open())
classids_1k = dict(class_dict.values())
nclass = len(class_dict); nclass
class_dict['0']
classid_lines = (PATH / WORDS_FN).open().readlines()
classid_lines[:5]
classids = dict(l.strip().split() for l in classid_lines)
len(classids), len(classids_1k)
lc_vec_d = {w.lower(): ft_vecs.get_word_vector(w) for w in ft_words[-1000000:]}
syn_wv = [(k, lc_vec_d[v.lower()]) for k, v in classids.items()
          if v.lower() in lc_vec_d]
syn_wv_1k = [(k, lc_vec_d[v.lower()]) for k, v in classids_1k.items()
          if v.lower() in lc_vec_d]
syn2wv = dict(syn_wv)
len(syn2wv)
pickle.dump(syn2wv, (TMP_PATH / 'syn2wv.pkl').open('wb'))
pickle.dump(syn_wv_1k, (TMP_PATH / 'syn_wv_1k.pkl').open('wb'))
syn2wv = pickle.load((TMP_PATH / 'syn2wv.pkl').open('rb'))
syn_wv_1k = pickle.load((TMP_PATH / 'syn_wv_1k.pkl').open('rb'))
images = []
img_vecs = []
for d in (PATH / 'train').iterdir():
    if d.name not in syn2wv: continue
    vec = syn2wv[d.name]
    for f in d.iterdir():
        images.append(str(f.relative_to(PATH)))
        img_vecs.append(vec)
n_val = 0
for d in (PATH / 'valid').iterdir():
    if d.name not in syn2wv: continue
    vec = syn2wv[d.name]
    for f in d.iterdir():
        images.append(str(f.relative_to(PATH)))
        img_vecs.append(vec)
        n_val += 1
n_val
img_vecs = np.stack(img_vecs)
img_vecs.shape
pickle.dump(images, (TMP_PATH / 'images.pkl').open('wb'))
pickle.dump(img_vecs, (TMP_PATH / 'img_vecs.pkl').open('wb'))
images = pickle.load((TMP_PATH / 'images.pkl').open('rb'))
img_vecs = pickle.load((TMP_PATH / 'img_vecs.pkl').open('rb'))
arch = resnet50
n = len(images); n
val_idxs = list(range(n - 28650, n))
tfms = tfms_from_model(arch, 224, transforms_side_on, max_zoom=1.1)
md = ImageClassifierData.from_names_and_array(PATH, images, img_vecs, val_idxs=val_idxs,
        classes=None, tfms=tfms, continuous=True, bs=256)
x, y = next(iter(md.val_dl))
models = ConvnetBuilder(arch, md.c, is_multi=False, is_reg=True, xtra_fc=[1024], ps=[0.2, 0.2])
learn = ConvLearner(md, models, precompute=True)
learn.opt_fn = partial(optim.Adam, betas=(0.9, 0.99))
def cos_loss(inp, targ): return 1 - F.cosine_similarity(inp, targ).mean()
learn.crit = cos_loss
learn.lr_find(start_lr=1e-4, end_lr=1e15)
learn.sched.plot()
lr = 1e-2
wd = 1e-7
learn.precompute = True
learn.fit(lr, 1, cycle_len=20, wds=wd, use_clr=(20, 10))
learn.bn_freeze(True)
learn.fit(lr, 1, cycle_len=20, wds=wd, use_clr=(20, 10))
lrs = np.array([lr / 1000, lr / 100, lr])
learn.precompute = False
learn.freeze_to(1)
learn.save('pre0')
learn.load('pre0')
# ## Image search
# ### Search imagenet classes
syns, wvs = list(zip(*syn_wv_1k))
wvs = np.array(wvs)
get_ipython().run_line_magic('time', 'pred_wv = learn.predict()')
start = 300
denorm = md.val_ds.denorm
def show_img(im, figsize=None, ax=None):
    if not ax: fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.axis('off')
    return ax
def show_imgs(ims, cols, figsize=None):
    fig, axes = plt.subplots(len(ims) // cols, cols, figsize=figsize)
    for i, ax in enumerate(axes.flat): show_img(ims[i], ax=ax)
    plt.tight_layout()
show_imgs(denorm(md.val_ds[start:start + 25][0]), 5, (10, 10))
import nmslib
def create_index(a):
    index = nmslib.init(space='angulardist')
    index.addDataPointBatch(a)
    index.createIndex()
    return index
def get_knns(index, vecs):
     return zip(*index.knnQueryBatch(vecs, k=10, num_threads=4))
def get_knn(index, vec): return index.knnQuery(vec, k=10)
nn_wvs = create_index(wvs)
idxs, dists = get_knns(nn_wvs, pred_wv)
[[classids[syns[id]] for id in ids[:3]] for ids in idxs[start:start + 10]]
# ### Search all wordnet noun classes
all_syns, all_wvs = list(zip(*syn2wv.items()))
all_wvs = np.array(all_wvs)
nn_allwvs = create_index(all_wvs)
idxs, dists = get_knns(nn_allwvs, pred_wv)
[[classids[all_syns[id]] for id in ids[:3]] for ids in idxs[start:start + 10]]
# ### Text -> image search
nn_predwv = create_index(pred_wv)
en_vecd = pickle.load(open(TRANS_PATH / 'wiki.en.pkl', 'rb'))
vec = en_vecd['boat']
idxs, dists = get_knn(nn_predwv, vec)
show_imgs([open_image(PATH / md.val_ds.fnames[i]) for i in idxs[:3]], 3, figsize=(9, 3));
vec = (en_vecd['engine'] + en_vecd['boat']) / 2
idxs, dists = get_knn(nn_predwv, vec)
show_imgs([open_image(PATH / md.val_ds.fnames[i]) for i in idxs[:3]], 3, figsize=(9, 3));
vec = (en_vecd['sail'] + en_vecd['boat']) / 2
idxs, dists = get_knn(nn_predwv, vec)
show_imgs([open_image(PATH / md.val_ds.fnames[i]) for i in idxs[:3]], 3, figsize=(9, 3));
# ### Image->image
fname = 'valid/n01440764/ILSVRC2012_val_00007197.JPEG'
img = open_image(PATH / fname)
show_img(img);
t_img = md.val_ds.transform(img)
pred = learn.predict_array(t_img[None])
idxs, dists = get_knn(nn_predwv, pred)
show_imgs([open_image(PATH / md.val_ds.fnames[i]) for i in idxs[1:4]], 3, figsize=(9, 3));
