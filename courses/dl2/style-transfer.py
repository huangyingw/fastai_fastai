# coding: utf-8
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
# ## Style transfer
from fastai.conv_learner import *
from pathlib import Path
from scipy import ndimage
torch.cuda.set_device(3)
torch.backends.cudnn.benchmark = True
# wget http://files.fast.ai/data/imagenet-sample-train.tar.gz
PATH = Path('data/imagenet')
PATH_TRN = PATH / 'train'
m_vgg = to_gpu(vgg16(True)).eval()
set_trainable(m_vgg, False)
img_fn = PATH_TRN / 'n01558993' / 'n01558993_9684.JPEG'
img = open_image(img_fn)
plt.imshow(img);
sz = 288
trn_tfms, val_tfms = tfms_from_model(vgg16, sz)
img_tfm = val_tfms(img)
img_tfm.shape
opt_img = np.random.uniform(0, 1, size=img.shape).astype(np.float32)
plt.imshow(opt_img);
opt_img = scipy.ndimage.filters.median_filter(opt_img, [8, 8, 1])
plt.imshow(opt_img);
opt_img = val_tfms(opt_img) / 2
opt_img_v = V(opt_img[None], requires_grad=True)
opt_img_v.shape
m_vgg = nn.Sequential(*children(m_vgg)[:37])
targ_t = m_vgg(VV(img_tfm[None]))
targ_v = V(targ_t)
targ_t.shape
max_iter = 1000
show_iter = 100
optimizer = optim.LBFGS([opt_img_v], lr=0.5)
def actn_loss(x): return F.mse_loss(m_vgg(x), targ_v) * 1000
def step(loss_fn):
    global n_iter
    optimizer.zero_grad()
    loss = loss_fn(opt_img_v)
    loss.backward()
    n_iter += 1
    if n_iter % show_iter == 0: print(f'Iteration: {n_iter}, loss: {loss.data[0]}')
    return loss
n_iter = 0
while n_iter <= max_iter: optimizer.step(partial(step, actn_loss))
x = val_tfms.denorm(np.rollaxis(to_np(opt_img_v.data), 1, 4))[0]
plt.figure(figsize=(7, 7))
plt.imshow(x);
# ## forward hook
class SaveFeatures():
    features = None

    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output): self.features = output

    def close(self): self.hook.remove()
m_vgg = to_gpu(vgg16(True)).eval()
set_trainable(m_vgg, False)
block_ends = [i - 1 for i, o in enumerate(children(m_vgg))
              if isinstance(o, nn.MaxPool2d)]
block_ends
sf = SaveFeatures(children(m_vgg)[block_ends[3]])
def get_opt():
    opt_img = np.random.uniform(0, 1, size=img.shape).astype(np.float32)
    opt_img = scipy.ndimage.filters.median_filter(opt_img, [8, 8, 1])
    opt_img_v = V(val_tfms(opt_img / 2)[None], requires_grad=True)
    return opt_img_v, optim.LBFGS([opt_img_v])
opt_img_v, optimizer = get_opt()
m_vgg(VV(img_tfm[None]))
targ_v = V(sf.features.clone())
targ_v.shape
def actn_loss2(x):
    m_vgg(x)
    out = V(sf.features)
    return F.mse_loss(out, targ_v) * 1000
n_iter = 0
while n_iter <= max_iter: optimizer.step(partial(step, actn_loss2))
x = val_tfms.denorm(np.rollaxis(to_np(opt_img_v.data), 1, 4))[0]
plt.figure(figsize=(7, 7))
plt.imshow(x);
sf.close()
# ## Style match
# wget https://raw.githubusercontent.com/jeffxtang/fast-style-transfer/master/images/starry_night.jpg
style_fn = PATH / 'style' / 'starry_night.jpg'
style_img = open_image(style_fn)
style_img.shape, img.shape
plt.imshow(style_img);
def scale_match(src, targ):
    h, w, _ = src.shape
    sh, sw, _ = targ.shape
    rat = max(h / sh, w / sw); rat
    res = cv2.resize(targ, (int(sw * rat), int(sh * rat)))
    return res[:h, :w]
style = scale_match(img, style_img)
plt.imshow(style)
style.shape, img.shape
opt_img_v, optimizer = get_opt()
sfs = [SaveFeatures(children(m_vgg)[idx]) for idx in block_ends]
m_vgg(VV(img_tfm[None]))
targ_vs = [V(o.features.clone()) for o in sfs]
[o.shape for o in targ_vs]
style_tfm = val_tfms(style_img)
m_vgg(VV(style_tfm[None]))
targ_styles = [V(o.features.clone()) for o in sfs]
[o.shape for o in targ_styles]
def gram(input):
        b, c, h, w = input.size()
        x = input.view(b * c, -1)
        return torch.mm(x, x.t()) / input.numel() * 1e6
def gram_mse_loss(input, target): return F.mse_loss(gram(input), gram(target))
def style_loss(x):
    m_vgg(opt_img_v)
    outs = [V(o.features) for o in sfs]
    losses = [gram_mse_loss(o, s) for o, s in zip(outs, targ_styles)]
    return sum(losses)
n_iter = 0
while n_iter <= max_iter: optimizer.step(partial(step, style_loss))
x = val_tfms.denorm(np.rollaxis(to_np(opt_img_v.data), 1, 4))[0]
plt.figure(figsize=(7, 7))
plt.imshow(x);
for sf in sfs: sf.close()
# ## Style transfer
opt_img_v, optimizer = get_opt()
sfs = [SaveFeatures(children(m_vgg)[idx]) for idx in block_ends]
def comb_loss(x):
    m_vgg(opt_img_v)
    outs = [V(o.features) for o in sfs]
    losses = [gram_mse_loss(o, s) for o, s in zip(outs, targ_styles)]
    cnt_loss = F.mse_loss(outs[3], targ_vs[3]) * 1000000
    style_loss = sum(losses)
    return cnt_loss + style_loss
n_iter = 0
while n_iter <= max_iter: optimizer.step(partial(step, comb_loss))
x = val_tfms.denorm(np.rollaxis(to_np(opt_img_v.data), 1, 4))[0]
plt.figure(figsize=(9, 9))
plt.imshow(x, interpolation='lanczos')
plt.axis('off');
for sf in sfs: sf.close()
