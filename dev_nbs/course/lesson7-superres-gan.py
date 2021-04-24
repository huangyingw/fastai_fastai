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

# ## Pretrained GAN

from crappify import *
from fastai.vision.all import *
from fastai.vision.gan import *

path = untar_data(URLs.PETS)
path_hr = path / 'images'
path_lr = path / 'crappy'

# ## Crappified data

# Prepare the input data by crappifying images.


# Uncomment the first time you run this notebook.

# +
#items = get_image_files(path_hr)
#parallel(crappifier(path_lr, path_hr), items);
# -

# For gradual resizing we can change the commented line here.

bs, size = 32, 128
# bs,size = 24,160
#bs,size = 8,256
arch = resnet34


# ## Pre-train generator

# Now let's pretrain the generator.

def get_dls(bs, size):
    dblock = DataBlock(blocks=(ImageBlock, ImageBlock),
                       get_items=get_image_files,
                       get_y=lambda x: path_hr / x.name,
                       splitter=RandomSplitter(),
                       item_tfms=Resize(size),
                       batch_tfms=[*aug_transforms(max_zoom=2.), Normalize.from_stats(*imagenet_stats)])
    dls = dblock.dataloaders(path_lr, bs=bs, path=path)
    dls.c = 3
    return dls


dls_gen = get_dls(bs, size)

dls_gen.show_batch(max_n=4)

wd = 1e-3

y_range = (-3., 3.)

loss_gen = MSELossFlat()


def create_gen_learner():
    return unet_learner(dls_gen, arch, loss_func=loss_gen,
                        config=unet_config(blur=True, norm_type=NormType.Weight, self_attention=True, y_range=y_range))


learn_gen = create_gen_learner()

learn_gen.fit_one_cycle(2, pct_start=0.8, wd=wd)

learn_gen.unfreeze()

learn_gen.fit_one_cycle(3, slice(1e-6, 1e-3), wd=wd)

learn_gen.show_results(max_n=4)

learn_gen.save('gen-pre2')

# ## Save generated images

learn_gen.load('gen-pre2')

name_gen = 'image_gen'
path_gen = path / name_gen

# +
# shutil.rmtree(path_gen)
# -

path_gen.mkdir(exist_ok=True)


def save_preds(dl, learn):
    names = dl.dataset.items

    preds, _ = learn.get_preds(dl=dl)
    for i, pred in enumerate(preds):
        dec = dl.after_batch.decode((TensorImage(pred[None]),))[0][0]
        arr = dec.numpy().transpose(1, 2, 0)
        Image.fromarray(np.uint8(arr * 255), mode='RGB').save(path_gen / names[i].name)


# Remove shuffle, drop_last and data aug from the training set

dl = dls_gen.train.new(shuffle=False, drop_last=False, after_batch=[IntToFloatTensor, Normalize.from_stats(*imagenet_stats)])

save_preds(dl, learn_gen)


# ## Train critic

# Pretrain the critic on crappy vs not crappy.

def get_crit_dls(bs, size):
    crit_dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                            get_items=partial(get_image_files, folders=[name_gen, 'images']),
                            get_y=parent_label,
                            splitter=RandomSplitter(0.1, seed=42),
                            item_tfms=Resize(size),
                            batch_tfms=[Normalize.from_stats(*imagenet_stats)])
    return crit_dblock.dataloaders(path, bs=bs, path=path)


dls_crit = get_crit_dls(bs=bs, size=size)

dls_crit.show_batch()

loss_critic = AdaptiveLoss(nn.BCEWithLogitsLoss())


def create_critic_learner(dls, metrics):
    return Learner(dls, gan_critic(), metrics=metrics, loss_func=loss_critic)


learn_critic = create_critic_learner(dls_crit, accuracy_thresh_expand)

learn_critic.fit_one_cycle(6, 1e-3, wd=wd)

learn_critic.save('critic-pre2')


# ## GAN

# Now we'll combine those pretrained model in a GAN.

def get_crit_dls(bs, size):
    crit_dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                            get_items=partial(get_image_files, folders=['crappy', 'images']),
                            get_y=parent_label,
                            splitter=RandomSplitter(0.1, seed=42),
                            item_tfms=Resize(size),
                            batch_tfms=[Normalize.from_stats(*imagenet_stats)])
    return crit_dblock.dataloaders(path, bs=bs, path=path)


dls_crit = get_crit_dls(bs=bs, size=size)

learn_crit = create_critic_learner(dls_crit, metrics=None).load('critic-pre2')

learn_gen = create_gen_learner().load('gen-pre2')


# To define a GAN Learner, we just have to specify the learner objects foor the generator and the critic. The switcher is a callback that decides when to switch from discriminator to generator and vice versa. Here we do as many iterations of the discriminator as needed to get its loss back < 0.5 then one iteration of the generator.
#
# The loss of the critic is given by `learn_crit.loss_func`. We take the average of this loss function on the batch of real predictions (target 1) and the batch of fake predicitions (target 0).
#
# The loss of the generator is weighted sum (weights in `weights_gen`) of `learn_crit.loss_func` on the batch of fake (passed throught the critic to become predictions) with a target of 1, and the `learn_gen.loss_func` applied to the output (batch of fake) and the target (corresponding batch of superres images).

class GANDiscriminativeLR(Callback):
    "`Callback` that handles multiplying the learning rate by `mult_lr` for the critic."

    def __init__(self, mult_lr=5.): self.mult_lr = mult_lr

    def before_batch(self):
        "Multiply the current lr if necessary."
        if not self.learn.gan_trainer.gen_mode and self.training:
            self.learn.opt.set_hyper('lr', learn.opt.hypers[0]['lr'] * self.mult_lr)

    def after_batch(self):
        "Put the LR back to its value if necessary."
        if not self.learn.gan_trainer.gen_mode:
            self.learn.opt.set_hyper('lr', learn.opt.hypers[0]['lr'] / self.mult_lr)


switcher = AdaptiveGANSwitcher(critic_thresh=0.65)
learn = GANLearner.from_learners(learn_gen, learn_crit, weights_gen=(1., 50.), show_img=False, switcher=switcher,
                                 opt_func=partial(Adam, mom=0.), cbs=GANDiscriminativeLR(mult_lr=5.))

lr = 1e-4

learn.fit(10, lr, wd=wd)

learn.show_results(max_n=4)

# ## fin
