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

# ## MNIST CNN

# %matplotlib inline

from fastai.vision.all import *

# ### Data

path = untar_data(URLs.MNIST)

path.ls()

items = get_image_files(path)
items[0]

im = PILImageBW.create(items[0])
im.show()

splits = GrandparentSplitter(train_name='training', valid_name='testing')(items)

dsets = Datasets(items, tfms=[[PILImageBW.create], [parent_label, Categorize]], splits=splits)

dsets

dsets.train

dsets.valid

show_at(dsets.train, 0)

tfms = [ToTensor(), CropPad(size=34, pad_mode=PadMode.Zeros), RandomCrop(size=28)]
bs = 128

dls = dsets.dataloaders(bs=bs, after_item=tfms, after_batch=[IntToFloatTensor, Normalize])

dsrc1 = Datasets([items[0]] * 128, tfms=[[PILImageBW.create], [parent_label, Categorize]], splits=[list(range(128)), []])

dbunch1 = dsrc1.dataloaders(bs=bs, after_item=tfms, after_batch=[IntToFloatTensor()])

dbunch1.show_batch(figsize=(8, 8), cmap='gray')

dls.show_batch(figsize=(5, 5))

xb, yb = dls.one_batch()
xb.shape, yb.shape


# ### Basic CNN with batchnorm

def conv(ni, nf): return nn.Conv2d(ni, nf, kernel_size=3, stride=2, padding=1)


model = nn.Sequential(
    conv(1, 8),  # 14
    nn.BatchNorm2d(8),
    nn.ReLU(),
    conv(8, 16),  # 7
    nn.BatchNorm2d(16),
    nn.ReLU(),
    conv(16, 32),  # 4
    nn.BatchNorm2d(32),
    nn.ReLU(),
    conv(32, 16),  # 2
    nn.BatchNorm2d(16),
    nn.ReLU(),
    conv(16, 10),  # 1
    nn.BatchNorm2d(10),
    Flatten()     # remove (1,1) grid
)

learn = Learner(dls, model, loss_func=nn.CrossEntropyLoss(), metrics=accuracy)

print(learn.summary())

xb = xb.cuda()

model(xb).shape

learn.lr_find(end_lr=100)

learn.fit_one_cycle(3, lr_max=0.1)


# ### Refactor

def conv2(ni, nf): return ConvLayer(ni, nf, stride=2)


model = nn.Sequential(
    conv2(1, 8),   # 14
    conv2(8, 16),  # 7
    conv2(16, 32),  # 4
    conv2(32, 16),  # 2
    conv2(16, 10),  # 1
    Flatten()      # remove (1,1) grid
)

learn = Learner(dls, model, loss_func=nn.CrossEntropyLoss(), metrics=accuracy)

learn.fit_one_cycle(10, lr_max=0.1)


# ### Resnet-ish

class ResBlock(Module):
    def __init__(self, nf):
        self.conv1 = ConvLayer(nf, nf)
        self.conv2 = ConvLayer(nf, nf)

    def forward(self, x): return x + self.conv2(self.conv1(x))


model = nn.Sequential(
    conv2(1, 8),
    ResBlock(8),
    conv2(8, 16),
    ResBlock(16),
    conv2(16, 32),
    ResBlock(32),
    conv2(32, 16),
    ResBlock(16),
    conv2(16, 10),
    Flatten()
)


def conv_and_res(ni, nf): return nn.Sequential(conv2(ni, nf), ResBlock(nf))


model = nn.Sequential(
    conv_and_res(1, 8),
    conv_and_res(8, 16),
    conv_and_res(16, 32),
    conv_and_res(32, 16),
    conv2(16, 10),
    Flatten()
)

learn = Learner(dls, model, loss_func=nn.CrossEntropyLoss(), metrics=accuracy)

learn.lr_find(end_lr=100)

learn.fit_one_cycle(12, lr_max=0.05)

print(learn.summary())

# ## fin
