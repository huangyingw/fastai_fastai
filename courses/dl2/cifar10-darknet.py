# coding: utf-8
# ## CIFAR 10
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from fastai.conv_learner import *
PATH = Path("data/cifar10/")
os.makedirs(PATH, exist_ok=True)
torch.cuda.set_device(1)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
stats = (np.array([0.4914, 0.48216, 0.44653]), np.array([0.24703, 0.24349, 0.26159]))
num_workers = num_cpus() // 2
bs = 256
sz = 32
tfms = tfms_from_stats(stats, sz, aug_tfms=[RandomFlip()], pad=sz // 8)
data = ImageClassifierData.from_paths(PATH, val_name='test', tfms=tfms, bs=bs)
def conv_layer(ni, nf, ks=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(ni, nf, kernel_size=ks, bias=False, stride=stride, padding=ks // 2),
        nn.BatchNorm2d(nf, momentum=0.01),
        nn.LeakyReLU(negative_slope=0.1, inplace=True))
class ResLayer(nn.Module):
    def __init__(self, ni):
        super().__init__()
        self.conv1 = conv_layer(ni, ni // 2, ks=1)
        self.conv2 = conv_layer(ni // 2, ni, ks=3)
        
    def forward(self, x):
        # changed to x.add, as x.add_ leads to error (happened on single GPU):
        # one of the variables needed for gradient computation has been modified by an inplace operation
        return x.add(self.conv2(self.conv1(x)))
class Darknet(nn.Module):
    def make_group_layer(self, ch_in, num_blocks, stride=1):
        return [conv_layer(ch_in, ch_in * 2, stride=stride)
               ] + [(ResLayer(ch_in * 2)) for i in range(num_blocks)]

    def __init__(self, num_blocks, num_classes, nf=32):
        super().__init__()
        layers = [conv_layer(3, nf, ks=3, stride=1)]
        for i, nb in enumerate(num_blocks):
            layers += self.make_group_layer(nf, nb, stride=2 - (i == 1))
            nf *= 2
        layers += [nn.AdaptiveAvgPool2d(1), Flatten(), nn.Linear(nf, num_classes)]
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x): return self.layers(x)
m = Darknet([1, 2, 4, 6, 3], num_classes=10, nf=32)
m = nn.DataParallel(m, device_ids=None)
# if you have several GPUs for true parallel processing enable
# m = nn.DataParallel(m, device_ids=[1, 2, 3])
lr = 1.3
learn = ConvLearner.from_model_data(m, data)
learn.crit = nn.CrossEntropyLoss()
learn.metrics = [accuracy]
wd = 1e-4
get_ipython().run_line_magic('time', 'learn.fit(lr, 1, wds=wd, cycle_len=30, use_clr_beta=(20, 20, 0.95, 0.85))')
# DP: m = WideResNet(depth=22, num_classes=10, widen_factor=6, dropRate=0.)
learn.fit(lr / 10, 1, wds=wd, cycle_len=1, use_clr_beta=(100, 1, 0.9, 0.8))
get_ipython().run_line_magic('time', 'learn.fit(lr, 1, wds=wd, cycle_len=30, use_clr_beta=(20, 20, 0.95, 0.85))')
learn.fit(lr / 10, 1, wds=wd, cycle_len=1, use_clr_beta=(100, 1, 0.9, 0.8))
get_ipython().run_line_magic('time', 'learn.fit(lr, 1, wds=wd, cycle_len=40, use_clr_beta=(10, 15, 0.95, 0.85))')
learn.fit(lr / 10, 1, wds=wd, cycle_len=1, use_clr_beta=(100, 1, 0.9, 0.8))
get_ipython().run_line_magic('time', 'learn.fit(1., 1, wds=wd, cycle_len=30, use_clr_beta=(10, 25, 0.95, 0.85))')
get_ipython().run_line_magic('time', 'learn.fit(lr, 1, wds=wd, cycle_len=40, use_clr_beta=(100, 15, 0.95, 0.85))')
# darknet 2222 lr 1.3 65 cl
get_ipython().run_line_magic('time', 'learn.fit(lr, 1, wds=wd, cycle_len=65, use_clr_beta=(30, 20, 0.95, 0.85))')
