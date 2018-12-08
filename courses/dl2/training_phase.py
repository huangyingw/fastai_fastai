
# coding: utf-8

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# # The TrainPhase API

# *This notebook was prepared by Sylvain Gugger - many thanks!*
#
# Here we show how to use a new API in the fastai library, that allows you all the flexibility you might want while training your model.
#
# All the examples will run on cifar10, so be sure to change the path to a directory that contains this dataset, with the usual hierarchy (a train and a valid folder, each of them containing ten subdirectories for each class).

from fastai.conv_learner import *
PATH = Path("../data/cifar10/")


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
stats = (np.array([0.4914, 0.48216, 0.44653]), np.array([0.24703, 0.24349, 0.26159]))


# This will allow us to grab data for a given image size and batch size.

def get_data(sz, bs):
    tfms = tfms_from_stats(stats, sz, aug_tfms=[RandomFlip()], pad=sz // 8)
    return ImageClassifierData.from_paths(PATH, val_name='test', tfms=tfms, bs=bs)


size = 32
batch_size = 64


data = get_data(size, batch_size)


# Now let's create a very simple model that we'll train: a neural net with a hidden layer.

def SimpleNet(layers):
    list_layers = [Flatten()]
    for i in range(len(layers) - 1):
        list_layers.append(nn.Linear(layers[i], layers[i + 1]))
        if i < len(layers) - 2: list_layers.append(nn.ReLU(inplace=True))
        else: list_layers.append(nn.LogSoftmax(dim=0))
    return nn.Sequential(*list_layers)


learn = ConvLearner.from_model_data(SimpleNet([32 * 32 * 3, 40, 10]), data)


# Now we can use our learner object to give examples of traning.
#
# With the new API, you don't use a pre-implemented training schedule but you can design your own with object called TrainingPhase. A training phase is just a class that will record all the parameters you want to apply during this part of the training loop, specifically:
# - a number of epochs for which these settings will be valid (can be a float)
# - an optimizer function (SGD, RMSProp, Adam...)
# - a learning rate (or array of lrs) or a range of learning rates (or array of lrs) if you want to change the lr.
# - a learning rate decay method (that will explain how you want to change the lr)
# - a momentum (which will beta1 if you're using Adam), or a range of momentums if you want to change it
# - a momentum decay method (that will explain how you want to change the momentum, if applicable)
# - optionally a weight decay (or array of wds)
# - optionally a beta parameter (which is the RMSProp alpha or the Adam beta2, if you want another vlaue than default)
#
# By combining those blocks as you wish, you can implement pretty much any method of training you could think of.

# # Basic lr decay

# Let's begin with something basic and say you want to train with SGD and momentum, with a learning rate of 1e-2 for 1 epoch then 1e-3 for two epochs. We'll just create a list of two phases for this.

phases = [TrainingPhase(epochs=1, opt_fn=optim.SGD, lr=1e-2), TrainingPhase(epochs=2, opt_fn=optim.SGD, lr=1e-3)]


# Note that we didn't set the momentum parameter because it will default to 0.9. If you don't want any momentum, you'll have to put it to 0.
#
# Now that we have created this list of phases, we just have to call fit_opt_sched.

learn.fit_opt_sched(phases)


# If we want to see what we did, we can use learn.sched.plot_lr()

learn.sched.plot_lr()


# The red dashed line represent the change of phase, and the optimizer name (plus its optional parameters) is indicated, in case you changed it. You can remove that off by using show_text=False

learn.sched.plot_lr(show_text=False)


# Here the momentums don't change at all so we might want to hide them. This is possible with just a simple option.

learn.sched.plot_lr(show_text=False, show_moms=False)


# Now let's complicate things a bit more and say we want to train at constant 1e-2 lr for one epoch, then decrease our learning rate to 1e-3 during one epoch and train for one last epoch at 1e-3.
#
# We have currently fours methods of decay (on top of NO that means constant): linear, cosine, exponential or polynomial. Let's have a look at what each of them does.

phases = [TrainingPhase(epochs=1, opt_fn=optim.SGD, lr=1e-2),
          TrainingPhase(epochs=1, opt_fn=optim.SGD, lr=(1e-2, 1e-3), lr_decay=DecayType.LINEAR),
          TrainingPhase(epochs=1, opt_fn=optim.SGD, lr=1e-3)]


learn.fit_opt_sched(phases)


# Linear is simply going from a to b with a line. The formula that gives the learning rate at batch i over n is:

lr_i = start_lr + (end_lr - start_lr) * i / n


# (to run the cell above give a value to all the parameters: start_lr, end_lr, i and n)

learn.sched.plot_lr(show_moms=False)


phases = [TrainingPhase(epochs=1, opt_fn=optim.SGD, lr=1e-2),
          TrainingPhase(epochs=1, opt_fn=optim.SGD, lr=(1e-2, 1e-3), lr_decay=DecayType.COSINE),
          TrainingPhase(epochs=1, opt_fn=optim.SGD, lr=1e-3)]


learn.fit_opt_sched(phases)


# Cosine is simply going from a to b with a following half a cosine. The formula that gives the learning rate at batch i over n is:

lr_i = end_lr + (start_lr - end_lr) / 2 * (1 + np.cos(i * np.pi) / n)


learn.sched.plot_lr(show_moms=False)


phases = [TrainingPhase(epochs=1, opt_fn=optim.SGD, lr=1e-2),
          TrainingPhase(epochs=1, opt_fn=optim.SGD, lr=(1e-2, 1e-3), lr_decay=DecayType.EXPONENTIAL),
          TrainingPhase(epochs=1, opt_fn=optim.SGD, lr=1e-3)]


learn.fit_opt_sched(phases)


# Exponential is multiplying the learning rate by the same thing at each step, this thing being computed to be exactly what's needed to go from our start point to our end point. Here the learning rate on batch i over n is:

lr_i = start_lr * (end_lr / start_lr)**(i / n)


learn.sched.plot_lr(show_moms=False)


phases = [TrainingPhase(epochs=1, opt_fn=optim.SGD, lr=1e-2),
          TrainingPhase(epochs=1, opt_fn=optim.SGD, lr=(1e-2, 1e-3), lr_decay=(DecayType.POLYNOMIAL, 2)),
          TrainingPhase(epochs=1, opt_fn=optim.SGD, lr=1e-3)]


learn.fit_opt_sched(phases)


# Note that this last POLYNOMIAL decay type needs a second argument: it's the value of the power you want in your polynomial decay. The formula that gives your update is:

lr_i = end_lr + (start_lr - end_lr) * (1 - i / n) ** p


# where p is this extra argument. Below, we can see the result for p = 2.

learn.sched.plot_lr(show_moms=False)


phases = [TrainingPhase(epochs=1, opt_fn=optim.SGD, lr=1e-2),
          TrainingPhase(epochs=1, opt_fn=optim.SGD, lr=(1e-2, 1e-3), lr_decay=(DecayType.POLYNOMIAL, 0.5)),
          TrainingPhase(epochs=1, opt_fn=optim.SGD, lr=1e-3)]


learn.fit_opt_sched(phases)


# And here is an example where p = 0.5. In general, the greater p is, the more 'exponential' your curve will look (p=2 wasn't very different from exponential already). p=1 is simply linear (and close values will give something close to a line). Lower values of p (close to 0) will give a curve that stays up a bit longer before going down.
#

learn.sched.plot_lr(show_moms=False)


# If you don't specify an end value to the lr, it will be assumed the value you gave is the starting value and the end value is 0. This doesn't work for the EXPONENTIAL type of decay.

phases = [TrainingPhase(epochs=1, opt_fn=optim.SGD, lr=1e-2),
          TrainingPhase(epochs=1, opt_fn=optim.SGD, lr=1e-2, lr_decay=DecayType.COSINE),
          TrainingPhase(epochs=1, opt_fn=optim.SGD, lr=1e-3)]


learn.fit_opt_sched(phases)


learn.sched.plot_lr(show_moms=False)


# All of those decays are implemented in the file sgdr.py in the function next_val of the class DecayScheduler. Adding your own is as simple as:
#    - adding it in the DecayType class
#    - implementing your own formula in the function next_val.

# # SGDR

# The traditional stochastic gradient with restart we used requires three arguments:
# - the number of cycles
# - the length of a cycle (cycle_len)
# - the value by which we should multiply the cycle_len after each cycle (cycle_mult)
#
# Also, note there is a warm-up at constant lr of 1/100th of the max value during 1/20th of the first cycle.
#
# This can easily be implemented in a function (which you should now try to do by yourself) which creates a list of phases (number of cycles + 1 here). Remember you can have a float number for the epochs in a Training Phase.

def phases_sgdr(lr, opt_fn, num_cycle, cycle_len, cycle_mult):
    phases = [TrainingPhase(epochs=cycle_len / 20, opt_fn=opt_fn, lr=lr / 100),
              TrainingPhase(epochs=cycle_len * 19 / 20, opt_fn=opt_fn, lr=lr, lr_decay=DecayType.COSINE)]
    for i in range(1, num_cycle):
        phases.append(TrainingPhase(epochs=cycle_len * (cycle_mult**i), opt_fn=opt_fn, lr=lr, lr_decay=DecayType.COSINE))
    return phases


learn.fit_opt_sched(phases_sgdr(1e-2, optim.Adam, 2, 1, 2))


learn.sched.plot_lr(show_text=False, show_moms=False)


# Looks familiar?

# # 1cycle

# Leslie Smith's 1cycle policy states to pick a maximum learning rate with our traditional learning rate finder, choose a factor div by which to divide this learning rate, then do two phases of equal length going linearly from our minimal lr to our maximal lr, then back to the minimum.
#
# In parallel, the momentum should begin high (like 0.95) and decrease to a minimum (like 0.85) as the lr grows, then returns to the maximum as the lr decreases.
#
# To complete this cycle, allow a bit of time to let the learning rate decrease even more at the end (I chose to go from the minimum_lr to 1/100th of its value linearly since it was the thing that seemed to work the best in my experiments).

# So let's create a function 1cycle that takes the arguments:
# - cycle_len: the total length of the cycle (in epochs)
# - lr: the maximum learning rate
# - div: by how much do we want to divide the maximum lr
# - pct: what percentage of epochs should be left at the end for the final annealing
# - max_mom: the maximum momentum
# - min_mom: the minimum momentum
#
# The optim function should be SGD (with momentum).

def phases_1cycle(cycle_len, lr, div, pct, max_mom, min_mom):
    tri_cyc = (1 - pct / 100) * cycle_len
    return [TrainingPhase(epochs=tri_cyc / 2, opt_fn=optim.SGD, lr=(lr / div, lr), lr_decay=DecayType.LINEAR,
                          momentum=(max_mom, min_mom), momentum_decay=DecayType.LINEAR),
           TrainingPhase(epochs=tri_cyc / 2, opt_fn=optim.SGD, lr=(lr, lr / div), lr_decay=DecayType.LINEAR,
                          momentum=(min_mom, max_mom), momentum_decay=DecayType.LINEAR),
           TrainingPhase(epochs=cycle_len - tri_cyc, opt_fn=optim.SGD, lr=(lr / div, lr / (100 * div)), lr_decay=DecayType.LINEAR,
                          momentum=max_mom)]


learn.fit_opt_sched(phases_1cycle(3, 1e-2, 10, 10, 0.95, 0.85))


learn.sched.plot_lr(show_text=False)


# Now you can easily test different types of annealing at the end of the 1cycle.

# # It supports discriminative learning rates.

# When you unfreeze a pretrained model, you often use differential learning rates, and this works with this new API too. Just pass an array or a list of learning rates instead of a single value.

learn = ConvLearner.pretrained(resnet34, data, metrics=[accuracy])


learn.unfreeze()
lr = 1e-2
lrs = np.array([lr / 100, lr / 10, lr])


phases = [TrainingPhase(epochs=1, opt_fn=optim.Adam, lr=(lrs / 10, lrs), lr_decay=DecayType.LINEAR),
          TrainingPhase(epochs=2, opt_fn=optim.Adam, lr=lrs, lr_decay=DecayType.COSINE)]


learn.fit_opt_sched(phases)


# What is plotted in these cases is the highest learning rate.

learn.sched.plot_lr(show_text=False, show_moms=False)


# # A customized LR Finder

# This API can also be used to run a learning rate finder: just put a very low starting LR and a very large ending one, and choose exponential or linear decay. To stop when the loss go wild, we add the option stop_div= True. This can also be used in a regular fit if you want to stop the training in case the loss suddenly spikes.
#
# As you can choose the number of epochs, this is particularly useful when you have a small dataset (you can run more epoxhs to have a clearer curve) or a very large one where a fraction of an epoch is plenty. In general 100-200 values are plenty to have a clear curve, so since we know we have 782 batches on cifar10, we'll run this customized LR Finder on 0.25 epochs.

phases = [TrainingPhase(epochs=0.25, opt_fn=optim.SGD, lr=(1e-5, 10), lr_decay=DecayType.EXPONENTIAL, momentum=0.9)]


learn = ConvLearner.from_model_data(SimpleNet([32 * 32 * 3, 40, 10]), data)


# Be careful to save your model before fitting as the LR Finder will alter it.

learn.save('tmp')
learn.fit_opt_sched(phases, stop_div=True)
learn.load('tmp')


# Then we can draw the curve with the usual command:

learn.sched.plot()


# If you choose a linear decay, the curve will be plotted without using a log scale for the learning rates. Be careful than we you use a linear scale for the lrs, you get very fast to the high ones so you can't put as wide a range.

phases = [TrainingPhase(epochs=0.25, opt_fn=optim.SGD, lr=(0.001, 0.1), lr_decay=DecayType.LINEAR, momentum=0.9)]
learn.save('tmp')
learn.fit_opt_sched(phases, stop_div=True)
learn.load('tmp')


learn.sched.plot()


# You can always force the scale to be linear (or log scale) by using the optional argument linear in the plot function.

phases = [TrainingPhase(epochs=0.25, opt_fn=optim.SGD, lr=(1e-5, 10), lr_decay=DecayType.EXPONENTIAL, momentum=0.9)]
learn.save('tmp')
learn.fit_opt_sched(phases, stop_div=True)
learn.load('tmp')


learn.sched.plot(linear=True)


# # But there's more!

# The first thing that's possible with these blocks and didn't try yet is that you can change the optimizer at each phase. For instance, we can do a 1cycle with SGD and cyclical momentum but then anneal with Adam.

phases = [TrainingPhase(epochs=1, opt_fn=optim.SGD, lr=(1e-3, 1e-2), lr_decay=DecayType.LINEAR,
                          momentum=(0.95, 0.85), momentum_decay=DecayType.LINEAR),
           TrainingPhase(epochs=1, opt_fn=optim.SGD, lr=(1e-2, 1e-3), lr_decay=DecayType.LINEAR,
                          momentum=(0.85, 0.95), momentum_decay=DecayType.LINEAR),
           TrainingPhase(epochs=1, opt_fn=optim.Adam, lr=1e-3, lr_decay=DecayType.COSINE, momentum=0.9)]


learn.fit_opt_sched(phases)


learn.sched.plot_lr()


# Lastly, you can even change your data during the training. This is only applicable for a full CNN that can works with any size, but you could decide to train it for a bit with a smaller size before increasing it.

def ConvBN(n_in, n_out, stride):
    return nn.Sequential(nn.Conv2d(n_in, n_out, 3, stride=stride, padding=1), nn.BatchNorm2d(n_out))


def ShallowConvNet():
    listlayers = [ConvBN(3, 64, 20), nn.ReLU(inplace=True), ConvBN(64, 128, 2), nn.ReLU(inplace=True),
                  nn.AdaptiveAvgPool2d(1), Flatten(), nn.Linear(128, 10), nn.LogSoftmax(dim=0)]
    return nn.Sequential(*listlayers)


# Let's grab the data for two different sizes.

data1 = get_data(28, batch_size)
data2 = get_data(32, batch_size)


# And create a learner object.

learn = ConvLearner.from_model_data(ShallowConvNet(), data1)


# And let's say we want to phases of cos anneal with Adam, but want to train on the size 28 for the first, then on the size 32.

phases = [TrainingPhase(epochs=1, opt_fn=optim.Adam, lr=1e-2, lr_decay=DecayType.COSINE),
          TrainingPhase(epochs=2, opt_fn=optim.Adam, lr=1e-2, lr_decay=DecayType.COSINE)]


# It's as simple as passing a list of data in the arguments of fir_opt_sched. One thing to pay attention to, this list must have the same size as phases, so if the same data object should be used through multiple phases, repeat it as needed.

learn.fit_opt_sched(phases, data_list=[data1, data2])


learn.sched.plot_lr(show_moms=False)
