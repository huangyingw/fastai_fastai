
# coding: utf-8

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


from fastai.learner import *
from fastai.dataset import *


X = np.array([[0., 0.], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
data = (X, y)


md = ImageClassifierData.from_arrays('.', data, data, bs=4)


learn = Learner.from_model_data(SimpleNet([2, 10, 2]), md)
learn.crit = nn.CrossEntropyLoss()
learn.opt_fn = optim.SGD


learn.fit(1., 30, metrics=[accuracy])
