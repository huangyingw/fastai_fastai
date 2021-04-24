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

# hide
# skip
from fastai.vision.core import TensorPoint, TensorBBox
from nbdev.export import *
from transformers import AutoTokenizer, AutoModel
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from fastai.text.all import TextDataLoaders, text_classifier_learner, AWD_LSTM
from fastai.vision.all import Resize, RandomSubsetSplitter, aug_transforms, cnn_learner, resnet18
from fastai.vision.data import *
from fastai.text.all import LMLearner, TextLearner
from fastai.callback.hook import hook_output
from fastai.callback.fp16 import ModelToHalf
from torch.utils.tensorboard import SummaryWriter
import tensorboard
from fastai.basics import *
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# +
# default_exp callback.tensorboard

# +
# all_slow
# -

# export

# # Tensorboard
#
# > Integration with [tensorboard](https://www.tensorflow.org/tensorboard)

# First thing first, you need to install tensorboard with
# ```
# pip install tensorboard
# ```
# Then launch tensorboard with
# ```
# tensorboard --logdir=runs
# ```
# in your terminal. You can change the logdir as long as it matches the `log_dir` you pass to `TensorBoardCallback` (default is `runs` in the working directory).

# ## Tensorboard Embedding Projector support

# > Tensorboard Embedding Projector is currently only supported for image classification

# ### Export Image Feutures during Training

# Tensorboard [Embedding Projector](https://www.tensorflow.org/tensorboard/tensorboard_projector_plugin) is supported in `TensorBoardCallback` (set parameter `projector=True`) during training. The validation set embeddings will be written after each epoch.
#
# ```
# cbs = [TensorBoardCallback(projector=True)]
# learn = cnn_learner(dls, resnet18, metrics=accuracy)
# learn.fit_one_cycle(3, cbs=cbs)
# ```

# ### Export Image Features during Inference

# To write the embeddings for a custom dataset (e. g. after loading a learner) use `TensorBoardProjectorCallback`. Add the callback manually to the learner.
#
# ```
# learn = load_learner('path/to/export.pkl')
# learn.add_cb(TensorBoardProjectorCallback())
# dl = learn.dls.test_dl(files, with_labels=True)
# _ = learn.get_preds(dl=dl)
# ```

# If using a custom model (non fastai-resnet) pass the layer where the embeddings should be extracted as a callback-parameter.
#
# ```
# layer = learn.model[1][1]
# cbs = [TensorBoardProjectorCallback(layer=layer)]
# preds = learn.get_preds(dl=dl, cbs=cbs)
# ```

# ### Export Word Embeddings from Language Models

# To export word embeddings from Language Models (tested with AWD_LSTM (fast.ai) and GPT2 / BERT (transformers)) but works with every model that contains an embedding layer.

# For a **fast.ai TextLearner or LMLearner** just pass the learner - the embedding layer and vocab will be extracted automatically:
# ```
# dls = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid='test')
# learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
# projector_word_embeddings(learn=learn, limit=2000, start=2000)
# ```

# For other language models - like the ones in the **transformers library** - you'll have to pass the layer and vocab. Here's an example for a **BERT** model.
# ```
# from transformers import AutoTokenizer, AutoModel
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# model = AutoModel.from_pretrained("bert-base-uncased")
#
# # get the word embedding layer
# layer = model.embeddings.word_embeddings
#
# # get and sort vocab
# vocab_dict = tokenizer.get_vocab()
# vocab = [k for k, v in sorted(vocab_dict.items(), key=lambda x: x[1])]
#
# # write the embeddings for tb projector
# projector_word_embeddings(layer=layer, vocab=vocab, limit=2000, start=2000)
# ```

# export


# export
class TensorBoardBaseCallback(Callback):
    order = Recorder.order + 1
    "Base class for tensorboard callbacks"

    def __init__(self): self.run_projector = False

    def after_pred(self):
        if self.run_projector:
            self.feat = _add_projector_features(self.learn, self.h, self.feat)

    def after_validate(self):
        if not self.run_projector:
            return
        self.run_projector = False
        self._remove()
        _write_projector_embedding(self.learn, self.writer, self.feat)

    def after_fit(self):
        if self.run:
            self.writer.close()

    def _setup_projector(self):
        self.run_projector = True
        self.h = hook_output(self.learn.model[1][1] if not self.layer else self.layer)
        self.feat = {}

    def _setup_writer(self): self.writer = SummaryWriter(log_dir=self.log_dir)
    def __del__(self): self._remove()

    def _remove(self):
        if getattr(self, 'h', None):
            self.h.remove()


# export
class TensorBoardCallback(TensorBoardBaseCallback):
    "Saves model topology, losses & metrics for tensorboard and tensorboard projector during training"

    def __init__(self, log_dir=None, trace_model=True, log_preds=True, n_preds=9, projector=False, layer=None):
        super().__init__()
        store_attr()

    def before_fit(self):
        self.run = not hasattr(self.learn, 'lr_finder') and not hasattr(self, "gather_preds") and rank_distrib() == 0
        if not self.run:
            return
        self._setup_writer()
        if self.trace_model:
            if hasattr(self.learn, 'mixed_precision'):
                raise Exception("Can't trace model in mixed precision, pass `trace_model=False` or don't use FP16.")
            b = self.dls.one_batch()
            self.learn._split(b)
            self.writer.add_graph(self.model, *self.xb)

    def after_batch(self):
        self.writer.add_scalar('train_loss', self.smooth_loss, self.train_iter)
        for i, h in enumerate(self.opt.hypers):
            for k, v in h.items():
                self.writer.add_scalar(f'{k}_{i}', v, self.train_iter)

    def after_epoch(self):
        for n, v in zip(self.recorder.metric_names[2:-1], self.recorder.log[2:-1]):
            self.writer.add_scalar(n, v, self.train_iter)
        if self.log_preds:
            b = self.dls.valid.one_batch()
            self.learn.one_batch(0, b)
            preds = getattr(self.loss_func, 'activation', noop)(self.pred)
            out = getattr(self.loss_func, 'decodes', noop)(preds)
            x, y, its, outs = self.dls.valid.show_results(b, out, show=False, max_n=self.n_preds)
            tensorboard_log(x, y, its, outs, self.writer, self.train_iter)

    def before_validate(self):
        if self.projector:
            self._setup_projector()


# export
class TensorBoardProjectorCallback(TensorBoardBaseCallback):
    "Extracts and exports image featuers for tensorboard projector during inference"

    def __init__(self, log_dir=None, layer=None):
        super().__init__()
        store_attr()

    def before_fit(self):
        self.run = not hasattr(self.learn, 'lr_finder') and hasattr(self, "gather_preds") and rank_distrib() == 0
        if not self.run:
            return
        self._setup_writer()

    def before_validate(self):
        self._setup_projector()


# export
def _write_projector_embedding(learn, writer, feat):
    lbls = [learn.dl.vocab[l] for l in feat['lbl']] if getattr(learn.dl, 'vocab', None) else None
    vecs = feat['vec'].squeeze()
    writer.add_embedding(vecs, metadata=lbls, label_img=feat['img'], global_step=learn.train_iter)


# export
def _add_projector_features(learn, hook, feat):
    img = _normalize_for_projector(learn.x)
    first_epoch = True if learn.iter == 0 else False
    feat['vec'] = hook.stored if first_epoch else torch.cat((feat['vec'], hook.stored), 0)
    feat['img'] = img if first_epoch else torch.cat((feat['img'], img), 0)
    if getattr(learn.dl, 'vocab', None):
        feat['lbl'] = learn.y if first_epoch else torch.cat((feat['lbl'], learn.y), 0)
    return feat


# export
def _get_embeddings(model, layer):
    layer = model[0].encoder if layer == None else layer
    return layer.weight


# export
@typedispatch
def _normalize_for_projector(x: TensorImage):
    # normalize tensor to be between 0-1
    img = x.clone()
    sz = img.shape
    img = img.view(x.size(0), -1)
    img -= img.min(1, keepdim=True)[0]
    img /= img.max(1, keepdim=True)[0]
    img = img.view(*sz)
    return img


# export


# export
def projector_word_embeddings(learn=None, layer=None, vocab=None, limit=-1, start=0, log_dir=None):
    "Extracts and exports word embeddings from language models embedding layers"
    if not layer:
        if isinstance(learn, LMLearner):
            layer = learn.model[0].encoder
        elif isinstance(learn, TextLearner):
            layer = learn.model[0].module.encoder
    emb = layer.weight
    img = torch.full((len(emb), 3, 8, 8), 0.7)
    vocab = learn.dls.vocab[0] if vocab == None else vocab
    vocab = list(map(lambda x: f'{x}_', vocab))
    writer = SummaryWriter(log_dir=log_dir)
    end = start + limit if limit >= 0 else -1
    writer.add_embedding(emb[start:end], metadata=vocab[start:end], label_img=img[start:end])
    writer.close()


# export


# export
@typedispatch
def tensorboard_log(x: TensorImage, y: TensorCategory, samples, outs, writer, step):
    fig, axs = get_grid(len(samples), add_vert=1, return_fig=True)
    for i in range(2):
        axs = [b.show(ctx=c) for b, c in zip(samples.itemgot(i), axs)]
    axs = [r.show(ctx=c, color='green' if b == r else 'red')
           for b, r, c in zip(samples.itemgot(1), outs.itemgot(0), axs)]
    writer.add_figure('Sample results', fig, step)


# export


# export
@typedispatch
def tensorboard_log(x: TensorImage, y: (TensorImageBase, TensorPoint, TensorBBox), samples, outs, writer, step):
    fig, axs = get_grid(len(samples), add_vert=1, return_fig=True, double=True)
    for i in range(2):
        axs[::2] = [b.show(ctx=c) for b, c in zip(samples.itemgot(i), axs[::2])]
    for x in [samples, outs]:
        axs[1::2] = [b.show(ctx=c) for b, c in zip(x.itemgot(0), axs[1::2])]
    writer.add_figure('Sample results', fig, step)


# ## TensorBoardCallback


# +
path = untar_data(URLs.PETS)

db = DataBlock(blocks=(ImageBlock, CategoryBlock),
               get_items=get_image_files,
               item_tfms=Resize(128),
               splitter=RandomSubsetSplitter(train_sz=0.1, valid_sz=0.01),
               batch_tfms=aug_transforms(size=64),
               get_y=using_attr(RegexLabeller(r'(.+)_\d+.*$'), 'name'))

dls = db.dataloaders(path / 'images')
# -

learn = cnn_learner(dls, resnet18, metrics=accuracy)

learn.unfreeze()
learn.fit_one_cycle(3, cbs=TensorBoardCallback(Path.home() / 'tmp' / 'runs' / 'tb', trace_model=True))

# ## Projector

# ### Projector in TensorBoardCallback

path = untar_data(URLs.PETS)

# +
db = DataBlock(blocks=(ImageBlock, CategoryBlock),
               get_items=get_image_files,
               item_tfms=Resize(128),
               splitter=RandomSubsetSplitter(train_sz=0.05, valid_sz=0.01),
               batch_tfms=aug_transforms(size=64),
               get_y=using_attr(RegexLabeller(r'(.+)_\d+.*$'), 'name'))

dls = db.dataloaders(path / 'images')
# -

cbs = [TensorBoardCallback(log_dir=Path.home() / 'tmp' / 'runs' / 'vision1', projector=True)]
learn = cnn_learner(dls, resnet18, metrics=accuracy)

learn.unfreeze()
learn.fit_one_cycle(3, cbs=cbs)

# ### TensorBoardProjectorCallback

path = untar_data(URLs.PETS)

# +
db = DataBlock(blocks=(ImageBlock, CategoryBlock),
               get_items=get_image_files,
               item_tfms=Resize(128),
               splitter=RandomSubsetSplitter(train_sz=0.1, valid_sz=0.01),
               batch_tfms=aug_transforms(size=64),
               get_y=using_attr(RegexLabeller(r'(.+)_\d+.*$'), 'name'))

dls = db.dataloaders(path / 'images')
# -

files = get_image_files(path / 'images')
files = files[:256]

dl = learn.dls.test_dl(files, with_labels=True)

learn = cnn_learner(dls, resnet18, metrics=accuracy)
layer = learn.model[1][0].ap
cbs = [TensorBoardProjectorCallback(layer=layer, log_dir=Path.home() / 'tmp' / 'runs' / 'vision2')]

_ = learn.get_preds(dl=dl, cbs=cbs)

# ## projector_word_embeddings

# ### fastai text or lm learner


dls = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid='test')
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)

projector_word_embeddings(learn, limit=1000, log_dir=Path.home() / 'tmp' / 'runs' / 'text')

# ### transformers

# #### GPT2

# +

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
layer = model.transformer.wte
vocab_dict = tokenizer.get_vocab()
vocab = [k for k, v in sorted(vocab_dict.items(), key=lambda x: x[1])]

projector_word_embeddings(layer=layer, vocab=vocab, limit=2000, log_dir=Path.home() / 'tmp' / 'runs' / 'transformers')
# -

# #### BERT

# +

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

layer = model.embeddings.word_embeddings

vocab_dict = tokenizer.get_vocab()
vocab = [k for k, v in sorted(vocab_dict.items(), key=lambda x: x[1])]

projector_word_embeddings(layer=layer, vocab=vocab, limit=2000, start=2000, log_dir=Path.home() / 'tmp' / 'runs' / 'transformers')
# -

# ### Validate results in tensorboard

# Run the following command in the command line to check if the projector embeddings have been correctly wirtten:
#
# ```
# tensorboard --logdir=~/tmp/runs
# ```
#
# Open http://localhost:6006 in browser (TensorBoard Projector doesn't work correctly in Safari!)

# ## Export -

# hide
notebook2script()
