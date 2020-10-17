# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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
from fastai.text.all import *
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# # Tutorial - Transformers
#
# > An example of how to incorporate the transfomers library from HuggingFace with fastai

# +
# all_slow
# -

# In this tutorial, we will see how we can use the fastai library to fine-tune a pretrained transformer model from the [transformers library](https://github.com/huggingface/transformers) by HuggingFace. We will use the mid-level API to gather the data. Even if this tutorial is self contained, it might help to check the [imagenette tutorial](http://docs.fast.ai/tutorial.imagenette) to have a second look on the mid-level API (with a gentle introduction using the higher level APIs) in computer vision.

# ## Importing a transformers pretrained model

# First things first, we will need to install the transformers library. If you haven't done it yet, install the library:
#
# ```
# ```

# Then let's import what will need: we will fine-tune the GPT2 pretrained model and fine-tune on wikitext-2 here. For this, we need the `GPT2LMHeadModel` (since we want a language model) and the `GPT2Tokenizer` to prepare the data.


# We can use several versions of this GPT2 model, look at the [transformers documentation](https://huggingface.co/transformers/pretrained_models.html) for more details. Here we will use the basic version (that already takes a lot of space in memory!) You can change the model used by changing the content of `pretrained_weights` (if it's not a GPT2 model, you'll need to change the classes used for the model and the tokenizer of course).

pretrained_weights = 'gpt2'
tokenizer = GPT2TokenizerFast.from_pretrained(pretrained_weights)
model = GPT2LMHeadModel.from_pretrained(pretrained_weights)

# Before we move on to the fine-tuning part, let's have a look at this `tokenizer` and this `model`. The tokenizers in HuggingFace usually do the tokenization and the numericalization in one step (we ignore the padding warning for now):

ids = tokenizer.encode('This is an example of text, and')
ids

# Like fastai `Transform`s, the tokenizer has a `decode` method to give you back a text from ids:

tokenizer.decode(ids)

# The model can be used to generate predictions (it is pretrained). It has a `generate` method that expects a batch of prompt, so we feed it our ids and add one batch dimension (there is a padding warning we can ignore as well):


t = torch.LongTensor(ids)[None]
preds = model.generate(t)

# The predictions, by default, are of length 20:

preds.shape, preds[0]

# We can use the decode method (that prefers a numpy array to a tensor):

tokenizer.decode(preds[0].numpy())

# ## Bridging the gap with fastai

# Now let's see how we can use fastai to fine-tune this model on wikitext-2, using all the training utilities (learning rate finder, 1cycle policy etc...). First, we import all the text utilities:


# ### Preparing the data

# Then we download the dataset (if not present), it comes as two csv files:

path = untar_data(URLs.WIKITEXT_TINY)
path.ls()

# Let's have a look at what those csv files look like:

df_train = pd.read_csv(path / 'train.csv', header=None)
df_valid = pd.read_csv(path / 'test.csv', header=None)
df_train.head()

# We gather all texts in one numpy array (since it will be easier to use this way with fastai):

all_texts = np.concatenate([df_train[0].values, df_valid[0].values])


# To process this data to train a model, we need to build a `Transform` that will be applied lazily. In this case we could do the pre-processing once and for all and only use the transform for decoding (we will see how just after), but the fast tokenizer from HuggingFace is, as its name indicates, fast, so it doesn't really impact performance to do it this way.
#
# In a fastai `Transform` you can define:
# - an <code>encodes</code> method that is applied when you call the transform (a bit like the `forward` method in a `nn.Module`)
# - a <code>decodes</code> method that is applied when you call the `decode` method of the transform, if you need to decode anything for showing purposes (like converting ids to a text here)
# - a <code>setups</code> method that sets some inner state of the `Transform` (not needed here so we skip it)

class TransformersTokenizer(Transform):
    def __init__(self, tokenizer): self.tokenizer = tokenizer
    def encodes(self, x):
        toks = self.tokenizer.tokenize(x)
        return tensor(self.tokenizer.convert_tokens_to_ids(toks))
    def decodes(self, x): return TitledStr(self.tokenizer.decode(x.cpu().numpy()))


# Two comments on the code above:
# - in <code>encodes</code> we don't use the `tokenizer.encode` method since it does some additional preprocessing for the model after tokenizing and numericalizing (the part throwing a warning before). Here we don't need any post-processing so it's fine to skip it.
# - in <code>decodes</code> we return a `TitledStr` object and not just a plain string. That's a fastai class that adds a `show` method to the string, which will allow us to use all the fastai show methods.

# You can then group your data with this `Transform` using a `TfmdLists`. It has an s in its name because it contains the training and validation set. We indicate the indices of the training set and the validation set with `splits` (here all the first indices until `len(df_train)` and then all the remaining indices):

splits = [range_of(df_train), list(range(len(df_train), len(all_texts)))]
tls = TfmdLists(all_texts, TransformersTokenizer(tokenizer), splits=splits, dl_type=LMDataLoader)

# We specify `dl_type=LMDataLoader` for when we will convert this `TfmdLists` to `DataLoaders`: we will use an `LMDataLoader` since we have a language modeling problem, not the usual fastai `TfmdDL`.
#
# In a `TfmdLists` you can access the elements of the training or validation set quite easily:

tls.train[0], tls.valid[0]

# They look the same but only because they begin and end the same way. We can see the shapes are different:

tls.tfms(tls.train.items[0]).shape, tls.tfms(tls.valid.items[0]).shape

# And we can have a look at both decodes using `show_at`:

# +
# show_at(tls.train, 0)

# +
# show_at(tls.valid, 0)
# -

# The fastai library expects the data to be assembled in a `DataLoaders` object (something that has a training and validation dataloader). We can get one by using the `dataloaders` method. We just have to specify a batch size and a sequence length. Since the GPT2 model was trained with sequences of size 1024, we use this sequence length (it's a stateless model, so it will change the perplexity if we use less):

bs, sl = 8, 1024
dls = tls.dataloaders(bs=bs, seq_len=sl)

# Note that you may have to reduce the batch size depending on your GPU RAM.

# In fastai, as soon as we have a `DataLoaders`, we can use `show_batch` to have a look at the data (here texts for inputs, and the same text shifted by one token to the right for validation):

dls.show_batch(max_n=2)


# Another way to gather the data is to preprocess the texts once and for all and only use the transform to decode the tensors to texts:

# +
def tokenize(text):
    toks = tokenizer.tokenize(text)
    return tensor(tokenizer.convert_tokens_to_ids(toks))

tokenized = [tokenize(t) for t in progress_bar(all_texts)]


# -

# Now we change the previous `Tokenizer` like this:

class TransformersTokenizer(Transform):
    def __init__(self, tokenizer): self.tokenizer = tokenizer
    def encodes(self, x):
        return x if isinstance(x, Tensor) else tokenize(x)

    def decodes(self, x): return TitledStr(self.tokenizer.decode(x.cpu().numpy()))


# In the <code>encodes</code> method, we still account for the case where we get something that's not already tokenized, just in case we were to build a dataset with new texts using this transform.

tls = TfmdLists(tokenized, TransformersTokenizer(tokenizer), splits=splits, dl_type=LMDataLoader)
dls = tls.dataloaders(bs=bs, seq_len=sl)

# And we can check it still works properly for showing purposes:

dls.show_batch(max_n=2)


# ### Fine-tuning the model

# The HuggingFace model will return a tuple in outputs, with the actual predictions and some additional activations (should we want to use them in some regularization scheme). To work inside the fastai training loop, we will need to drop those using a `Callback`: we use those to alter the behavior of the training loop.
#
# Here we need to write the event `after_pred` and replace `self.learn.pred` (which contains the predictions that will be passed to the loss function) by just its first element. In callbacks, there is a shortcut that lets you access any of the underlying `Learner` attributes so we can write `self.pred[0]` instead of `self.learn.pred[0]`. That shortcut only works for read access, not write, so we have to write `self.learn.pred` on the right side (otherwise we would set a `pred` attribute in the `Callback`).

class DropOutput(Callback):
    def after_pred(self): self.learn.pred = self.pred[0]


# Of course we could make this a bit more complex and add some penalty to the loss using the other part of the tuple of predictions, like the `RNNRegularizer`.
#
# Now, we are ready to create our `Learner`, which is a fastai object grouping data, model and loss function and handles model training or inference. Since we are in a language model setting, we pass perplexity as a metric, and we need to use the callback we just defined. Lastly, we use mixed precision to save every bit of memory we can (and if you have a modern GPU, it will also make training faster):

learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), cbs=[DropOutput], metrics=Perplexity()).to_fp16()

# We can check how good the model is without any fine-tuning step (spoiler alert, it's pretty good!)

learn.validate()

# This lists the validation loss and metrics (so 26.6 as perplexity is kind of amazing).
#
# Now that we have a `Learner` we can use all the fastai training loop capabilities: learning rate finder, training with 1cycle etc...

learn.lr_find()

# The learning rate finder curve suggests picking something between 1e-4 and 1e-3.

learn.fit_one_cycle(1, 1e-4)

# Now with just one epoch of fine-tuning and not much regularization, our model did not really improve since it was already amazing. To have a look at some generated texts, let's take a prompt that looks like a wikipedia article:

df_valid.head(1)

# Article seems to begin with new line and the title between = signs, so we will mimic that:

prompt = "\n = Unicorn = \n \n A unicorn is a magical creature with a rainbow tail and a horn"

# The prompt needs to be tokenized and numericalized, so we use the same function as before to do this, before we use the `generate` method of the model.

prompt_ids = tokenizer.encode(prompt)
inp = tensor(prompt_ids)[None].cuda()
inp.shape

preds = learn.model.generate(inp, max_length=40, num_beams=5, temperature=1.5)

tokenizer.decode(preds[0])
