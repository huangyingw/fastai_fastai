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

from spacy.lang.en import English
from fastai.text.all import *

# +
# chunked??
# -

# Let's look at how long it takes to tokenize a sample of 1000 IMDB review.

path = untar_data(URLs.IMDB_SAMPLE)
df = pd.read_csv(path / 'texts.csv')
df.head(2)

ss = L(list(df.text))
ss[0]


# We'll start with the simplest approach:

def delim_tok(s, delim=' '): return L(s.split(delim))


s = ss[0]
delim_tok(s)


# ...and a general way to tokenize a bunch of strings:

def apply(func, items): return list(map(func, items))


# Let's time it:

# %%timeit -n 2 -r 3
global t
t = apply(delim_tok, ss)

# ...and the same thing with 2 workers:

# %%timeit -n 2 -r 3
parallel(delim_tok, ss, n_workers=2, progress=False)

# How about if we put half the work in each worker?

batches32 = [L(list(o)).map(str) for o in np.array_split(ss, 32)]
batches8 = [L(list(o)).map(str) for o in np.array_split(ss, 8)]
batches = [L(list(o)).map(str) for o in np.array_split(ss, 2)]

# %%timeit -n 2 -r 3
parallel(partial(apply, delim_tok), batches, progress=False, n_workers=2)

# So there's a lot of overhead in using parallel processing in Python. :(
#
# Let's see why. What if we do nothing interesting in our function?

# %%timeit -n 2 -r 3
global t
t = parallel(noop, batches, progress=False, n_workers=2)


# That's quite fast! (Although still slower than single process.)
#
# What if we don't return much data?

def f(x): return 1


# %%timeit -n 2 -r 3
global t
t = parallel(f, batches, progress=False, n_workers=2)


# That's a bit faster still.
#
# What if we don't actually return the lists of tokens, but create them still?

def f(items):
    o = [s.split(' ') for s in items]
    return [s for s in items]


# So creating the tokens, isn't taking the time, but returning them over the process boundary is.

# %%timeit -n 2 -r 3
global t
t = parallel(f, batches, progress=False, n_workers=2)

# Is numpy any faster?

sarr = np.array(ss)

# %%timeit -n 2 -r 3
global t
t = np.char.split(sarr)

# ## Spacy

# +


def conv_sp(doc): return L(doc).map(str)


class SpTok:
    def __init__(self):
        nlp = English()
        self.tok = nlp.Defaults.create_tokenizer(nlp)

    def __call__(self, x): return L(self.tok(str(x))).map(conv_sp)


# -

# Let's see how long it takes to create a tokenizer in Spacy:

# %%timeit -n 2 -r 3
SpTok()

nlp = English()
sp_tokenizer = nlp.Defaults.create_tokenizer(nlp)


def spacy_tok(s): return L(sp_tokenizer(str(s))).map(str)


# Time tokenize in Spacy using a loop:

# %%timeit -r 3
global t
t = apply(spacy_tok, ss)

# ...and the same thing in parallel:

# %%timeit -r 3
global t
t = parallel(partial(apply, spacy_tok), batches, progress=False, n_workers=2)

# ...and with more workers:

# %%timeit -r 3
global t
t = parallel(partial(apply, spacy_tok), batches8, progress=False, n_workers=8)


# ...and with creating the tokenizer in the child process:

def f(its):
    tok = SpTok()
    return [[str(o) for o in tok(p)] for p in its]


# %%timeit -r 3
global t
t = parallel(f, batches8, progress=False, n_workers=8)

# Let's try `pipe`

# %%timeit -r 3
global t
t = L(nlp.tokenizer.pipe(ss)).map(conv_sp)


def f(its): return L(nlp.tokenizer.pipe(its)).map(conv_sp)


# %%timeit -r 3
global t
t = parallel(f, batches8, progress=False, n_workers=8)

test_eq(chunked(range(12), n_chunks=4), [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
test_eq(chunked(range(11), n_chunks=4), [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10]])
test_eq(chunked(range(10), n_chunks=4), [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]])
test_eq(chunked(range(9), n_chunks=3), [[0, 1, 2], [3, 4, 5], [6, 7, 8]])

# %%timeit -r 3
global t
t = parallel_chunks(f, ss, n_workers=8, progress=False)


def array_split(arr, n): return chunked(arr, math.floor(len(arr) / n))
