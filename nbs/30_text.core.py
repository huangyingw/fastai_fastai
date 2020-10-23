# -*- coding: utf-8 -*-
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
import html
import spacy
from nbdev.export import notebook2script
from spacy.symbols import ORTH
from nbdev.showdoc import *
from fastai.data.all import *
from fastai.torch_basics import *
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# export

# hide

# +
# default_exp text.core
# default_cls_lvl 3
# -

# # Text core
#
# > Basic function to preprocess text before assembling it in a `DataLoaders`.

# export

# ## Preprocessing rules

# The following are rules applied to texts before or after it's tokenized.

# export
# special tokens
UNK, PAD, BOS, EOS, FLD, TK_REP, TK_WREP, TK_UP, TK_MAJ = "xxunk xxpad xxbos xxeos xxfld xxrep xxwrep xxup xxmaj".split()

# export
_all_ = ["UNK", "PAD", "BOS", "EOS", "FLD", "TK_REP", "TK_WREP", "TK_UP", "TK_MAJ"]

# +
# export
_re_spec = re.compile(r'([/#\\])')


def spec_add_spaces(t):
    "Add spaces around / and #"
    return _re_spec.sub(r' \1 ', t)


# -

test_eq(spec_add_spaces('#fastai'), ' # fastai')
test_eq(spec_add_spaces('/fastai'), ' / fastai')
test_eq(spec_add_spaces('\\fastai'), ' \\ fastai')

# +
# export
_re_space = re.compile(' {2,}')


def rm_useless_spaces(t):
    "Remove multiple spaces"
    return _re_space.sub(' ', t)


# -

test_eq(rm_useless_spaces('a  b   c'), 'a b c')

# +
# export
_re_rep = re.compile(r'(\S)(\1{2,})')


def replace_rep(t):
    "Replace repetitions at the character level: cccc -- TK_REP 4 c"
    def _replace_rep(m):
        c, cc = m.groups()
        return f' {TK_REP} {len(cc)+1} {c} '
    return _re_rep.sub(_replace_rep, t)


# -

# It starts replacing at 3 repetitions of the same character or more.

test_eq(replace_rep('aa'), 'aa')
test_eq(replace_rep('aaaa'), f' {TK_REP} 4 a ')

# export
_re_wrep = re.compile(r'(?:\s|^)(\w+)\s+((?:\1\s+)+)\1(\s|\W|$)')

# hide
"""
Matches any word repeated at least four times with spaces between them
(?:\s|^)       Non-Capture either a whitespace character or the beginning of text
(\w+)          Capture any alphanumeric character
\s+            One or more whitespace
((?:\1\s+)+)   Capture a repetition of one or more times \1 followed by one or more whitespace
\1             Occurrence of \1
(\s|\W|$)      Capture last whitespace, non alphanumeric character or end of text
"""


# export
def replace_wrep(t):
    "Replace word repetitions: word word word word -- TK_WREP 4 word"
    def _replace_wrep(m):
        c, cc, e = m.groups()
        return f' {TK_WREP} {len(cc.split())+2} {c} {e}'
    return _re_wrep.sub(_replace_wrep, t)


# It starts replacing at 3 repetitions of the same word or more.

test_eq(replace_wrep('ah ah'), 'ah ah')
test_eq(replace_wrep('ah ah ah'), f' {TK_WREP} 3 ah ')
test_eq(replace_wrep('ah ah   ah  ah'), f' {TK_WREP} 4 ah ')
test_eq(replace_wrep('ah ah ah ah '), f' {TK_WREP} 4 ah  ')
test_eq(replace_wrep('ah ah ah ah.'), f' {TK_WREP} 4 ah .')
test_eq(replace_wrep('ah ah ahi'), f'ah ah ahi')


# export
def fix_html(x):
    "Various messy things we've seen in documents"
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace('nbsp;', ' ').replace(
        '#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace('<br />', "\n").replace(
        '\\"', '"').replace('<unk>', UNK).replace(' @.@ ', '.').replace(' @-@ ', '-').replace('...', ' …')
    return html.unescape(x)


test_eq(fix_html('#39;bli#146;'), "'bli'")
test_eq(fix_html('Sarah amp; Duck...'), 'Sarah & Duck …')
test_eq(fix_html('a nbsp; #36;'), 'a   $')
test_eq(fix_html('\\" <unk>'), f'" {UNK}')
test_eq(fix_html('quot;  @.@  @-@ '), "' .-")
test_eq(fix_html('<br />text\\n'), '\ntext\n')

# export
_re_all_caps = re.compile(r'(\s|^)([A-Z]+[^a-z\s]*)(?=(\s|$))')

# hide
"""
Catches any word in all caps, even with ' or - inside
(\s|^)        Capture either a whitespace or the beginning of text
([A-Z]+       Capture one capitalized letter or more...
[^a-z\s]*)    ...followed by anything that's non lowercase or whitespace
(?=(\s|$))    Look ahead for a space or end of text
"""


# export
def replace_all_caps(t):
    "Replace tokens in ALL CAPS by their lower version and add `TK_UP` before."
    def _replace_all_caps(m):
        tok = f'{TK_UP} ' if len(m.groups()[1]) > 1 else ''
        return f"{m.groups()[0]}{tok}{m.groups()[1].lower()}"
    return _re_all_caps.sub(_replace_all_caps, t)


test_eq(replace_all_caps("I'M SHOUTING"), f"{TK_UP} i'm {TK_UP} shouting")
test_eq(replace_all_caps("I'm speaking normally"), "I'm speaking normally")
test_eq(replace_all_caps("I am speaking normally"), "i am speaking normally")

# export
_re_maj = re.compile(r'(\s|^)([A-Z][^A-Z\s]*)(?=(\s|$))')

# hide
"""
Catches any capitalized word
(\s|^)       Capture either a whitespace or the beginning of text
([A-Z]       Capture exactly one capitalized letter...
[^A-Z\s]*)   ...followed by anything that's not uppercase or whitespace
(?=(\s|$))   Look ahead for a space of end of text
"""


# export
def replace_maj(t):
    "Replace tokens in Sentence Case by their lower version and add `TK_MAJ` before."
    def _replace_maj(m):
        tok = f'{TK_MAJ} ' if len(m.groups()[1]) > 1 else ''
        return f"{m.groups()[0]}{tok}{m.groups()[1].lower()}"
    return _re_maj.sub(_replace_maj, t)


test_eq(replace_maj("Jeremy Howard"), f'{TK_MAJ} jeremy {TK_MAJ} howard')
test_eq(replace_maj("I don't think there is any maj here"), ("i don't think there is any maj here"),)


# export
def lowercase(t, add_bos=True, add_eos=False):
    "Converts `t` to lowercase"
    return (f'{BOS} ' if add_bos else '') + t.lower().strip() + (f' {EOS}' if add_eos else '')


# export
def replace_space(t):
    "Replace embedded spaces in a token with unicode line char to allow for split/join"
    return t.replace(' ', '▁')


# export
defaults.text_spec_tok = [UNK, PAD, BOS, EOS, FLD, TK_REP, TK_WREP, TK_UP, TK_MAJ]
defaults.text_proc_rules = [fix_html, replace_rep, replace_wrep, spec_add_spaces, rm_useless_spaces,
                            replace_all_caps, replace_maj, lowercase]
defaults.text_postproc_rules = [replace_space]


# ## Tokenizing

# A tokenizer is a class that must implement `__call__`. This method receives a iterator of texts and must return a generator with their tokenized versions. Here is the most basic example:

# export
class BaseTokenizer():
    "Basic tokenizer that just splits on spaces"

    def __init__(self, split_char=' ', **kwargs): self.split_char = split_char
    def __call__(self, items): return (t.split(self.split_char) for t in items)


tok = BaseTokenizer()
test_eq(tok(["This is a text"]), [["This", "is", "a", "text"]])
tok = BaseTokenizer('x')
test_eq(tok(["This is a text"]), [["This is a te", "t"]])


# export
class SpacyTokenizer():
    "Spacy tokenizer for `lang`"

    def __init__(self, lang='en', special_toks=None, buf_sz=5000):
        self.special_toks = ifnone(special_toks, defaults.text_spec_tok)
        nlp = spacy.blank(lang, disable=["parser", "tagger", "ner"])
        for w in self.special_toks:
            nlp.tokenizer.add_special_case(w, [{ORTH: w}])
        self.pipe, self.buf_sz = nlp.pipe, buf_sz

    def __call__(self, items):
        return (L(doc).attrgot('text') for doc in self.pipe(map(str, items), batch_size=self.buf_sz))


# export
WordTokenizer = SpacyTokenizer

tok = SpacyTokenizer()
inp, exp = "This isn't the easiest text.", ["This", "is", "n't", "the", "easiest", "text", "."]
test_eq(L(tok([inp, inp])), [exp, exp])


# export
class TokenizeWithRules:
    "A wrapper around `tok` which applies `rules`, then tokenizes, then applies `post_rules`"

    def __init__(self, tok, rules=None, post_rules=None):
        self.rules = L(ifnone(rules, defaults.text_proc_rules))
        self.post_f = compose(*L(ifnone(post_rules, defaults.text_postproc_rules)))
        self.tok = tok

    def __call__(self, batch):
        return (L(o).map(self.post_f) for o in self.tok(maps(*self.rules, batch)))


f = TokenizeWithRules(BaseTokenizer(), rules=[replace_all_caps])
test_eq(f(["THIS isn't a problem"]), [[TK_UP, 'this', "isn't", 'a', 'problem']])
f = TokenizeWithRules(SpacyTokenizer())
test_eq(f(["This isn't a problem"]), [[BOS, TK_MAJ, 'this', 'is', "n't", 'a', 'problem']])
f = TokenizeWithRules(BaseTokenizer(split_char="'"), rules=[])
test_eq(f(["This isn't a problem"]), [['This▁isn', 't▁a▁problem']])

# The main function that will be called during one of the processes handling tokenization. It will iterate through the `batch` of texts, apply them `rules` and tokenize them.

texts = ["this is a text", "this is another text"]
tok = TokenizeWithRules(BaseTokenizer(), texts.__getitem__)
test_eq(tok([0, 1]), [['this', 'is', 'a', 'text'], ['this', 'is', 'another', 'text']])


# export
@delegates(TokenizeWithRules)
def tokenize1(text, tok, **kwargs):
    "Call `TokenizeWithRules` with a single text"
    return first(TokenizeWithRules(tok=tok, **kwargs)([text]))


test_eq(tokenize1("This isn't a problem", SpacyTokenizer()),
        [BOS, TK_MAJ, 'this', 'is', "n't", 'a', 'problem'])
test_eq(tokenize1("This isn't a problem", tok=BaseTokenizer(), rules=[]),
        ['This', "isn't", 'a', 'problem'])


# export
def parallel_tokenize(items, tok=None, rules=None, n_workers=defaults.cpus, **kwargs):
    "Calls optional `setup` on `tok` before launching `TokenizeWithRules` using `parallel_gen"
    if tok is None:
        tok = WordTokenizer()
    if hasattr(tok, 'setup'):
        tok.setup(items, rules)
    return parallel_gen(TokenizeWithRules, items, tok=tok, rules=rules, n_workers=n_workers, **kwargs)


# Note that since this uses `parallel_gen` behind the scenes, the generator returned contains tuples of indices and results. There is no guarantee that the results are returned in order, so you should sort by the first item of the tuples (the indices) if you need them ordered.

res = parallel_tokenize(['0 1', '1 2'], rules=[], n_workers=2)
idxs, toks = zip(*L(res).sorted(itemgetter(0)))
test_eq(toks, [['0', '1'], ['1', '2']])

# hide
res1 = parallel_tokenize(['0 1', '1 2'], tok=BaseTokenizer(), rules=[], n_workers=0)
idxs1, toks1 = zip(*L(res1).sorted(itemgetter(0)))
test_eq(toks, toks1)

# ### Tokenize texts in files

# Preprocessing function for texts in filenames. Tokenized texts will be saved in a similar fashion in a directory suffixed with `_tok` in the parent folder of `path` (override with `output_dir`). This directory is the return value.

# export
fn_counter_pkl = 'counter.pkl'
fn_lengths_pkl = 'lengths.pkl'


# export
def _tokenize_files(func, files, path, output_dir=None, output_names=None, n_workers=defaults.cpus, rules=None, tok=None,
                    encoding='utf8', skip_if_exists=False):
    "Tokenize text `files` in parallel using `n_workers`"
    if tok is None:
        tok = WordTokenizer()
    output_dir = Path(ifnone(output_dir, path.parent / f'{path.name}_tok'))
    if skip_if_exists and output_dir.exists():
        return output_dir
    output_dir.mkdir(exist_ok=True)
    if output_names is None:
        output_names = L(output_dir / f.relative_to(path) for f in files)
    rules = partial(Path.read_text, encoding=encoding) + L(ifnone(rules, defaults.text_proc_rules.copy()))

    lengths, counter = {}, Counter()
    for i, tok in parallel_tokenize(files, tok, rules, n_workers=n_workers):
        out = func(i, output_dir)
        out.mk_write(' '.join(tok))
        lengths[str(files[i].relative_to(path))] = len(tok)
        counter.update(tok)

    save_pickle(output_dir / fn_lengths_pkl, lengths)
    save_pickle(output_dir / fn_counter_pkl, counter)
    return output_dir


# export
@delegates(_tokenize_files)
def tokenize_folder(path, extensions=None, folders=None, output_dir=None, skip_if_exists=True, **kwargs):
    "Tokenize text files in `path` in parallel using `n_workers`"
    path, extensions = Path(path), ifnone(extensions, ['.txt'])
    files = get_files(path, extensions=extensions, recurse=True, folders=folders)
    def _f(i, output_dir): return output_dir / files[i].relative_to(path)
    return _tokenize_files(_f, files, path, skip_if_exists=skip_if_exists, **kwargs)


# The result will be in `output_dir` (defaults to a folder in the same parent directory as `path`, with `_tok` added to `path.name`) with the same structure as in `path`. Tokenized texts for a given file will be in the file having the same name in `output_dir`. Additionally, a file with a .len suffix contains the number of tokens and the count of all words is stored in `output_dir/counter.pkl`.
#
# `extensions` will default to `['.txt']` and all text files in `path` are treated unless you specify a list of folders in `include`. `rules` (that defaults to `defaults.text_proc_rules`) are applied to each text before going in the tokenizer.

# export
@delegates(_tokenize_files)
def tokenize_files(files, path, output_dir, output_names=None, **kwargs):
    "Tokenize text `files` in parallel using `n_workers`"
    if output_names is None:
        output_names = L(output_dir / f.relative_to(path) for f in files)

    def _f(i, output_dir): return output_dir / output_names[i]
    return _tokenize_files(_f, files, path, output_dir=output_dir, **kwargs)


# ### Tokenize texts in a dataframe

# export
def _join_texts(df, mark_fields=False):
    "Join texts in row `idx` of `df`, marking each field with `FLD` if `mark_fields=True`"
    text_col = (f'{FLD} {1} ' if mark_fields else '') + df.iloc[:, 0].astype(str)
    for i in range(1, len(df.columns)):
        text_col += (f' {FLD} {i+1} ' if mark_fields else ' ') + df.iloc[:, i].astype(str)
    return text_col.values


# +
# hide
texts = [f"This is an example of text {i}" for i in range(10)]
df = pd.DataFrame({'text': texts, 'text1': texts}, columns=['text', 'text1'])
col = _join_texts(df, mark_fields=True)

for i in range(len(df)):
    test_eq(col[i], f'{FLD} 1 This is an example of text {i} {FLD} 2 This is an example of text {i}')


# -

# export
def tokenize_texts(texts, n_workers=defaults.cpus, rules=None, tok=None):
    "Tokenize `texts` in parallel using `n_workers`"
    rules = L(ifnone(rules, defaults.text_proc_rules.copy()))
    outputs = L(parallel_tokenize(texts, tok=tok, rules=rules, n_workers=n_workers)
                ).sorted().itemgot(1)
    return outputs


# export
def tokenize_df(df, text_cols, n_workers=defaults.cpus, rules=None, mark_fields=None,
                tok=None, res_col_name="text"):
    "Tokenize texts in `df[text_cols]` in parallel using `n_workers`"
    text_cols = [df.columns[c] if isinstance(c, int) else c for c in L(text_cols)]
    # mark_fields defaults to False if there is one column of texts, True if there are multiple
    if mark_fields is None:
        mark_fields = len(text_cols) > 1
    rules = L(ifnone(rules, defaults.text_proc_rules.copy()))
    texts = _join_texts(df[text_cols], mark_fields=mark_fields)
    outputs = L(parallel_tokenize(texts, tok, rules, n_workers=n_workers)
                ).sorted().itemgot(1)

    other_cols = df.columns[~df.columns.isin(text_cols)]
    res = df[other_cols].copy()
    res[res_col_name] = outputs
    res[f'{res_col_name}_length'] = [len(o) for o in outputs]
    return res, Counter(outputs.concat())


# This function returns a new dataframe with the same non-text columns, a column named text that contains the tokenized texts and a column named text_lengths that contains their respective length. It also returns a counter of all seen words to quickly build a vocabulary afterward.
#
# `rules` (that defaults to `defaults.text_proc_rules`) are applied to each text before going in the tokenizer. If `mark_fields` isn't specified, it defaults to `False` when there is a single text column, `True` when there are several. In that case, the texts in each of those columns are joined with `FLD` markers followed by the number of the field.

# export
def tokenize_csv(fname, text_cols, outname=None, n_workers=4, rules=None, mark_fields=None,
                 tok=None, header='infer', chunksize=50000):
    "Tokenize texts in the `text_cols` of the csv `fname` in parallel using `n_workers`"
    df = pd.read_csv(fname, header=header, chunksize=chunksize)
    outname = Path(ifnone(outname, fname.parent / f'{fname.stem}_tok.csv'))
    cnt = Counter()

    for i, dfp in enumerate(df):
        out, c = tokenize_df(dfp, text_cols, n_workers=n_workers, rules=rules,
                             mark_fields=mark_fields, tok=tok)
        out.text = out.text.str.join(' ')
        out.to_csv(outname, header=(None, header)[i == 0], index=False, mode=('a', 'w')[i == 0])
        cnt.update(c)

    save_pickle(outname.with_suffix('.pkl'), cnt)


# export
def load_tokenized_csv(fname):
    "Utility function to quickly load a tokenized csv ans the corresponding counter"
    fname = Path(fname)
    out = pd.read_csv(fname)
    for txt_col in out.columns[1:-1]:
        out[txt_col] = out[txt_col].str.split(' ')
    return out, load_pickle(fname.with_suffix('.pkl'))


# The result will be written in a new csv file in `outname` (defaults to the same as `fname` with the suffix `_tok.csv`) and will have the same header as the original file, the same non-text columns, a text and a text_lengths column as described in `tokenize_df`.
#
# `rules` (that defaults to `defaults.text_proc_rules`) are applied to each text before going in the tokenizer. If `mark_fields` isn't specified, it defaults to `False` when there is a single text column, `True` when there are several. In that case, the texts in each of those columns are joined with `FLD` markers followed by the number of the field.
#
# The csv file is opened with `header` and optionally with blocks of `chunksize` at a time. If this argument is passed, each chunk is processed independently and saved in the output file to save memory usage.

def _prepare_texts(tmp_d):
    "Prepare texts in a folder struct in tmp_d, a csv file and returns a dataframe"
    path = Path(tmp_d) / 'tmp'
    path.mkdir()
    for d in ['a', 'b', 'c']:
        (path / d).mkdir()
        for i in range(5):
            with open(path / d / f'text{i}.txt', 'w') as f:
                f.write(f"This is an example of text {d} {i}")

    texts = [f"This is an example of text {d} {i}" for i in range(5) for d in ['a', 'b', 'c']]
    df = pd.DataFrame({'text': texts, 'label': list(range(15))}, columns=['text', 'label'])
    csv_fname = tmp_d / 'input.csv'
    df.to_csv(csv_fname, index=False)
    return path, df, csv_fname


# hide
# integration test
with tempfile.TemporaryDirectory() as tmp_d:
    path, df, csv_fname = _prepare_texts(Path(tmp_d))
    # Tokenize as folders
    tokenize_folder(path)
    outp = Path(tmp_d) / 'tmp_tok'
    for d in ['a', 'b', 'c']:
        p = outp / d
        for i in range(5):
            test_eq((p / f'text{i}.txt').read_text(), ' '.join([
                BOS, TK_MAJ, 'this', 'is', 'an', 'example', 'of', 'text', d, str(i)]))
    cnt_a = load_pickle(outp / fn_counter_pkl)
    test_eq(cnt_a['this'], 15)
    test_eq(cnt_a['a'], 5)
    test_eq(cnt_a['0'], 3)

    # Tokenize as files
    files = get_text_files(path)
    tokenize_files(files, path, output_dir=path / 'd')
    for f in files:
        test_eq((path / 'd' / f.relative_to(path)).read_text(), ' '.join([
                BOS, TK_MAJ, 'this', 'is', 'an', 'example', 'of', 'text', f.parent.name, f.name[4]]))

    # Tokenize as individual texts
    out = tokenize_texts(df['text'].values)
    test_eq(out, [(outp / d / f'text{i}.txt').read_text().split(' ') for i in range(5) for d in ['a', 'b', 'c']])

    # Tokenize as a dataframe
    out, cnt_b = tokenize_df(df, text_cols='text')
    test_eq(list(out.columns), ['label', 'text', 'text_length'])
    test_eq(out['label'].values, df['label'].values)
    test_eq(out['text'], [(outp / d / f'text{i}.txt').read_text().split(' ') for i in range(5) for d in ['a', 'b', 'c']])
    test_eq(cnt_a, cnt_b)

    # Tokenize as a csv
    out_fname = Path(tmp_d) / 'output.csv'
    tokenize_csv(csv_fname, text_cols='text', outname=out_fname)
    test_eq((out, cnt_b), load_tokenized_csv(out_fname))


# ## `Tokenizer`-

# export
class Tokenizer(Transform):
    "Provides a consistent `Transform` interface to tokenizers operating on `DataFrame`s and folders"
    input_types = (str, list, L, tuple, Path)

    def __init__(self, tok, rules=None, counter=None, lengths=None, mode=None, sep=' '):
        if isinstance(tok, type):
            tok = tok()
        store_attr('tok,counter,lengths,mode,sep')
        self.rules = defaults.text_proc_rules if rules is None else rules

    @classmethod
    @delegates(tokenize_df, keep=True)
    def from_df(cls, text_cols, tok=None, rules=None, sep=' ', **kwargs):
        if tok is None:
            tok = WordTokenizer()
        res = cls(tok, rules=rules, mode='df')
        res.kwargs, res.train_setup = merge({'tok': tok}, kwargs), False
        res.text_cols, res.sep = text_cols, sep
        return res

    @classmethod
    @delegates(tokenize_folder, keep=True)
    def from_folder(cls, path, tok=None, rules=None, **kwargs):
        path = Path(path)
        if tok is None:
            tok = WordTokenizer()
        output_dir = tokenize_folder(path, tok=tok, rules=rules, **kwargs)
        res = cls(tok, counter=load_pickle(output_dir / fn_counter_pkl),
                  lengths=load_pickle(output_dir / fn_lengths_pkl), rules=rules, mode='folder')
        res.path, res.output_dir = path, output_dir
        return res

    def setups(self, dsets):
        if not self.mode == 'df' or not isinstance(dsets.items, pd.DataFrame):
            return
        dsets.items, count = tokenize_df(dsets.items, self.text_cols, rules=self.rules, **self.kwargs)
        if self.counter is None:
            self.counter = count
        return dsets

    def encodes(self, o: Path):
        if self.mode == 'folder' and str(o).startswith(str(self.path)):
            tok = self.output_dir / o.relative_to(self.path)
            return L(tok.read_text().split(' '))
        else:
            return self._tokenize1(o.read_text())

    def encodes(self, o: str): return self._tokenize1(o)
    def _tokenize1(self, o): return first(self.tok([compose(*self.rules)(o)]))

    def get_lengths(self, items):
        if self.lengths is None:
            return None
        if self.mode == 'df':
            if isinstance(items, pd.DataFrame) and 'text_lengths' in items.columns:
                return items['text_length'].values
        if self.mode == 'folder':
            try:
                res = [self.lengths[str(Path(i).relative_to(self.path))] for i in items]
                if len(res) == len(items):
                    return res
            except:
                return None

    def decodes(self, o): return TitledStr(self.sep.join(o))


with tempfile.TemporaryDirectory() as tmp_d:
    path, df, csv_fname = _prepare_texts(Path(tmp_d))
    items = get_text_files(path)
    splits = RandomSplitter()(items)
    dsets = Datasets(items, [Tokenizer.from_folder(path)], splits=splits)
    print(dsets.train[0])

    dsets = Datasets(df, [Tokenizer.from_df('text')], splits=splits)
    print(dsets.train[0][0].text)

tst = test_set(dsets, ['This is a test', 'this is another test'])
test_eq(tst, [(['xxbos', 'xxmaj', 'this', 'is', 'a', 'test'],),
              (['xxbos', 'this', 'is', 'another', 'test'],)])

# ## Sentencepiece

# export
eu_langs = ["bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "ga", "hr", "hu",
            "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro", "sk", "sl", "sv"]  # all European langs


# export
class SentencePieceTokenizer():  # TODO: pass the special tokens symbol to sp
    "SentencePiece tokenizer for `lang`"

    def __init__(self, lang='en', special_toks=None, sp_model=None, vocab_sz=None, max_vocab_sz=30000,
                 model_type='unigram', char_coverage=None, cache_dir='tmp'):
        try:
            from sentencepiece import SentencePieceTrainer, SentencePieceProcessor
        except ImportError:
            raise Exception('sentencepiece module is missing: run `pip install sentencepiece!=0.1.90,!=0.1.91`')
        self.sp_model, self.cache_dir = sp_model, Path(cache_dir)
        self.vocab_sz, self.max_vocab_sz, self.model_type = vocab_sz, max_vocab_sz, model_type
        self.char_coverage = ifnone(char_coverage, 0.99999 if lang in eu_langs else 0.9998)
        self.special_toks = ifnone(special_toks, defaults.text_spec_tok)
        if sp_model is None:
            self.tok = None
        else:
            self.tok = SentencePieceProcessor()
            self.tok.Load(str(sp_model))
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_vocab_sz(self, raw_text_path):
        cnt = Counter()
        with open(raw_text_path, 'r') as f:
            for line in f.readlines():
                cnt.update(line.split())
                if len(cnt) // 4 > self.max_vocab_sz:
                    return self.max_vocab_sz
        res = len(cnt) // 4
        while res % 8 != 0:
            res += 1
        return max(res, 29)

    def train(self, raw_text_path):
        "Train a sentencepiece tokenizer on `texts` and save it in `path/tmp_dir`"
        from sentencepiece import SentencePieceTrainer
        vocab_sz = self._get_vocab_sz(raw_text_path) if self.vocab_sz is None else self.vocab_sz
        spec_tokens = ['\u2581' + s for s in self.special_toks]
        SentencePieceTrainer.Train(" ".join([
            f"--input={raw_text_path} --vocab_size={vocab_sz} --model_prefix={self.cache_dir/'spm'}",
            f"--character_coverage={self.char_coverage} --model_type={self.model_type}",
            f"--unk_id={len(spec_tokens)} --pad_id=-1 --bos_id=-1 --eos_id=-1 --minloglevel=2",
            f"--user_defined_symbols={','.join(spec_tokens)} --hard_vocab_limit=false"]))
        raw_text_path.unlink()
        return self.cache_dir / 'spm.model'

    def setup(self, items, rules=None):
        from sentencepiece import SentencePieceProcessor
        if rules is None:
            rules = []
        if self.tok is not None:
            return {'sp_model': self.sp_model}
        raw_text_path = self.cache_dir / 'texts.out'
        with open(raw_text_path, 'w') as f:
            for t in progress_bar(maps(*rules, items), total=len(items), leave=False):
                f.write(f'{t}\n')
        sp_model = self.train(raw_text_path)
        self.tok = SentencePieceProcessor()
        self.tok.Load(str(sp_model))
        return {'sp_model': sp_model}

    def __call__(self, items):
        if self.tok is None:
            self.setup(items)
        for t in items:
            yield self.tok.EncodeAsPieces(t)


# export
SubwordTokenizer = SentencePieceTokenizer

texts = [f"This is an example of text {i}" for i in range(10)]
df = pd.DataFrame({'text': texts, 'label': list(range(10))}, columns=['text', 'label'])
out, cnt = tokenize_df(df, text_cols='text', tok=SentencePieceTokenizer(vocab_sz=34), n_workers=1)

# +
with tempfile.TemporaryDirectory() as tmp_d:
    path, df, csv_fname = _prepare_texts(Path(tmp_d))
    items = get_text_files(path)
    splits = RandomSplitter()(items)
    tok = SentencePieceTokenizer(special_toks=[])
    dsets = Datasets(items, [Tokenizer.from_folder(path, tok=tok)], splits=splits)
    print(dsets.train[0][0])

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=np.VisibleDeprecationWarning)
    dsets = Datasets(df, [Tokenizer.from_df('text', tok=tok)], splits=splits)
    print(dsets.train[0][0].text)
# -

# ## Export -

# hide
notebook2script()
