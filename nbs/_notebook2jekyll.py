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
import jupyter_contrib_nbextensions
import nbformat
from IPython.display import FileLink
from nbconvert.preprocessors import Preprocessor
from fastai.core.imports import *
from nbdev.export2html import *
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab


class NBConvertor:

    def _exporter():
        exporter = MarkdownExporter(Config())
        exporter.exclude_input_prompt = True
        exporter.exclude_output_prompt = True
        exporter.template_file = 'jekyll.tpl'
        exporter.template_path.append(str((Path() / 'local' / 'notebook').absolute()))
        return exporter

    _re_title = re.compile(r'^\s*#\s+([^\n]*)\n')
    cell_type, outputs, source, code, text = 'cell_type', 'outputs', 'source', 'code', 'text'

    def process_output(c, s, o):
        if c[cell_type] != code or o is None:
            return s, o

        def _f(x):
            if text not in x:
                return x
            x[text] = re.sub(r'^(.*\S)', r'> \1', x[text], flags=re.MULTILINE)
            return x
        return s, [_f(o_) for o_ in o]

    def process_title(c, s, o):
        if s.startswith('#hide'):
            return
        if c[cell_type] == code:
            return s, o
        if _re_title.search(s):
            s = '---\n' + _re_title.sub(r'title: "\1"', s) + '\n---'
            s = re.sub('^- ', '', s, flags=re.MULTILINE)
        return s, o

    def apply_all(x, fs, **kwargs):
        for f in fs:
            s, o = f(x, x[source], x.get(outputs, None), **kwargs) or (None, None)
            x[source] = s
            if s is None:
                x = None
                break
            elif o is not None:
                x[outputs] = o
        return x

    def convert(fname, dest=None, cell_procs=None):
        fname = Path(fname)
        (fname.parent / 'md_out').mkdir(exist_ok=True)
        if dest is None:
            dest = (fname.parent / 'md_out' / fname.name).with_suffix('.md')
        if cell_procs is None:
            cell_procs = [process_title, process_output]
        with open(fname, 'r') as f:
            nb = nbformat.reads(f.read(), as_version=4)
        nb['cells'] = [o for o in [apply_all(c, cell_procs) for c in nb['cells']] if o is not None]
        exp = _exporter()
        with open(dest, 'w') as f:
            f.write(exp.from_notebook_node(nb)[0])


name = 'delegation'
convert(Path.cwd() / f'{name}.ipynb')

FileLink(f'md_out/{name}.html')
