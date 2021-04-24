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
import tarfile
import zipfile
from wheel.metadata import pkginfo_to_metadata
from conda.gateways.disk.read import read_python_record
from wheel import wheelfile
from email.parser import Parser
from fastai.torch_basics import *
! [-e / content] & & pip install - Uqq fastai  # upgrade fastai on colab

# convert pip wheel -> conda package
#
# whl   zip
#  - contains cymem-2.0.2.dist-info : LICENSE  METADATA  RECORD  top_level.txt  WHEEL
#
# conda tgz
#  - includes lib/python{v}/site-packages/
#  - contains info : "about.json  files  git  hash_input.json  index.json  paths.json"
#
# conversion:
#
# 1. move main modules to lib/...
# 2. create info
#
# ---
#
# - about: home license summary
# - files: list
# - git: empty
# - hash_input: empty
# - index.json: create
# - paths: files with sha256 and size


path = Path('~/git/spacy_conda/cymem').expanduser()
pkg = path / 'pkg/cymem-2.0.2-py37_0.tar.bz2'
whl = path / 'whl/cymem-2.0.2-cp37-cp37m-manylinux1_x86_64.whl'
pkg.exists(), whl.exists()

zpkg = tarfile.open(pkg, "r:bz2")

lpkg = zpkg.getmembers()
[(i, o.name) for i, o in enumerate(lpkg) if o.name.startswith('info/')]

f = (zpkg.extractfile('info/about.json').read().decode())
print(f)
fn.name

fwhl = wheelfile.WheelFile(whl)

fwhl.parsed_filename.groupdict()

fwhl.dist_info_path

fwhl.record_path

read_python_record(whl.parent, fwhl.record_path, '3.7')['paths_data'].paths

fwhl.namelist()


meta = fwhl.read(f'{fwhl.dist_info_path}/METADATA').decode()

m = Parser().parsestr(meta)
m.items()

zip_ref = zipfile.ZipFile(whl, "r")
ls = zip_ref.filelist
[(i, o.filename) for i, o in enumerate(ls)]

fn = ls[11]
f = zip_ref.read(fn)
print(f.decode())
fn.filename
