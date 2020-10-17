#!/bin/bash -
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd "$SCRIPTPATH"

git submodule init
git submodule sync --recursive
git submodule update --recursive

conda install -c fastai -c pytorch -c anaconda fastai gh anaconda
pip install -Uqq fastbook
pip install waterfallcharts treeinterpreter dtreeviz
pip install nbdev --upgrade
#pip install -e ".[dev]"
#pip install -e "fastai[dev]"

conda install -c conda-forge \
    graphviz \
    jupytext \
    neovim
