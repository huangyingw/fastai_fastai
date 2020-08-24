#!/bin/bash -
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd "$SCRIPTPATH"

git submodule init
git submodule sync --recursive
git submodule update --recursive

pip install -e "fastai[dev]"

conda install -c conda-forge \
    jupytext \
    neovim
