#!/bin/bash -
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd "$SCRIPTPATH"

source /root/anaconda3/bin/activate fastai

cd courses/dl1
python lesson1.py
