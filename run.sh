#!/bin/bash -
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd "$SCRIPTPATH"

source /root/anaconda3/bin/activate fastai

python ./courses/dl1/lesson1.py
