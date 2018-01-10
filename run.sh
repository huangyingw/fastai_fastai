#!/bin/bash -
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd "$SCRIPTPATH"

source /etc/profile
source ~/.profile

cd courses/dl1
~/anaconda3/bin/python lesson1.py
