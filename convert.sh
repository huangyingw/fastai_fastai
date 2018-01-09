#!/bin/bash -
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd "$SCRIPTPATH"

find . -type f -name \*.ipynb -exec jupyter nbconvert --to=python --template=python.tpl {} \;
find . -type f -name *.ipynb | while read ss
do
    autopep8 --in-place --aggressive "${ss%.ipynb}.py"
done
