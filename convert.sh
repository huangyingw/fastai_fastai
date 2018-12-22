#!/bin/bash -
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd "$SCRIPTPATH"

find . -type f -name \*.ipynb -exec jupyter nbconvert --to=python --template=python.tpl {} \;
find . -type f -name *.ipynb | while read ss
do
    pyFile="${ss%.ipynb}.py"
    sed -i"" '/^$/d' "$pyFile"
    sed -i"" '/^#$/d' "$pyFile"
    autopep8 --in-place "$pyFile"
done
