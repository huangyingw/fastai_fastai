#!/bin/bash -
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd "$SCRIPTPATH"

copyDir="~/myproject/git/AI/fastai/fastai"
while read ss
do
    #ssh -n $ss "apt-get update && apt-get install realpath"
    ./copy.sh "$ss" "$copyDir"
    ssh -nY $ss "$copyDir/run.sh"
done < deploy.list
