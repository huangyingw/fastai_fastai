#!/bin/bash -
SCRIPT=$(realpath "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
cd "$SCRIPTPATH"

server="$1"
dir="$2"

rsync -aHv --progress --delete \
    --exclude-from=./excludeFile \
    ./ \
    "$server":"$dir"
