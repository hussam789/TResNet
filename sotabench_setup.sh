#!/usr/bin/env bash -x
source /workspace/venv/bin/activate
PYTHON=${PYTHON:-"python"}
# REPO="$( cd "$(dirname "$0")" ; cd .. ; pwd -P )"
# cd $REPO

$PYTHON -m pip install -e .
$PYTHON -m pip install torch>=1.3
$PYTHON -m pip install torchvision>=0.4.0
$PYTHON -m pip install git+https://github.com/mapillary/inplace_abn.git@v1.0.11
