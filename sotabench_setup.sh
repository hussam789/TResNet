#!/usr/bin/env bash -x
source /workspace/venv/bin/activate
PYTHON=${PYTHON:-"python"}
# REPO="$( cd "$(dirname "$0")" ; cd .. ; pwd -P )"
# cd $REPO

$PYTHON -m pip install -e .
$PYTHON -m pip install torch>=1.3
$PYTHON -m pip install torchvision>=0.4.0
$PYTHON -m pip install git+https://github.com/mapillary/inplace_abn.git@v1.0.11
$PYTHON -m pip install gdown
gdown https://drive.google.com/uc?id=12_VnXYI-4JaUYOOIsZXYCJdpiLCJ4dHV
gdown https://drive.google.com/uc?id=14X7xL1uf3c9dVGzRdV49bTjCBNvT9koA
gdown https://drive.google.com/uc?id=1HKFZEWPIwk5ZR0_MjJNkpQLyM_0yX0Qq
