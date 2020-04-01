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

# pillow simd
apt-get -qq update
apt-get install -y libwebp-dev
CFLAGS="${CFLAGS} -mavx2" pip install --upgrade --no-cache-dir --force-reinstall \
--no-binary :all:--compile https://github.com/mrT23/pillow-simd/zipball/simd/7.0.x
apt-get install libjpeg-dev
python -c "from PIL import Image; print(Image.PILLOW_VERSION)"
python -c "from PIL import features; assert features.check_feature('libjpeg_turbo'), 'libjpeg_turbo import issue'"

gdown https://drive.google.com/uc?id=12_VnXYI-4JaUYOOIsZXYCJdpiLCJ4dHV
gdown https://drive.google.com/uc?id=14X7xL1uf3c9dVGzRdV49bTjCBNvT9koA
gdown https://drive.google.com/uc?id=1HKFZEWPIwk5ZR0_MjJNkpQLyM_0yX0Qq
