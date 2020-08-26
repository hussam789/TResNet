#!/usr/bin/env bash -x
source /workspace/venv/bin/activate
PYTHON=${PYTHON:-"python"}
export DEBIAN_FRONTEND=noninteractive
# REPO="$( cd "$(dirname "$0")" ; cd .. ; pwd -P )"
# cd $REPO

$PYTHON -m pip install -e .
$PYTHON -m pip install torch>=1.3
$PYTHON -m pip install torchvision>=0.4.0
$PYTHON -m pip install git+https://github.com/mapillary/inplace_abn.git@v1.0.10
$PYTHON -m pip install gdown
$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install --upgrade Pillow

### Pillow-simd with libjpeg turbo
#$PYTHON -m pip uninstall pillow
#CC="cc -mavx2" $PYTHON -m pip install -U --force-reinstall pillow-simd
#python -m pip uninstall -y pillow
#python -m pip install -U --force-reinstall pip
#conda install -c conda-forge libjpeg-turbo pillow==7.0.0
#apt-get -qq update
#apt-get install -y libwebp-dev
#CFLAGS="${CFLAGS} -mavx2" pip install --upgrade --no-cache-dir --force-reinstall \
#--no-binary :all:--compile https://github.com/mrT23/pillow-simd/zipball/simd/7.0.x
#conda install -y jpeg libtiff
#python -c "from PIL import Image; print(Image.PILLOW_VERSION)"

gdown https://drive.google.com/uc?id=12_VnXYI-4JaUYOOIsZXYCJdpiLCJ4dHV
gdown https://drive.google.com/uc?id=14X7xL1uf3c9dVGzRdV49bTjCBNvT9koA
gdown https://drive.google.com/uc?id=1HKFZEWPIwk5ZR0_MjJNkpQLyM_0yX0Qq
gdown https://drive.google.com/file/d/1pySDEdfvLRROaNXi-fWrd_aV9N6W9Yjo/view?usp=sharing
#gdown https://drive.google.com/uc?id=1cTOwVmxLqWhNl8zg2ZZJfiyejEcOq1RR
#gdown https://drive.google.com/uc?id=1jgKXMoJpZ6Sow_sJhgi5tec7AyeQCLbH
#gdown https://drive.google.com/uc?id=1IIPU77tcS83cBRTAfKeGvAmcBL5D196r
