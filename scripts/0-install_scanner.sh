#!/bin/bash

# Install tensorflow
sudo apt-get --assume-yes update
sudo apt-get --assume-yes install python-pip python-dev
pip install tensorflow

# Install scanner
SCANNER_PATH=$HOME/scanner
git clone https://github.com/scanner-research/scanner.git ${SCANNER_PATH}
cd ${SCANNER_PATH}
git checkout 4776102
sudo apt-get --assume-yes install \
   build-essential \
   cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev \
   libswscale-dev unzip llvm clang libc++-dev libgflags-dev libgtest-dev \
   libssl-dev libcurl3-dev liblzma-dev libeigen3-dev  \
   libgoogle-glog-dev libatlas-base-dev libsuitesparse-dev libgflags-dev \
   libx264-dev libopenjpeg-dev libxvidcore-dev \
   libpng-dev libjpeg-dev libbz2-dev git python-pip wget \
   libleveldb-dev libsnappy-dev libhdf5-serial-dev liblmdb-dev python-dev \
   python-tk autoconf autogen libtool libtbb-dev libopenblas-dev \
   liblapacke-dev swig yasm python2.7 cpio \
   automake libass-dev libfreetype6-dev libsdl2-dev libtheora-dev libtool \
   libva-dev libvdpau-dev libvorbis-dev libxcb1-dev libxcb-shm0-dev \
   libxcb-xfixes0-dev mercurial pkg-config texinfo wget zlib1g-dev \
   curl unzip libcap-dev htop

pip install -r requirements.txt
pip install docopt
# All assume no
yes n | bash ./deps.sh 

cat >> $HOME/.bashrc <<EOM
export LD_LIBRARY_PATH=\$HOME/scanner/thirdparty/install/lib:\$LD_LIBRARY_PATH
export PATH=\$HOME/scanner/thirdparty/install/bin:\$PATH
export PYTHONPATH=\$HOME/scanner/thirdparty/install/lib/python2.7/dist-packages:\$PYTHONPATH
EOM

export LD_LIBRARY_PATH=$HOME/scanner/thirdparty/install/lib:$LD_LIBRARY_PATH
export PATH=$HOME/scanner/thirdparty/install/bin:$PATH
export PYTHONPATH=$HOME/scanner/thirdparty/install/lib/python2.7/dist-packages:$PYTHONPATH

cd ${SCANNER_PATH}
mkdir build
cd build
cmake -D BUILD_TESTS=ON -D BUILD_CUDA=OFF ..
make -j8

cd ${SCANNER_PATH}
python python/setup.py bdist_wheel
pip install dist/scannerpy-0.1.13-py2-none-any.whl

