#!/bin/bash
sudo python3.6 /usr/bin/add-apt-repository ppa:beineri/opt-qt-5.15.2-bionic
sudo apt update
sudo apt-get install -y libgl-dev
sudo apt-get install qt515base
source /opt/qt515/bin/qt515-env.sh
mkdir armadillo
cd armadillo
wget https://sourceforge.net/projects/arma/files/armadillo-12.2.0.tar.xz
tar -xf armadillo-12.2.0.tar.xz
cd armadillo-12.2.0
cmake .
sudo make install
git clone https://github.com/rjhogan/Adept-2.git
cd Adept-2
./configure
make
make check
sudo make install
git clone https://github.com/itsoulos/OPTIMUS.git
export OPTIMUSPATH=/home/morse/squander/OPTIMUS/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OPTIMUSPATH/lib/:$OPTIMUSPATH/PROBLEMS/
cd $OPTIMUSPATH
./compile.sh
