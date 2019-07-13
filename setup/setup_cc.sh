#!/bin/bash

module load python/3.7.0
python -m venv venv
source venv/bin/activate

# installing dependencies
pip install gym
pip install sacred pymongo GitPython
pip install pandas matplotlib

# module avail cuda
module load cuda/10.0.130
pip install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp37-cp37m-linux_x86_64.whl


pip download pybullet
tar -xzf pybullet*.tar.gz
cd $(find -type d -iname "pybullet-*")
sed -i -- 's/2 \* multiprocessing.cpu_count()/4/g' setup.py
python setup.py install
cd ..
rm -r pybullet-*
