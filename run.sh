#!/bin/bash

pip install --upgrade pip --user

pip install -r requirements.txt --user

pip install ./configs/torch_cluster-1.5.9-cp36-cp36m-linux_x86_64.whl --user

cd torchsearchsorted

pip install .

cd ..

# run the work
basedir=./logs/dynamic_synthesis

python run.py --config configs/dynamic/synthesis/trex.py --render_test --basedir $basedir


