#!/bin/bash

# use CUDNN

#export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cudnn/lib64
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cudnn/lib64
#export CPATH=$CPATH:/usr/local/cudnn/include

export THEANO_FLAGS=device=cuda,floatX=float32,optimizer_including=cudnn,gpuarray.preallocate=0.95
nohup python -u ./train_nmt.py ./models/ 1>log.txt 2>&1 &
tail -f log.txt

