#!/usr/bin/env bash

#PBS -q APPLI
#PBS -o /home/s1620007/deepNN/deepNN.out
#PBS -e /home/s1620007/deepNN/deepNN.in
#PBS -N deepNN
#PBS -j oe

cd /home/s1620007/deepNN

setenv PATH ${PBS_O_PATH}

root="$PWD"

THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float64 python $root/main.py -config config/sentence_memNN.cfg -mode deep_memNN