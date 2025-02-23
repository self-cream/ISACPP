#!/bin/bash

: ${BATCH_SIZE:=2048}
: ${LEARNING_RATE:=0.005}
: ${EPOCHS:=1}

python3 gnmt_train.py --seed 2 --train-global-batch-size ${BATCH_SIZE} --epochs ${EPOCHS} --lr ${LEARNING_RATE} --math 'manual_fp16'
