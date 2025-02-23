#!/bin/bash

: ${BATCH_SIZE:=128}
: ${NET:=''}
: ${EPOCHS:=1}

nohup python -u server/server.py > server.log 2>&1 &
python train.py -net ${NET} -gpu -b ${BATCH_SIZE} -epochs ${EPOCHS}
