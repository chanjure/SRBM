#!/bin/bash

python3 -m torch.utils.collect_env

#SRBM_PATH=/home/s.2217762/workspace/SRBM/
#DATA_PATH=/home/s.2217762/workspace/qftml/data/
#MODEL_PATH=/home/s.2217762/workspace/qftml/model/

SRBM_PATH=/home/chanju/Dropbox/Lab/swansea/workspace/Git/SRBM/
DATA_PATH=/home/chanju/Dropbox/Lab/swansea/workspace/projects/QFTML/code/datas/
MODEL_PATH=/home/chanju/Dropbox/Lab/swansea/workspace/projects/QFTML/code/models/

python3 $SRBM_PATH/bin/train_mnist.py --srbm_path $SRBM_PATH --data_path $DATA_PATH --model_path $MODEL_PATH --gpu_id 1 --batch_size 16 --lr 1e-3 --lr_decay 0.9999 --epoch 1 --k 3 --seed 42 --m 12 --sig 1. --m_scheme 0 --n_h 10
