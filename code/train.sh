#!/bin/bash

name=(1 2 3 4 5)
# bert_tiny


for ((i=0; i<5; i++))
do

    python train.py \
                --dataFile "../data/HWU64/0${name[i]}" \
                --fileVocab "../bert_tiny/vocab.txt" \
                --fileModelConfig "../bert_tiny/config.json" \
                --fileModel "../bert_tiny/pytorch_model.bin" \
                --fileModelSave "../model/HWU64_n5k5_r${name[i]}" \
                --numDevice 1 \
                --learning_rate 1e-4 \
                --epochs 40 \
                --numNWay 5 \
                --numKShot 5 \
                --numQShot 5 \
                --episodeTrain 100 \
                --episodeTest 500 \
                --numFreeze 0 \
                --warmup_steps 100 \
                --dropout_rate 0.1 \
                --weight_decay 0.2 \
                --tolerate 10
                            
done         
