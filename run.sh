#!/bin/bash

lr=(0.0003 
  0.0003 0.0003 0.0003 0.0003 
  0.0001 0.0001 0.0001
  0.00005 0.00005
  0.00001)
# evaluate performance every epoch
for i in $(seq 1 10)
do
    echo "############" $i "training #################"
  #   python train.py --checkpoint_path=/home/zisang/Documents/lesson9/vgg16-ckpt/vgg_16.ckpt \
  # --train_dir=../output/train \
  # --dataset_train=./fcn_train.record \
  # --batch_size=16 \
  # --max_steps=2

    python train_eval.py --checkpoint_path=/home/zisang/Documents/lesson9/vgg16-ckpt/vgg_16.ckpt \
  --train_dir=../output/train \
  --eval_dir=../output/eval/train \
  --dataset_train=./fcn_train.record \
  --dataset_val=./fcn_train.record \
  --batch_size=16 \
  --num_pics=5 \
  --max_steps=200 \
  --learning_rate=${lr[$i]}

    echo "############" $i "evaluating #################"
    python eval.py --train_dir=../output/train \
  --eval_dir=../output/eval/eval \
  --dataset_val=./fcn_val.record \
  --num_pics=5
done

# # 设置目录，避免module找不到的问题
# export PYTHONPATH=$PYTHONPATH:/home/zisang/objDetect:/home/zisang/objDetect/slim:/home/zisang/objDetect/object_detection

# python convert_fcn_dataset.py --data_dir=/home/zisang/Documents/lesson9/VOCdevkit/VOC2012/ --output_dir=./

# python train.py --checkpoint_path=/home/zisang/Documents/lesson9/vgg16-ckpt/vgg_16.ckpt \
#   --output_dir=../output \
#   --dataset_train=./fcn_train.record \
#   --dataset_val=./fcn_val.record \
#   --batch_size=16 \
#   --max_steps 20