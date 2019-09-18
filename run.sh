#!/bin/bash
echo -e "\n---- Start ----\n"
model="resnet18" #resnet_unet
optimizer="sgd"
lr=0.01 # learning_rate
min_lr=0.00001
epochs=200
batch_size=32
img_size=224
crop_size=224
model_path="results/"$model"_"$optimizer"_"$img_size"to"$crop_size"_lr"$lr"_NoDa/"

CUDA_VISIBLE_DEVICES=0 python train.py --model_name $model --optimizer $optimizer --learning_rate $lr \
                                       --min_learning_rate $min_lr  --batch_size $batch_size --epochs $epochs \
                                       --img_size $img_size --crop_size $crop_size --output_dir $model_path \
                                       --steps_scheduler