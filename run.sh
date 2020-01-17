#!/bin/bash
echo -e "\n---- Start ----\n"
model="resnet50"
optimizer="adam"
lr=0.0001 # learning_rate
min_lr=0.0000001
epochs=200
batch_size=128
img_size=256
crop_size=224

model_path="results/"$model"_"$optimizer"_"$img_size"to"$crop_size"_lr"$lr"_DA_pretrained_OnlyDamaged/"
#### --steps_scheduler / --plateau_scheduler / --pretrained / --weighted_loss / --binary_problem / --damaged_problem
CUDA_VISIBLE_DEVICES=0,1 python train.py --model_name $model --optimizer $optimizer --learning_rate $lr \
                                       --min_learning_rate $min_lr  --batch_size $batch_size --epochs $epochs \
                                       --img_size $img_size --crop_size $crop_size --output_dir $model_path \
                                       --steps_scheduler --pretrained --data_augmentation --damaged_problem

model_path="results/"$model"_"$optimizer"_"$img_size"to"$crop_size"_lr"$lr"_DA_pretrained_Binary/"
CUDA_VISIBLE_DEVICES=0,1 python train.py --model_name $model --optimizer $optimizer --learning_rate $lr \
                                       --min_learning_rate $min_lr  --batch_size $batch_size --epochs $epochs \
                                       --img_size $img_size --crop_size $crop_size --output_dir $model_path \
                                       --steps_scheduler --pretrained --data_augmentation --binary_problem

model_path="results/"$model"_"$optimizer"_"$img_size"to"$crop_size"_lr"$lr"_DA_pretrained_Full/"
CUDA_VISIBLE_DEVICES=0,1 python train.py --model_name $model --optimizer $optimizer --learning_rate $lr \
                                       --min_learning_rate $min_lr  --batch_size $batch_size --epochs $epochs \
                                       --img_size $img_size --crop_size $crop_size --output_dir $model_path \
                                       --steps_scheduler --pretrained --data_augmentation
