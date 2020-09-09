#!/bin/bash
echo -e "\n---- Start ----\n"
model="mobilenetwd4"
optimizer="adam"
min_lr=0.0000001
epochs=200
batch_size=256
img_size=256
crop_size=224

for lrFor in 0.001 0.0001
do

lr=$lrFor # learning_rate

#model_path="results/"$model"_"$optimizer"_"$img_size"to"$crop_size"_lr"$lr"_DA_pretrained_OnlyDamaged/"
##### --steps_scheduler / --plateau_scheduler / --pretrained / --weighted_loss / --binary_problem / --damaged_problem
#CUDA_VISIBLE_DEVICES=0,1 python train.py --model_name $model --optimizer $optimizer --learning_rate $lr \
#                                       --min_learning_rate $min_lr  --batch_size $batch_size --epochs $epochs \
#                                       --img_size $img_size --crop_size $crop_size --output_dir $model_path \
#                                       --steps_scheduler --pretrained --data_augmentation --damaged_problem
#
#model_path="results/"$model"_"$optimizer"_"$img_size"to"$crop_size"_lr"$lr"_DA_pretrained_Binary/"
#CUDA_VISIBLE_DEVICES=0,1 python train.py --model_name $model --optimizer $optimizer --learning_rate $lr \
#                                       --min_learning_rate $min_lr  --batch_size $batch_size --epochs $epochs \
#                                       --img_size $img_size --crop_size $crop_size --output_dir $model_path \
#                                       --steps_scheduler --pretrained --data_augmentation --binary_problem

#model_path="results/"$model"_"$optimizer"_"$img_size"to"$crop_size"_lr"$lr"_DA_pretrained_Full/"
#CUDA_VISIBLE_DEVICES=0 python train.py --model_name $model --optimizer $optimizer --learning_rate $lr \
#                                       --min_learning_rate $min_lr  --batch_size $batch_size --epochs $epochs \
#                                       --img_size $img_size --crop_size $crop_size --output_dir $model_path \
#                                       --steps_scheduler --pretrained --data_augmentation


# Segmentation models -> unet unet_small unet_extra_small unet_nano
model="unet_extra_small"
img_size=512
crop_size=512
batch_size=6
selected_class="Grietas longitudinales"
model_path="results/segmentation/$selected_class/"$model"_"$optimizer"_"$img_size"to"$crop_size"_lr"$lr"/"
CUDA_VISIBLE_DEVICES=0,1 python train.py --model_name $model --optimizer $optimizer --learning_rate $lr \
                                       --min_learning_rate $min_lr  --batch_size $batch_size --epochs $epochs \
                                       --img_size $img_size --crop_size $crop_size --output_dir "$model_path" \
                                       --steps_scheduler --data_augmentation --segmentation_problem \
                                       --selected_class "$selected_class" --masks_overlays 10


done