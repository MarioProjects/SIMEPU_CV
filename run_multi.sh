#!/bin/bash
echo -e "\n---- Start ----\n"

######################################################################################################################
############################################ MULTI LABEL CLASSIFICATION ##############################################
######################################################################################################################


model="resnet34"
optimizer="adam"
min_lr=0.0000001
epochs=120
batch_size=64
img_size=512
crop_size=512
lr=0.001
randaug_n=4
randaug_m=10
cutout_size=30

for fold in 0 1 2 3 4
do

model_path="results/MultiLabelClassification/fold"$fold"/"$model"_"$optimizer"_"$img_size"to"$crop_size"_lr"$lr"_DA_pretrained/"
CUDA_VISIBLE_DEVICES=0,1 python train.py --model_name $model --optimizer $optimizer --learning_rate $lr \
   --min_learning_rate $min_lr --batch_size $batch_size --epochs $epochs \
   --img_size $img_size --crop_size $crop_size --output_dir $model_path \
   --steps_scheduler --multilabel_problem --fold $fold \
   --randaug_n $randaug_n --randaug_m $randaug_m --cutout_size $cutout_size

done


######################################################################################################################
############################################### SEGMENTATION #########################################################
######################################################################################################################
# Segmentation models -> unet unet_small unet_extra_small unet_nano
: '
lr=0.01 # learning_rate

for fold in 0 1 2 3 4
do


model="unet_extra_small"
img_size=512
crop_size=512
batch_size=6
scale_factor=2
selected_class="Grietas"
model_path="results/segmentation/fold"$fold"/$selected_class/"$model"_"$optimizer"_"$img_size"to"$crop_size"_lr"$lr"/"
CUDA_VISIBLE_DEVICES=0,1 python train.py --model_name $model --optimizer $optimizer --learning_rate $lr \
                                       --min_learning_rate $min_lr --batch_size $batch_size --epochs $epochs \
                                       --img_size $img_size --crop_size $crop_size --output_dir "$model_path" \
                                       --steps_scheduler --data_augmentation --segmentation_problem \
                                       --selected_class "$selected_class" --masks_overlays 40 --fold $fold \
                                       --unet_scale_factor $scale_factor


done


model="unet_extra_small"
img_size=512
crop_size=512
batch_size=32
selected_class="Grietas longitudinales"
model_path="results/segmentation/fold"$fold"/$selected_class/"$model"_"$optimizer"_"$img_size"to"$crop_size"_lr"$lr"/"
CUDA_VISIBLE_DEVICES=0,1 python train.py --model_name $model --optimizer $optimizer --learning_rate $lr \
                                       --min_learning_rate $min_lr --batch_size $batch_size --epochs $epochs \
                                       --img_size $img_size --crop_size $crop_size --output_dir "$model_path" \
                                       --steps_scheduler --data_augmentation --segmentation_problem \
                                       --selected_class "$selected_class" --masks_overlays 40 --fold $fold

model="unet_extra_small"
img_size=512
crop_size=512
batch_size=32
selected_class="Grietas transversales"
model_path="results/segmentation/fold"$fold"/$selected_class/"$model"_"$optimizer"_"$img_size"to"$crop_size"_lr"$lr"/"
CUDA_VISIBLE_DEVICES=0,1 python train.py --model_name $model --optimizer $optimizer --learning_rate $lr \
                                        --min_learning_rate $min_lr --batch_size $batch_size --epochs $epochs \
                                        --img_size $img_size --crop_size $crop_size --output_dir "$model_path" \
                                        --steps_scheduler --data_augmentation --segmentation_problem \
                                        --selected_class "$selected_class" --masks_overlays 40 --fold $fold


model="unet_extra_small"
img_size=512
crop_size=512
batch_size=32
selected_class="Huecos"
model_path="results/segmentation/fold"$fold"/$selected_class/"$model"_"$optimizer"_"$img_size"to"$crop_size"_lr"$lr"/"
CUDA_VISIBLE_DEVICES=0,1 python train.py --model_name $model --optimizer $optimizer --learning_rate $lr \
                                       --min_learning_rate $min_lr --batch_size $batch_size --epochs $epochs \
                                       --img_size $img_size --crop_size $crop_size --output_dir "$model_path" \
                                       --steps_scheduler --data_augmentation --segmentation_problem \
                                       --selected_class "$selected_class" --masks_overlays 40 --fold $fold


model="unet_extra_small"
img_size=512
crop_size=512
batch_size=8
selected_class="Parcheo"
model_path="results/segmentation/fold"$fold"/$selected_class/"$model"_"$optimizer"_"$img_size"to"$crop_size"_lr"$lr"/"
CUDA_VISIBLE_DEVICES=0,1 python train.py --model_name $model --optimizer $optimizer --learning_rate $lr \
                                      --min_learning_rate $min_lr --batch_size $batch_size --epochs $epochs \
                                       --img_size $img_size --crop_size $crop_size --output_dir "$model_path" \
                                       --steps_scheduler --data_augmentation --segmentation_problem \
                                       --selected_class "$selected_class" --masks_overlays 40 --fold $fold

done
'
