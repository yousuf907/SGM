#training script for vanilla model
EXPT_NAME=Vanilla_ImageNet1K_Places365_Std_1365C
IMAGENET_DATA_DIR=/data/datasets/ImageNet1K ## pretrain/base dataset
IMAGENET_LABEL_DIR=./imagenet_files/
PLACES_DATA_DIR=/data/datasets/Places365_standard ## new dataset for CL
PLACES_LABEL_DIR=./imagenet_files/places_indices
BASE_INIT_CKPT=./convnextv2_femto_1k_224_ema.pt # pretrained on pretrain/base dataset
GPU=0
MAX_BUFFER_SIZE=5400 # 5400/ 9600/ 38400 # << budget 24K/ 48K/ 192K >>
BASE_INIT_CLASSES=1000 # ImageNet-1K pretrain/base dataset
CL_MIN_CLASS=0
CL_MAX_CLASS=365 # Places-365
CLASS_INCREMENT=73 # Places
NUM_CLASSES=1365 # Imagenet-1K + Places-365
ITER=1200 # Hyperparameter (Adjust based on compute budget which is ITER x BATCH)
BATCH=128 # Number of rehearsal samples in a minibatch during training
LR=1e-3 # Hyperparameter
WD=5e-2 # Hyperparameter
SEED=1993


CUDA_VISIBLE_DEVICES=${GPU} python -u train_vanilla.py \
--seed ${SEED} \
--images_dir_old ${IMAGENET_DATA_DIR} \
--label_dir_old ${IMAGENET_LABEL_DIR} \
--images_dir ${PLACES_DATA_DIR} \
--label_dir ${PLACES_LABEL_DIR} \
--max_buffer_size ${MAX_BUFFER_SIZE} \
--num_classes ${NUM_CLASSES} \
--cl_min_class ${CL_MIN_CLASS} \
--cl_max_class ${CL_MAX_CLASS} \
--base_init_classes ${BASE_INIT_CLASSES} \
--class_increment ${CLASS_INCREMENT} \
--classifier_ckpt ${BASE_INIT_CKPT} \
--num_iter ${ITER} \
--batch_size ${BATCH} \
--weight_decay ${WD} \
--lr ${LR} \
--base_arch ConvNeXtNet \
--classifier ConvNeXt_block2 \
--extract_features_from model.downsample_layers.2 \
--hidden_dim 384 \
--save_dir ${EXPT_NAME} \
--ckpt_file ${EXPT_NAME}.pth \
--expt_name ${EXPT_NAME} > logs/${EXPT_NAME}.log
