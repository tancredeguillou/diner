#!/bin/bash

#SBATCH -n 4
#SBATCH --mem-per-cpu=2048
#SBATCH -J train_TransMVSNet
#SBATCH -o out_train_TransMVSNet.out
#SBATCH -e err_train_TransMVSNet.err
#SBATCH -t 00-24
#SBATCH --gpus=rtx_3090:1

MVS_TRAINING="/cluster/scratch/tguillou/facescape_color_calibrated"          # path to dataset mvs_training
LOG_DIR="outputs/facescape/TransMVSNet_training" # path to checkpoints
NGPUS=1
BATCH_SIZE=1

if [ ! -d $LOG_DIR ]; then
  mkdir -p $LOG_DIR
fi

python -m torch.distributed.launch --nproc_per_node=$NGPUS deps/TransMVSNet/train.py \
  --logdir=$LOG_DIR \
  --dataset=facescape \
  --batch_size=$BATCH_SIZE \
  --epochs=20 \
  --trainpath=$MVS_TRAINING \
  --numdepth=384 \
  --ndepths="96,64,16" \
  --nviews=2 \
  --wd=0.0001 \
  --depth_inter_r="4.0,1.0,0.5" \
  --lrepochs="1,2,3:2" \
  --dlossw="1.0,1.0,1.0"
