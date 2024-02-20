#!/bin/bash

#SBATCH -n 4
#SBATCH --mem-per-cpu=2048
#SBATCH -J write_to_multiface
#SBATCH -o outputs/write_to_multiface.out
#SBATCH -e outputs/write_to_multiface.err
#SBATCH -t 00-24
#SBATCH --gpus=rtx_3090:1

export PYTHONPATH="deps/TransMVSNet:$PYTHONPATH"

DATA_ROOT="/cluster/scratch/tguillou/diner_multiface" # path to processed facescape dataset
OUTDEPTHNAME="TransMVSNet"  # prefix of the output depth files
LOG_DIR="outputs/multiface/TransMVSNet_writing"
CKPT="assets/ckpts/facescape/TransMVSNet.ckpt"  # path to pretrained checkpoint
NGPUS=1
BATCH_SIZE=1

if [ ! -d $LOG_DIR ]; then
	mkdir -p $LOG_DIR
fi
python -m torch.distributed.launch --nproc_per_node=$NGPUS deps/TransMVSNet/train_old.py \
  --mode="write_prediction" \
  --outdepthname=$OUTDEPTHNAME \
  --maskoutput \
  --loadckpt=$CKPT \
	--logdir=$LOG_DIR \
	--dataset=multiface \
	--batch_size=$BATCH_SIZE \
	--trainpath=$DATA_ROOT \
	--numdepth=384 \
	--ndepths="96,64,16" \
	--nviews=4 \
	--depth_inter_r="4.0,1.0,0.5" \
	--lrepochs="7,10,15:2" | tee -a $LOG_DIR/log.txt
