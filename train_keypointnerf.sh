#!/bin/bash

#SBATCH -n 4
#SBATCH --mem-per-cpu=2048
#SBATCH -J train_KeypointNeRF_val
#SBATCH -o outputs/train_KeypointNeRF_val.out
#SBATCH -e outputs/train_KeypointNeRF_val.err
#SBATCH -t 10-24
#SBATCH --gpus=rtx_3090:1
#SBATCH -A es_tang

python python_scripts/train.py configs/train_keypointnerf_facescape.yaml KeypointNeRF
