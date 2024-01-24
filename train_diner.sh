#!/bin/bash

#SBATCH -n 4
#SBATCH --mem-per-cpu=2048
#SBATCH -J train_DINER_OG
#SBATCH -o outputs/train_DINER_OG.out
#SBATCH -e outputs/train_DINER_OG.err
#SBATCH -t 10-24
#SBATCH --gpus=rtx_3090:1
#SBATCH -A es_tang

python python_scripts/train.py configs/train_diner_facescape.yaml DINER