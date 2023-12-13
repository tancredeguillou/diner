#!/bin/bash

#SBATCH -n 4
#SBATCH --mem-per-cpu=2048
#SBATCH -J train_DINER
#SBATCH -o outputs/train_DINER.out
#SBATCH -e outputs/train_DINER.err
#SBATCH -t 05-24
#SBATCH --gpus=rtx_3090:1
#SBATCH -A es_tang

python python_scripts/train.py configs/train_facescape.yaml DINER
