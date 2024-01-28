#!/bin/bash

#SBATCH -n 1
#SBATCH --mem-per-cpu=7168
#SBATCH -J train_NOVEL2
#SBATCH -o outputs/train_NOVEL2.out
#SBATCH -e outputs/train_NOVEL2.err
#SBATCH -t 10-00
#SBATCH --gpus=rtx_3090:1

python python_scripts/train.py configs/train_novel_facescape.yaml NOVEL
