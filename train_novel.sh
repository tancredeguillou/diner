#!/bin/bash

#SBATCH -n 1
#SBATCH --mem-per-cpu=8192
#SBATCH -J train_NOVEL_MLP
#SBATCH -o outputs/train_NOVEL_MLP.out
#SBATCH -e outputs/train_NOVEL_MLP.err
#SBATCH -t 10-00
#SBATCH --gpus=rtx_3090:1

python python_scripts/train.py configs/train_novel_facescape.yaml NOVEL
