#!/bin/bash

#SBATCH -n 4
#SBATCH --mem-per-cpu=2048
#SBATCH -J train_NOVEL
#SBATCH -o outputs/train_NOVEL.out
#SBATCH -e outputs/train_NOVEL.err
#SBATCH -t 10-24
#SBATCH --gpus=rtx_3090:1
#SBATCH -A es_tang

python python_scripts/train.py configs/train_novel_facescape.yaml NOVEL
