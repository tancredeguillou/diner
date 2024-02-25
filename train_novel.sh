#!/bin/bash

#SBATCH -n 4
#SBATCH --mem-per-cpu=2048
#SBATCH -J gen
#SBATCH -o outputs/gen.out
#SBATCH -e outputs/gen.err
#SBATCH -t 10-00
#SBATCH --gpus=rtx_3090:1
#SBATCH -A es_tang

python python_scripts/train.py configs/train_novel_facescape.yaml NOVEL NOC
