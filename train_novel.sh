#!/bin/bash

#SBATCH -n 1
#SBATCH --mem-per-cpu=8192
#SBATCH -J NCO_tgt
#SBATCH -o outputs/NCO_tgt.out
#SBATCH -e outputs/NCO_tgt.err
#SBATCH -t 10-00
#SBATCH --gpus=rtx_3090:1

python python_scripts/train.py configs/train_novel_facescape.yaml NOVEL NCC
