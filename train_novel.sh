#!/bin/bash

#SBATCH -n 4
#SBATCH --mem-per-cpu=2048
#SBATCH -J gen_pe
#SBATCH -o outputs/gen_pe.out
#SBATCH -e outputs/gen_pe.err
#SBATCH -t 10-00
#SBATCH --gpus=rtx_3090:1
#SBATCH -A es_tang

python python_scripts/train.py configs/train_novel_facescape.yaml NOVEL NOC
