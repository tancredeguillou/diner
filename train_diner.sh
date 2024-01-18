#!/bin/bash

#SBATCH -n 4
#SBATCH --mem-per-cpu=2048
#SBATCH -J train_MESH_DINER_0
#SBATCH -o outputs/train_MESH_DINER_0.out
#SBATCH -e outputs/train_MESH_DINER_0.err
#SBATCH -t 10-24
#SBATCH --gpus=rtx_3090:1
#SBATCH -A es_tang

python python_scripts/train.py configs/train_diner_facescape.yaml DINER