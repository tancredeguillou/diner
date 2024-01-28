#!/bin/bash

#SBATCH -n 1
#SBATCH --mem-per-cpu=6144
#SBATCH -J eval_nMerge
#SBATCH -o outputs/eval_nMerge.out
#SBATCH -e outputs/eval_nMerge.err
#SBATCH -t 00-24
#SBATCH --gpus=rtx_3090:1
#SBATCH -A es_tang

python python_scripts/create_prediction_folder.py --config configs/evaluate_on_facescape.yaml --ckpt assets/ckpts/facescape/merge_330.ckpt --out outputs/facescape/merge330_full_evaluation --model DINER
