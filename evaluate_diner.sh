#!/bin/bash

#SBATCH -n 4
#SBATCH --mem-per-cpu=2048
#SBATCH -J eval_DINER
#SBATCH -o outputs/eval_DINER.out
#SBATCH -e outputs/eval_DINER.err
#SBATCH -t 00-24
#SBATCH --gpus=rtx_3090:1
#SBATCH -A es_tang

python python_scripts/create_prediction_folder.py --config configs/evaluate_on_facescape.yaml --ckpt assets/ckpts/facescape/DINER.ckpt --out outputs/facescape/diner_full_evaluation --model DINER
