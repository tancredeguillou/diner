#!/bin/bash

#SBATCH -n 4
#SBATCH --mem-per-cpu=2048
#SBATCH -J eval_780000
#SBATCH -o outputs/eval_780000.out
#SBATCH -e outputs/eval_780000.err
#SBATCH -t 00-24
#SBATCH --gpus=rtx_3090:1
#SBATCH -A es_tang

python python_scripts/create_prediction_folder.py --config configs/evaluate_on_facescape.yaml --ckpt assets/ckpts/facescape/780000.ckpt --out outputs/facescape/keypointnerf_780000 --model KeypointNeRF
