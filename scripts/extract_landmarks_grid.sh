#!/bin/bash
#SBATCH --account=your_project_name
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --error=/scratch/your_project_name/logs/extract_landmarks_grid_%j.log

set -e

python /scratch/your_project_name/codes/fusion-avsr/scripts/extract_landmarks_grid.py \
  grid_root=/scratch/your_project_name/datasets/kaggle_lipnet/datasets/jedidiahangekouakou/grid-corpus-dataset-for-training-lipnet/versions/1/data \
  landmarks_root=/scratch/your_project_name/datasets/grid_landmarks \
  device=cuda:0