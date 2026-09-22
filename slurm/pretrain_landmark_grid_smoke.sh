#!/bin/bash
#SBATCH --account=your_project_name
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/scratch/your_project_name/logs/pretrain_grid_smoke_%j.log

# Smoke test: 200 clips, 2 epochs. There is no real per-epoch timing yet
# for this model, so run this first and read its log's real wall-clock
# time before guessing --time= for the full run
# (slurm/pretrain_landmark_grid.sh).

module purge
export PATH="/projappl/your_project_name/env/fusion-avsr-gpu-env/bin:$PATH"

set -e

python scripts/pretrain_landmark_grid.py \
  grid_root=/scratch/your_project_name/datasets/kaggle_lipnet/datasets/jedidiahangekouakou/grid-corpus-dataset-for-training-lipnet/versions/1/data \
  landmarks_root=/scratch/your_project_name/datasets/grid_landmarks \
  audio_output_dir=/scratch/your_project_name/datasets/extracted_audio \
  limit=200 \
  num_epochs=2 \
  checkpoint_dir=/scratch/your_project_name/checkpoints/landmark_pretrain_smoke
