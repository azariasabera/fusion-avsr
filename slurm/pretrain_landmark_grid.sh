#!/bin/bash
#SBATCH --account=your_project_name
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:1
#SBATCH --time=00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/scratch/your_project_name/logs/pretrain_grid_%j.log

# TODO: --time=00:00:00 above is a deliberate placeholder, not a real
# estimate -- SLURM will reject it. Run slurm/pretrain_landmark_grid_smoke.sh
# first, read its log for real per-epoch wall-clock time, then set this to
# (per-epoch time) x (num_epochs, or early_stopping_patience's likely
# worst case) plus a real buffer, before submitting this job.

# Full run: no limit=, so the whole GRID dataset; num_epochs and
# early_stopping_patience are left at their config defaults (see
# configs/scripts/pretrain_landmark_grid.yaml) so early stopping decides
# the real endpoint, not a fixed wall-clock guess.

module purge
export PATH="/projappl/your_project_name/env/fusion-avsr-gpu-env/bin:$PATH"

set -e

python scripts/pretrain_landmark_grid.py \
  grid_root=/scratch/your_project_name/datasets/kaggle_lipnet/datasets/jedidiahangekouakou/grid-corpus-dataset-for-training-lipnet/versions/1/data \
  landmarks_root=/scratch/your_project_name/datasets/grid_landmarks \
  audio_output_dir=/scratch/your_project_name/datasets/extracted_audio \
  checkpoint_dir=/scratch/your_project_name/checkpoints/landmark_pretrain
