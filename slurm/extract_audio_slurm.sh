#!/bin/bash
#SBATCH --account=your_project_name
#SBATCH --partition=interactive
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --error=/scratch/your_project_name/logs/extract_audio_%j.log

set -e

bash /scratch/your_project_name/codes/fusion-avsr/scripts/extract_audio.sh \
  /scratch/your_project_name/datasets/lrs3/ainncy/trainval \
  /scratch/your_project_name/datasets/extracted_audio \
  8

bash /scratch/your_project_name/codes/fusion-avsr/scripts/extract_audio.sh \
  /scratch/your_project_name/datasets/lrs3/test \
  /scratch/your_project_name/datasets/extracted_audio \
  8

bash /scratch/your_project_name/codes/fusion-avsr/scripts/extract_audio.sh \
  /scratch/your_project_name/datasets/kaggle_lipnet/datasets/jedidiahangekouakou/grid-corpus-dataset-for-training-lipnet/versions/1/data \
  /scratch/your_project_name/datasets/extracted_audio \
  8