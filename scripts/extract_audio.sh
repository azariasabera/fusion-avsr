#!/usr/bin/env bash
#
# Extract 16kHz mono .wav audio from GRID (.mpg) or LRS3 trainval/test (.mp4)
# video files, in parallel, via ffmpeg + xargs.
#
# Usage:
#   scripts/extract_audio.sh <dataset_root> <output_dir> <num_jobs>
#
# <dataset_root> depends on which dataset you are extracting:
#
#   - LRS3 (trainval or test): point at .../ainncy/trainval or .../lrs3/test
#     (the directory containing one <video_id>/ subdirectory per video, each holding
#     that video's <clip_id>.mp4 clips). Output files are named
#     <video_id>_<clip_id>.wav.
#
#   - GRID: point at the GRID data root (the directory containing one
#     s<N>_processed/ subdirectory per speaker, each holding that
#     speaker's <clip_id>.mpg clips directly). Output files are named
#     <speaker>_<clip_id>.wav, where <speaker> is the s<N>_processed
#     directory name with the "_processed" suffix stripped.
#
# This script auto-detects which of the two layouts it is looking at by
# searching for .mp4 vs .mpg files under <dataset_root> -- run it once
# per dataset (it does not mix LRS3 and GRID clips in one run).
#
# Already-extracted files (an output .wav already present at the
# expected path) are skipped, so a large batch job can be safely resumed
# after an interruption.
#
# These output filenames MUST exactly match the sample_id convention
# used by fusion_avsr/data/manifest_builder.py -- the manifest builder
# does not run ffmpeg itself, it only looks for .wav files already
# written by this script.

set -euo pipefail

DATASET_ROOT="$1"
OUTPUT_DIR="$2"
NUM_JOBS="$3"

mkdir -p "$OUTPUT_DIR"

extract_one() {
    input_path="$1"
    output_dir="$2"
    ext="${input_path##*.}"

    if [ "$ext" = "mp4" ]; then
        clip_id=$(basename "$input_path" .mp4)
        video_id=$(basename "$(dirname "$input_path")")
        sample_id="${video_id}_${clip_id}"
    else
        clip_id=$(basename "$input_path" .mpg)
        speaker_dir_name=$(basename "$(dirname "$input_path")")
        speaker="${speaker_dir_name%_processed}"
        sample_id="${speaker}_${clip_id}"
    fi

    output_path="${output_dir}/${sample_id}.wav"
    if [ ! -f "$output_path" ]; then
        if ffmpeg -y -loglevel error -i "$input_path" -vn -ac 1 -ar 16000 "$output_path" 2>> "${output_dir}/extract_errors.log"; then
            echo "$sample_id" >> "${output_dir}/extract_progress.log"
        else
            echo "$sample_id FAILED: $input_path" >> "${output_dir}/extract_errors.log"
        fi
    fi
}
export -f extract_one

# Exactly one of the two patterns will match, depending on which dataset
# DATASET_ROOT points at.
find "$DATASET_ROOT" \( -name "*.mp4" -o -name "*.mpg" \) -print0 \
    | xargs -0 -P "$NUM_JOBS" -I{} bash -c 'extract_one "$1" "$2"' _ {} "$OUTPUT_DIR"

echo "Done. Extracted audio written to $OUTPUT_DIR"
