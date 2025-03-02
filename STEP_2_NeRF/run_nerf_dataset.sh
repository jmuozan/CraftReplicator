#!/bin/bash

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install it before running this script."
    exit 1
fi

# Default values
IMAGE_DIR="./db"
MASK_DIR="./segmentation_output/masks"
OUTPUT_DIR="./nerf_dataset"
IMAGE_SCALE="1.0"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the dataset creation script
python3 nerf_dataset_creator.py \
    --image_dir "$IMAGE_DIR" \
    --mask_dir "$MASK_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --image_scale "$IMAGE_SCALE"

echo "Script execution completed."