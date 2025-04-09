#!/bin/bash

# Display usage instructions if no arguments provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <path_to_checkpoint>"
    echo "Example: $0 /path/to/your/model.ckpt"
    echo ""
    echo "This script converts a PyTorch Lightning checkpoint to SafeTensors format."
    echo "It first consolidates the checkpoint and then converts it to SafeTensors."
    exit 1
fi

# Set the path to the checkpoint from the first argument
CHECKPOINT_PATH="$1"

# Define the consolidated checkpoint output path
CONSOLIDATED_PATH="${CHECKPOINT_PATH%.*}_consolidated.ckpt"

# Consolidate the checkpoint
echo "Consolidating checkpoint..."
python -m lightning.pytorch.utilities.consolidate_checkpoint "$CHECKPOINT_PATH" --output_file "$CONSOLIDATED_PATH"

# Convert the consolidated checkpoint to safety tensors
echo "Converting to SafeTensors format..."
python ./src/genmo/mochi_preview/save_safetytensors.py --path "$CONSOLIDATED_PATH"

# Clean up consolidated checkpoint
echo "Cleaning up temporary files..."
rm -rf "$CONSOLIDATED_PATH"

echo "Conversion complete!"