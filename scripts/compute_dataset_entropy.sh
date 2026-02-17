#!/bin/bash
# Script to compute entropy of a dataset

export HYDRA_FULL_ERROR=1

# Default values
DATA=${DATA:-openwebtext}
MAX_SAMPLES=${MAX_SAMPLES:-}
MODEL_LENGTH=${MODEL_LENGTH:-}

# Build command
CMD="python -u compute_dataset_entropy.py data.train=${DATA}"

# Add model.length if specified (needed to match cached dataset)
if [ ! -z "$MODEL_LENGTH" ]; then
    CMD="$CMD model.length=${MODEL_LENGTH}"
fi

# Add max_samples if specified
if [ ! -z "$MAX_SAMPLES" ]; then
    CMD="$CMD max_samples=${MAX_SAMPLES}"
fi

echo "Computing entropy for dataset: ${DATA}"
if [ ! -z "$MODEL_LENGTH" ]; then
    echo "Using model.length=${MODEL_LENGTH} (to match cached dataset)"
fi
if [ ! -z "$MAX_SAMPLES" ]; then
    echo "Using first ${MAX_SAMPLES} samples"
fi
echo "Command: $CMD"
echo ""

# Run the command
$CMD

