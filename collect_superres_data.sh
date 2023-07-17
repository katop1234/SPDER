#!/bin/bash

# Collect the model type from the command line argument
MODEL_TYPE=$1

# Define your datasets
DATASETS=("DIV2K" "ffhq0" "ffhq1" "ffhq2" "Flickr2K_Train")

# Iterate over each dataset
for DATASET in "${DATASETS[@]}"; do
    # Call your Python script with the current arguments
    python SPDER_superres.py $DATASET $MODEL_TYPE
    done
done
