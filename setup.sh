#!/bin/bash

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda could not be found. Please install Anaconda or Miniconda first."
    exit
fi

# Create the conda environment from the aml file
echo "Creating the conda environment..."
conda env create -f project2.yml

# Activate the environment
echo "Activating the conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate project2

echo "Environment setup complete. You can now run your Jupyter Notebook with 'jupyter notebook'."
