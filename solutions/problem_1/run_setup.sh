#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e 

echo "--- Creating shared directories ---"
BASE_DIR="/scratch/users/$USER/big_data_analytics/bda-exercises-3-group-1"

mkdir -p "$BASE_DIR/data"
mkdir -p "$BASE_DIR/nltk_data"


echo "--- Loading Python module ---"
module purge
module load lang/Python/3.11.5-GCCcore-13.2.0

echo "--- Building virtual environment ---"

VENV_DIR="$BASE_DIR/nlp_env"

if [ ! -d "$VENV_DIR" ]; then
    python -m venv $VENV_DIR
    echo "Virtual environment created at $VENV_DIR"
else
    echo "Virtual environment already exists at $VENV_DIR. Proceeding..."
fi

echo "---  Activating environment and installing dependencies ---"
source $VENV_DIR/bin/activate
pip install --upgrade pip
pip install nltk scipy numpy

echo "--- Downloading NLTK models ---"
# This prevents the spark executors from downloading data during the Map tasks
python -c "import nltk; nltk.download('punkt', download_dir='$BASE_DIR/nltk_data'); nltk.download('wordnet', download_dir='$BASE_DIR/nltk_data')"

echo "---  Cleanup ---"
deactivate
echo "Environment setup complete! Your Spark executors will now use this isolated environment."