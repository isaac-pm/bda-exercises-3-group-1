#!/bin/bash -l
#SBATCH --job-name=BDA_LSA_Grid
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/problem_1/lsa_%j.out
#SBATCH --error=logs/problem_1/lsa_%j.err

set -euo pipefail

# Setup
module purge

module load devel/Spark/3.5.4-foss-2023b-Java-17
module load lang/Python/3.11.5-GCCcore-13.2.0


# IMPORTANT:
# Dataset must be EXTRACTED beforehand
#WORK_DIR="/home/users/$USER/uni/2_semester/big_data_analytics/assignment3"
WORK_DIR=$SLURM_SUBMIT_DIR/solutions/problem_1
LOG_DIR="$SLURM_SUBMIT_DIR/logs/problem_1"
DATA_DIR=$SLURM_SUBMIT_DIR/data/problem_1/Wikipedia-En-41784-Articles
STOPWORDS_PATH="$SLURM_SUBMIT_DIR/data/problem_1/stopwords.txt"
SCRIPT_PATH="$WORK_DIR/RunLSA.py"

BASE_DIR="/scratch/users/$USER/big_data_analytics/bda-exercises-3-group-1"
NLTK_DIR="$BASE_DIR/nltk_data"
VENV_DIR="$BASE_DIR/nlp_env"
RESULTS_DIR="$BASE_DIR/results"


mkdir -p "$RESULTS_DIR"
mkdir -p "$LOG_DIR"

# ============================================================
# 3. PYTHON ENVIRONMENT
# ============================================================

source "$VENV_DIR/bin/activate"

# ============================================================
# 4. SPARK ENVIRONMENT
# ============================================================

export PYSPARK_PYTHON="$VENV_DIR/bin/python"

export PYTHONPATH="$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9.7-src.zip:$PYTHONPATH"

export NLTK_DATA="$NLTK_DIR"

CORES=${SLURM_CPUS_PER_TASK:-16}

echo "============================================================"
echo "LSA GRID SEARCH STARTED"
echo "============================================================"

echo "WORK_DIR: $WORK_DIR"
echo "BASE_DIR: $BASE_DIR"

echo "SPARK_HOME: $SPARK_HOME"
echo "PYSPARK_PYTHON: $PYSPARK_PYTHON"

echo "DATA_DIR: $DATA_DIR"
echo "STOPWORDS_PATH: $STOPWORDS_PATH"
echo "NLTK_DIR: $NLTK_DIR"

echo "CORES: $CORES"

echo "============================================================"

# ============================================================
# 5. SANITY CHECKS
# ============================================================

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: RunLSA.py not found"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Wikipedia dataset directory not found"
    echo "Expected: $DATA_DIR"
    exit 1
fi

if [ ! -f "$STOPWORDS_PATH" ]; then
    echo "ERROR: stopwords.txt not found"
    exit 1
fi

if [ ! -d "$NLTK_DIR" ]; then
    echo "ERROR: nltk_data directory not found"
    exit 1
fi

# ============================================================
# 6. RUNTIME CSV HEADER
# ============================================================

RUNTIME_CSV="$RESULTS_DIR/runtime_summary.csv"

if [ ! -f "$RUNTIME_CSV" ]; then
    echo "run_name,tokenizer,numFreq,k,runtime_seconds" \
        > "$RUNTIME_CSV"
fi

# ============================================================
# 7. GRID SEARCH EXPERIMENTS
# ============================================================

for TOKENIZER in regex nlp
do
    for NUM_FREQ in 5000 10000 20000
    do
        for K_VAL in 25 100 250
        do

            RUN_NAME="${TOKENIZER}_freq${NUM_FREQ}_k${K_VAL}"

            OUTPUT_FILE="$RESULTS_DIR/${RUN_NAME}.txt"

            echo ""
            echo "========================================================"
            echo "RUN: $RUN_NAME"
            echo "========================================================"

            START_TIME=$(date +%s)

            python "$SCRIPT_PATH" \
                --input "$DATA_DIR" \
                --stopwords "$STOPWORDS_PATH" \
                --nltk_data "$NLTK_DIR" \
                --tokenizer "$TOKENIZER" \
                --numFreq "$NUM_FREQ" \
                --k "$K_VAL" \
                --sampleSize 1.0 \
                > "$OUTPUT_FILE" 2>&1

            END_TIME=$(date +%s)

            RUNTIME=$((END_TIME - START_TIME))

            echo ""
            echo "Completed: $RUN_NAME"
            echo "Runtime: ${RUNTIME} seconds"

            echo "${RUN_NAME},${TOKENIZER},${NUM_FREQ},${K_VAL},${RUNTIME}" \
                >> "$RUNTIME_CSV"

        done
    done
done

# ============================================================
# 8. SEARCH ENGINE QUERIES
# ============================================================

echo ""
echo "========================================================"
echo "RUNNING SEARCH ENGINE TESTS"
echo "========================================================"

BEST_NUM_FREQ=20000
BEST_K=250

# ------------------------------------------------------------

python "$SCRIPT_PATH" \
    --input "$DATA_DIR" \
    --stopwords "$STOPWORDS_PATH" \
    --nltk_data "$NLTK_DIR" \
    --tokenizer nlp \
    --numFreq $BEST_NUM_FREQ \
    --k $BEST_K \
    --query "artificial intelligence neural networks machine learning" \
    > "$RESULTS_DIR/query_ai.txt" 2>&1

# ------------------------------------------------------------

python "$SCRIPT_PATH" \
    --input "$DATA_DIR" \
    --stopwords "$STOPWORDS_PATH" \
    --nltk_data "$NLTK_DIR" \
    --tokenizer nlp \
    --numFreq $BEST_NUM_FREQ \
    --k $BEST_K \
    --query "quantum physics relativity particle theory" \
    > "$RESULTS_DIR/query_quantum.txt" 2>&1

# ------------------------------------------------------------

python "$SCRIPT_PATH" \
    --input "$DATA_DIR" \
    --stopwords "$STOPWORDS_PATH" \
    --nltk_data "$NLTK_DIR" \
    --tokenizer nlp \
    --numFreq $BEST_NUM_FREQ \
    --k $BEST_K \
    --query "ancient rome empire caesar senate" \
    > "$RESULTS_DIR/query_rome.txt" 2>&1

# ------------------------------------------------------------

python "$SCRIPT_PATH" \
    --input "$DATA_DIR" \
    --stopwords "$STOPWORDS_PATH" \
    --nltk_data "$NLTK_DIR" \
    --tokenizer nlp \
    --numFreq $BEST_NUM_FREQ \
    --k $BEST_K \
    --query "football world cup fifa soccer" \
    > "$RESULTS_DIR/query_football.txt" 2>&1

# ------------------------------------------------------------

python "$SCRIPT_PATH" \
    --input "$DATA_DIR" \
    --stopwords "$STOPWORDS_PATH" \
    --nltk_data "$NLTK_DIR" \
    --tokenizer nlp \
    --numFreq $BEST_NUM_FREQ \
    --k $BEST_K \
    --query "biology genetics dna evolution" \
    > "$RESULTS_DIR/query_biology.txt" 2>&1

# ============================================================
# 9. CLEANUP
# ============================================================

deactivate

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETED"
echo "============================================================"

echo "Results stored in:"
echo "$RESULTS_DIR"

echo ""
echo "Runtime summary:"
echo "$RUNTIME_CSV"

echo "============================================================"