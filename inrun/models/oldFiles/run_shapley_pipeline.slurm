#!/bin/bash
# ================================================================
#  JHU APL GNN CONNECTOME PIPELINE
#  Runs: GAE training -> Shapley -> Normalize -> Motif extraction
#        -> Null graph comparison
#
#  USAGE:
#    sbatch run_pipeline.sh
#
#  To override config from the command line without editing:
#    sbatch --export=ALL,EPOCHS=500,LATENT_DIM=32 run_pipeline.sh
# ================================================================

#SBATCH --job-name=connectome_gnn
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your@email.edu        # <-- set your email here

# ================================================================
#  CONFIGURATION — edit this block before submitting
# ================================================================

# --- Paths ---
CSV_PATH="${CSV_PATH:-data/top5_k1.csv}"
FEATURE_PATH="${FEATURE_PATH:-data/neuron_features.pt}"
NULL_DIR="${NULL_DIR:-data/null_graphs}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"

# --- GAE model variant ---
# Mix and match flags:
#   ""                        standard GAE,  2-layer GCN encoder
#   "--variational"           VGAE,          2-layer GCN encoder
#   "--linear"                GAE,           single linear layer
#   "--variational --linear"  VGAE,          single linear layer
MODEL_FLAGS="${MODEL_FLAGS:-}"

# --- GAE hyperparameters ---
EPOCHS="${EPOCHS:-200}"
LATENT_DIM="${LATENT_DIM:-16}"

# --- Motif extraction ---
MOTIF_SIZES="${MOTIF_SIZES:-5 10 20}"   # space-separated list of subgraph sizes
TOP_K="${TOP_K:-3}"                      # how many top motifs to extract per size

# --- Null model files (filenames only, inside NULL_DIR) ---
NULL_FILES="${NULL_FILES:-null_graph_0.csv null_graph_1.csv null_graph_2.csv null_graph_3.csv}"

# ================================================================
#  ENVIRONMENT SETUP — uncomment whichever applies to your cluster
# ================================================================

# source activate myenv
# module load python/3.10
# source /p/mlatuva/jhu-graph/sidhya/venv/bin/activate

# ================================================================
#  BOILERPLATE
# ================================================================

set -e   # exit immediately if any step fails
set -o pipefail

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "  JHU APL GNN PIPELINE"
echo "  Job ID     : ${SLURM_JOB_ID:-local}"
echo "  Node       : $(hostname)"
echo "  Start time : $(date)"
echo "=============================================="
echo ""
echo "CONFIG:"
echo "  CSV_PATH     = ${CSV_PATH}"
echo "  FEATURE_PATH = ${FEATURE_PATH}"
echo "  MODEL_FLAGS  = '${MODEL_FLAGS}'"
echo "  EPOCHS       = ${EPOCHS}"
echo "  LATENT_DIM   = ${LATENT_DIM}"
echo "  MOTIF_SIZES  = ${MOTIF_SIZES}"
echo "  TOP_K        = ${TOP_K}"
echo "  OUTPUT_DIR   = ${OUTPUT_DIR}"
echo ""

# ================================================================
#  STEP 1 — Train GAE and compute raw Shapley values
# ================================================================

echo "----------------------------------------------"
echo "STEP 1: Training GAE + computing Shapley values"
echo "----------------------------------------------"

python models/node_graph_autoencoder.py \
    --epochs        "${EPOCHS}"      \
    --latent_dim    "${LATENT_DIM}"  \
    --csv_path      "${CSV_PATH}"    \
    --feature_path  "${FEATURE_PATH}" \
    --output_path   "${OUTPUT_DIR}/graph_node_shapley.value" \
    ${MODEL_FLAGS}

echo "STEP 1 done -> ${OUTPUT_DIR}/graph_node_shapley.value"
echo ""

# ================================================================
#  STEP 2 — Normalize Shapley values
# ================================================================

echo "----------------------------------------------"
echo "STEP 2: Normalizing Shapley values"
echo "----------------------------------------------"

python models/normalize.py \
    --input_path  "${OUTPUT_DIR}/graph_node_shapley.value" \
    --output_path "${OUTPUT_DIR}/graph_node_shapley_normalized.value"

echo "STEP 2 done -> ${OUTPUT_DIR}/graph_node_shapley_normalized.value"
echo ""

# ================================================================
#  STEP 3 — Extract motifs and visualize
# ================================================================

echo "----------------------------------------------"
echo "STEP 3: Extracting motifs and visualizing"
echo "----------------------------------------------"

python models/motifs.py \
    --csv_path     "${CSV_PATH}"    \
    --shapley_path "${OUTPUT_DIR}/graph_node_shapley_normalized.value" \
    --motif_sizes  ${MOTIF_SIZES}  \
    --top_k        "${TOP_K}"      \
    --output_dir   "${OUTPUT_DIR}"

echo "STEP 3 done -> motif PNGs + subgraph_results.pkl in ${OUTPUT_DIR}/"
echo ""

# ================================================================
#  STEP 4 — Null graph comparison (over/under expression)
# ================================================================

echo "----------------------------------------------"
echo "STEP 4: Null graph comparison"
echo "----------------------------------------------"

# Prepend null dir to each filename
NULL_PATHS=""
for f in ${NULL_FILES}; do
    NULL_PATHS="${NULL_PATHS} ${NULL_DIR}/${f}"
done

python models/null_comparison.py \
    --real_csv         "${CSV_PATH}"                          \
    --subgraph_results "${OUTPUT_DIR}/subgraph_results.pkl"   \
    --null_paths       ${NULL_PATHS}                          \
    --output_dir       "${OUTPUT_DIR}"

echo "STEP 4 done -> null comparison results in ${OUTPUT_DIR}/"
echo ""

# ================================================================
#  DONE
# ================================================================

echo "=============================================="
echo "  PIPELINE COMPLETE"
echo "  End time : $(date)"
echo "  All outputs in: ${OUTPUT_DIR}/"
echo "=============================================="