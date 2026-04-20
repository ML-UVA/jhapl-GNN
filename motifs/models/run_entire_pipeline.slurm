#!/bin/bash
#Runs: Graph construction, GAE + Shapley, Normalize, Motifs
#
#How to run: 
#    sbatch run_pipeline.sh
#
#How to override from command line:
#    sbatch --export=ALL,EPOCHS=500,LATENT_DIM=32 run_pipeline.sh

#SBATCH --job-name=connectome_pipeline
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00


#Set USE_EXISTING=true to use the existing top5_k1.csv
#Set USE_EXISTING=false to build a new graph from synapses_with_features.pt
#using the spatial thresholds below
USE_EXISTING="${USE_EXISTING:-true}"
EXISTING_CSV="${EXISTING_CSV:-data/top5_k1.csv}"

#Input files (only used if USE_EXISTING=false)
SYNAPSES_PATH="${SYNAPSES_PATH:-data/processed/synapses_with_features.pt}"
POSITIONS_PATH="${POSITIONS_PATH:-data/processed/positions.pt}"

#Spatial thresholds (only used if USE_EXISTING=false) ---
#Leave empty to skip that axis
X_MIN="${X_MIN:-}"
X_MAX="${X_MAX:-}"
Y_MIN="${Y_MIN:-}"
Y_MAX="${Y_MAX:-}"
Z_MIN="${Z_MIN:-}"
Z_MAX="${Z_MAX:-}"

#node features
FEATURE_PATH="${FEATURE_PATH:-data/processed/neuron_features.pt}"

MODEL_FLAGS="${MODEL_FLAGS:-}"
EPOCHS="${EPOCHS:-200}"
LATENT_DIM="${LATENT_DIM:-16}"
LR="${LR:-0.01}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"

MOTIF_SIZES="${MOTIF_SIZES:-5 10 20}"
TOP_K="${TOP_K:-3}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/motifs}"


source ~/.bashrc
module load miniforge/25.3.1-py3.12
conda activate condaEnv

# Run from repo root so `python -m motifs` and default data/ paths resolve.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

echo "  JHU APL NODE SHAPLEY PIPELINE"
echo "  Job ID     : ${SLURM_JOB_ID:-local}"
echo "  Node       : $(hostname)"
echo "  Start time : $(date)"
echo "=============================================="
echo ""
echo "CONFIG:"
echo "  USE_EXISTING  = ${USE_EXISTING}"
echo "  EXISTING_CSV  = ${EXISTING_CSV}"
echo "  MODEL_FLAGS   = '${MODEL_FLAGS}'"
echo "  EPOCHS        = ${EPOCHS}"
echo "  LATENT_DIM    = ${LATENT_DIM}"
echo "  MOTIF_SIZES   = ${MOTIF_SIZES}"
echo "  TOP_K         = ${TOP_K}"
echo "  OUTPUT_DIR    = ${OUTPUT_DIR}"

CMD="python -m motifs \
    --feature_path  ${FEATURE_PATH} \
    --epochs        ${EPOCHS} \
    --latent_dim    ${LATENT_DIM} \
    --lr            ${LR} \
    --save_interval ${SAVE_INTERVAL} \
    --motif_sizes   ${MOTIF_SIZES} \
    --top_k         ${TOP_K} \
    --output_dir    ${OUTPUT_DIR} \
    ${MODEL_FLAGS}"


if [ "${USE_EXISTING}" = "true" ]; then
    CMD="${CMD} --use_existing --existing_csv ${EXISTING_CSV}"
else
    CMD="${CMD} \
        --synapses_path  ${SYNAPSES_PATH} \
        --positions_path ${POSITIONS_PATH}"

    [ -n "${X_MIN}" ] && CMD="${CMD} --x_min ${X_MIN}"
    [ -n "${X_MAX}" ] && CMD="${CMD} --x_max ${X_MAX}"
    [ -n "${Y_MIN}" ] && CMD="${CMD} --y_min ${Y_MIN}"
    [ -n "${Y_MAX}" ] && CMD="${CMD} --y_max ${Y_MAX}"
    [ -n "${Z_MIN}" ] && CMD="${CMD} --z_min ${Z_MIN}"
    [ -n "${Z_MAX}" ] && CMD="${CMD} --z_max ${Z_MAX}"
fi

echo "Running: ${CMD}"
echo ""

eval ${CMD}

echo "  Pipeline Complete"
echo "  End time : $(date)"
echo "  Outputs  : ${OUTPUT_DIR}/"
