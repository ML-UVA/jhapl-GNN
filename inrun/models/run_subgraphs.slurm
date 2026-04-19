#!/bin/bash
#SBATCH --partition=cpu
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=500G
#SBATCH --time=30:00:00
#SBATCH --output=subgraph1.out
#SBATCH --error=subgraph1.err

module load miniforge
conda activate condaEnv

cd /p/mlatuva/jhu-graph/sidhya/jhapl-GNN/inrun/models

/u/gyz8db/.conda/envs/edgegnn/bin/python run_subgraph_analysis.py
