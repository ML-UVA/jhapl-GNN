# Axon-Dendrite Proximity and Co-Travel Distance

## Setup
This repository contains tools and analyses for analyzing the Axon-Dendrite Proxi

Axon-Dendrite proximity (ADP) represents how cloes an axon from one neuron gets to a dendrite of another neuron in 3D space.

This folder provides the tools and pipelines to efficiently calculate the ADP between pair of neurons given a large dataset efficiently.

The analyses assume that neuron morphologies are stored as NetworkX graph objects located in data/graph_exports/. These graphs encode morphological structure along with additional node- and edge-level features. These graph objects were stored as .pbz2 files.

Ex: 86491134110093308_0_auto_proof_v7_proofread.pbz2

The datasets used to develop and validate these analyses were sourced from the MICrONS project and processed through NEURD for automated proofreading and structural refinement.

ADP_Calculation_Full.py includes the CLI necessary to run full ADP metric calculations given a graph path (Where 3D neuron data is stored) and a data path (where outputs should be stored). This calculation does the following:

1. Generates skeletonized representations of neurons (assumes 1micron distance between each point in the 3D graph)
2. Builds KD-tree representations of each skeletonized representation for efficient pairwise search.
3. Constructs and voxelizes 3D space to calculate pairwise ADP between each neuron in a python dictionary.
4. Outputs list of common metrics associated with the measurements.
