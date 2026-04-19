# In-Run Data Shapley for Graph Autoencoders

#### This repository adapts in-run gradient dot product methods to graph autoencoders for subgraph evaluation.


## Introduction
This repository is built off of the [GhostSuite](https://github.com/Jiachen-T-Wang/GhostSuite/tree/v0.33) repository, which implements in-run gradient dot product techniques for data valuation in large language models.

We extend these ideas to graph autoencoders (GAEs), where training examples correspond to nodes or subgraphs instead. This repository adapts the ghost dot product machinery to support this new model while preserving the original structure of GhostSuite.