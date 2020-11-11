# Hierarchical DARTS - Neural Architecture Search (NAS)

## About

This is a repository for our work for CS 269 Research Seminar on Efficient Machine Learning taught by Professor Baharan Mirzasoleiman

This repository contains an implementation of Hierarchical DARTS (HDARTS) - a novel algorithm that combines the ideas of hierarchical search spaces for NAS from [Hierarchical Representations for Efficient Architecture Search](https://arxiv.org/abs/1711.00436) and differentiable architecture search from [DARTS](https://arxiv.org/abs/1806.09055).

The mathematical notation and the actual algorithm for HDARTS is detailed here at https://www.overleaf.com/project/5f9fdb9eacb45b000164049e.

## Design

- Classes that represent the composable representations - allow for G_ij(G_kl)) - sort of notation
- Class to get alpha_i
- Class to compute gradient
- Class to visualize 

## Results

| Dataset  | Test Accuracy  |  GPU Time (Days) |
|---|---|---|
| MNIST |   |   |
