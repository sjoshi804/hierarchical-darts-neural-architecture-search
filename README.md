# Hierarchical DARTS - Neural Architecture Search (NAS)

## About

This is a repository for our work for CS 269 Research Seminar on Efficient Machine Learning taught by Professor Baharan Mirzasoleiman

This repository contains an implementation of Hierarchical DARTS (HDARTS) - a novel algorithm that combines the ideas of hierarchical search spaces for NAS from [Hierarchical Representations for Efficient Architecture Search](https://arxiv.org/abs/1711.00436) and differentiable architecture search from [DARTS](https://arxiv.org/abs/1806.09055).

The mathematical notation and the actual algorithm for HDARTS is detailed here at https://www.overleaf.com/project/5f9fdb9eacb45b000164049e.

## Design

- Alpha - class that encapsulates the architecture parameters
- Model - class that is used to instantiate a network from a given Alpha and Primitives

## Results

| Dataset  | Test Accuracy  |  GPU Time (Days) |
|---|---|---|
| MNIST |   |   |


Done: Make every inner most value in alpha a nn.Parameter (Alpha)

Done: Find a new way to get alpha (ModelController) that conforms with - Did this by registering the self.alphas as a concatenation of all the alphas for each level and then for the get_alpha_level(level) function of model controller using the alpha object (self.alpha.get_alpha_level(level))

Done: Instead of constructing network on init, construct on .setAlpha (Model) - assuming this done because the mixed operation forward operation is the only bit tha actually depends on the actual values of the alpha parameters and would be called inside the higher level call to forward

TODO: Investigate tensorboard plot inside mixed op first and then at higher levels