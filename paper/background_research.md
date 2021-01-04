# HDARTS Background Research

## Theory-Inspired Path-Regularized Differential Network Architecture Search

https://proceedings.neurips.cc/paper/2020/file/5e1b18c4c6a6d31695acbae3fd70ecc6-Paper.pdf

- Tackles problem of too many skip-connections leading to poor performance of learnt networks
- Analyzes theoretical reasoning behind this bias and fixes it
- Proves that when optimizing $F_{train}(W, \beta)$, the convergence rate at each iteration depends on the weights of skip connections much heavier than other types of operations ($\beta$ in this paper is $\alpha$ from DARTS)
- Proposes Path Regularized DARTS which uses **group-structured sparsity penalizes the skip connection group heavier than another group to rectify the competitive advantage of skip connections** (don't understand this yet, need to read further)

## Differentiable Neural Architecture Search in Equivalent Space with Exploration Enhancement

- Theoretical approach to ensuring that the differential
- Fixes the rich-get-richer problem: architectures with better performance early on trained more frequently, the updated weights lead to higher probability of sampling which often leads to local optima
- Uses a **variational graph autoencoder** (need to read about this) to injectively transform the discrete architecture space into an equivalently continuous latent space 

## Other Papers

- https://openreview.net/forum?id=PKubaeJkw3
- https://arxiv.org/pdf/2010.13501.pdf

- https://openaccess.thecvf.com/content_ICCVW_2019/papers/NeurArch/Yan_HMNAS_Efficient_Neural_Architecture_Search_via_Hierarchical_Masking_ICCVW_2019_paper.pdf

- https://openreview.net/forum?id=PKubaeJkw3

## Ideas So Far

- Argue about weight sharing for alpha being useful while not running into same problems as conventional weight sharing -> avoid catastrophic forgetting 
- Issue of lack of correspondance between rich get richer and 
- Hessian norm - related to argmax - argue hierarchy ends up doing perturbation https://openreview.net/forum?id=PKubaeJkw3
- Larger search space 
- Argument for wider networks that converge faster (why is this the case?)
