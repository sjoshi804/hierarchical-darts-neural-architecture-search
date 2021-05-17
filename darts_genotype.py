from alpha import Alpha
import torch
import util

alpha_norm = Alpha(1, {0: 7}, {0: 8})
alpha_reduce = Alpha(1, {0: 7}, {0: 8})

for edge in alpha_norm.parameters[0][0]:
    alpha_norm.parameters[0][0][edge].requires_grad = False
for edge in alpha_reduce.parameters[0][0]:
    alpha_reduce.parameters[0][0][edge].requires_grad = False

# Set to DARTS Alpha Normal
alpha_norm.parameters[0][0][(0,2)][2] = 1
alpha_norm.parameters[0][0][(0,3)][2] = 1
alpha_norm.parameters[0][0][(0,4)][2] = 1
alpha_norm.parameters[0][0][(1,2)][2] = 1
alpha_norm.parameters[0][0][(1,3)][2] = 1
alpha_norm.parameters[0][0][(1,4)][8] = 1
alpha_norm.parameters[0][0][(1,5)][8] = 1
alpha_norm.parameters[0][0][(2,5)][5] = 1

# Set to DARTS Alpha Reduce
alpha_reduce.parameters[0][0][(0,2)][1] = 1
alpha_reduce.parameters[0][0][(0,4)][1] = 1
alpha_reduce.parameters[0][0][(1,2)][1] = 1
alpha_reduce.parameters[0][0][(1,3)][1] = 1
alpha_reduce.parameters[0][0][(1,5)][1] = 1
alpha_reduce.parameters[0][0][(2,3)][8] = 1
alpha_reduce.parameters[0][0][(2,4)][8] = 1
alpha_reduce.parameters[0][0][(2,5)][8] = 1

util.save_object(alpha_norm, "checkpoints_search/17-05-2021--12-23-29/best/alpha_normal.pkl")
util.save_object(alpha_reduce, "checkpoints_search/17-05-2021--12-23-29/best/alpha_reduce.pkl")
