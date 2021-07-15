import unittest
from mixed_operation import MixedOperation
from util import gumbel_softmax
import torch
from matplotlib import pyplot as plt
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_gumbel_softmax(self):
        temps = [0.1, 1, 5, 10]
        vec = torch.Tensor([2, 5])
        n_trials = 100
        temp_freqs = np.zeros((len(temps), len(vec)))
        for i, temp in enumerate(temps):
            inds = torch.argmax(gumbel_softmax(vec.repeat(n_trials, 1), temp), dim=-1)
            temp_freqs[i] = (torch.sum(torch.eye(len(vec))[inds], dim=0) / n_trials).numpy()

        print(temp_freqs[0])

        # plot the results
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
        for i, ax in enumerate(axs.flat):
            ax.bar(vec, temp_freqs[i])
            ax.set_title(f"temp = {temps[i]}")
            ax.set_ylabel("frequency")
            ax.set_xlabel("element value")
        plt.show()



if __name__ == '__main__':
    unittest.main()
