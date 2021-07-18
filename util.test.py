import unittest
from mixed_operation import MixedOperation
from util import gumbel_softmax
import torch
from matplotlib import pyplot as plt
import numpy as np


class MyTestCase(unittest.TestCase):
    def test_gumbel_softmax(self):
        temps = [0.01, 0.33, 1, 10]
        original_dist = torch.Tensor([0.2, 0.8])
        n_trials = 100
        data = []

        for temp in temps:
            sample = []
            for _ in range(n_trials):
                gumbel_dist = torch.distributions.Categorical(gumbel_softmax(original_dist, temp))
                for _ in range(100):
                    sample.append(int(gumbel_dist.sample()))
            data.append(sample)

        # plot the results
        fig, axs = plt.subplots(2, 2)
        for i in range(2):
            for j in range(2):
                axs[i][j].hist(data[i*2+j])
                axs[i][j].set_title("Temp: " + str(temps[i*2+j]))        
        # Labeling
        for ax in axs.flat:
            ax.set(xlabel='index chosen', ylabel='frequency')

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()

        plt.show()


if __name__ == '__main__':
    unittest.main()
