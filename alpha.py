from typing import Dict 
import unittest
import torch


class Alpha:

    def __init__(self, num_levels: int, num_nodes_at_level: Dict[int, int], num_ops_at_level: Dict[int, int]):
        '''
        - num_levels - how many levels to hierarchy, = 1 makes it equivalent to DARTs, must be >= 1
        - num_nodes_at_levels[i] specifies how many nodes the DAGs that make the operations of level i + 1 have, for the top most level this means that the dag - this dictionary must have values for i = 0 .... num_levels - 1 num_nodes_at_level[0] = number of primitive operations
        - num_ops_at_level[i] - specifies how many operations we will create for a given level - obviously 1 for the top most level, otherwise we would be creating multiple final architectures - hence dictionary must specify values for i = 0, ..., num_levels - 1
        '''

        # Initialize empty dictionary that maps i -> alpha_i i.e. alpha for a given level i
        self.parameters = {}

        # Ensure num_ops_at_level at top level is 1
        num_ops_at_level[num_levels] = 1

        # Define expression for getting number of edges in a complete dag of n nodes
        def num_edges_at_level(n: int):
            if ((n - 1) % 2 == 0):
                return int((n - 1) / 2) * n
            else: 
                return int(n/2) * (n - 1)

        # Initialize alpha_i for all i < num_levels
        for i in range(0, num_levels):
            alpha_i = [[torch.zeros(num_ops_at_level[i] + 2) for x in range(0, num_edges_at_level(num_nodes_at_level[i]))] for y in range(0, num_ops_at_level[i+1])]
            self.parameters[i] = alpha_i
    


class ATest(unittest.TestCase):
    '''
    A simple example
    ----------------
    Number of levels = 3
    Number of primitive operations = 5
    The operations of level 0 are the primitives

    alpha_0 = 
    [
        A 
        [
            [a_identity, a_zero, a_primitive_0, a_primitive_1, a_primitive_2, a_primitive_3, a_primitive_4],
            [a_identity, a_zero, a_primitive_0, a_primitive_1, a_primitive_2, a_primitive_3, a_primitive_4],
            [a_identity, a_zero, a_primitive_0, a_primitive_1, a_primitive_2, a_primitive_3, a_primitive_4]
        ],

        B
        [
            [a_identity, a_zero, a_primitive_0, a_primitive_1, a_primitive_2, a_primitive_3, a_primitive_4],
            [a_identity, a_zero, a_primitive_0, a_primitive_1, a_primitive_2, a_primitive_3, a_primitive_4],
            [a_identity, a_zero, a_primitive_0, a_primitive_1, a_primitive_2, a_primitive_3, a_primitive_4]
        ],

        C
        [
            [a_identity, a_zero, a_primitive_0, a_primitive_1, a_primitive_2, a_primitive_3, a_primitive_4],
            [a_identity, a_zero, a_primitive_0, a_primitive_1, a_primitive_2, a_primitive_3, a_primitive_4],
            [a_identity, a_zero, a_primitive_0, a_primitive_1, a_primitive_2, a_primitive_3, a_primitive_4]
        ]
    ]

    alpha_0 thus has specified how to create  operations of level 1 that have 3 nodes
    A, B, C specify how to create level 1 ops from level 0 ops  

    alpha_1 = 
    [
        D
        [
            [a_identity, a_zero, a_op^1_0, a_op^1_1, a_op^1_2],
            [a_identity, a_zero, a_op^1_0, a_op^1_1, a_op^1_2],
            [a_identity, a_zero, a_op^1_0, a_op^1_1, a_op^1_2]
        ],

        E
        [
            [a_identity, a_zero, a_op^1_0, a_op^1_1, a_op^1_2],
            [a_identity, a_zero, a_op^1_0, a_op^1_1, a_op^1_2],
            [a_identity, a_zero, a_op^1_0, a_op^1_1, a_op^1_2]
        ],

        F
        [
            [a_identity, a_zero, a_op^1_0, a_op^1_1, a_op^1_2],
            [a_identity, a_zero, a_op^1_0, a_op^1_1, a_op^1_2],
            [a_identity, a_zero, a_op^1_0, a_op^1_1, a_op^1_2]
        ]
    ]

    alpha_1 thus has specified how to create 3 operations of level 2 each with 3 nodes
    D, E, F specify how to create level 2 ops from level 1 ops

    alpha_2 = 
    [
        Final Architecture
        [
            [a_identity, a_zero, a_op^2_0, a_op^2_1, a_op^2_2],
            [a_identity, a_zero, a_op^2_0, a_op^2_1, a_op^2_2],
            [a_identity, a_zero, a_op^2_0, a_op^2_1, a_op^2_2]
        ]
    ]

    This specifies how to use level 2 ops: D, E, F and create a final architecture

    alpha can be thought of as [alpha_0, alpha_1, alpha_2]
    '''
    def test_initialization(self):
        num_levels = 3
        num_nodes_at_level = {0: 3, 1: 3, 2: 3}
        num_ops_at_level = {0: 5, 1: 3, 2: 3}
        testAlpha = Alpha(num_levels, num_nodes_at_level, num_ops_at_level)

        # Check parameters
        for i in range(0, num_levels):
            alpha_i = testAlpha.parameters[i]
            for op_num in range(0, num_ops_at_level[i + 1]):
                number_of_edges = 0
                for x in range(0, num_nodes_at_level[i]):
                    number_of_edges += x
                for edge_num in range(0, number_of_edges):
                    num_parameters = num_ops_at_level[i] + 2
                    assert(alpha_i[op_num][edge_num].equal(torch.zeros(num_parameters)))

if __name__ == '__main__':
    unittest.main()

 