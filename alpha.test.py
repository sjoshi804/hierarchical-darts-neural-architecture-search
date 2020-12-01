# External imports
import unittest
from torch import zeros

# Internal imports
from alpha import Alpha

class AlphaTest(unittest.TestCase):
    '''
    A simple example
    ----------------
    Number of levels = 3
    Number of primitive operations = 5
    The operations of level 0 are the primitives

    alpha_0 = 
    [
        # A 
        {
            (0,1): [a_primitive_0, a_primitive_1, a_primitive_2, a_primitive_3, a_primitive_4, a_identity, a_zero],
            
            (0,2): [a_primitive_0, a_primitive_1, a_primitive_2, a_primitive_3, a_primitive_4, a_identity, a_zero],
            
            (1,2): [a_primitive_0, a_primitive_1, a_primitive_2, a_primitive_3, a_primitive_4, a_identity, a_zero]
        }

        # B
        {
            (0,1): [a_primitive_0, a_primitive_1, a_primitive_2, a_primitive_3, a_primitive_4, a_identity, a_zero],

            (0,2): [a_primitive_0, a_primitive_1, a_primitive_2, a_primitive_3, a_primitive_4, a_identity, a_zero],
            
            (1,2): [a_primitive_0, a_primitive_1, a_primitive_2, a_primitive_3, a_primitive_4, a_identity, a_zero]
        },

        # C
        {
            (0,1): [a_primitive_0, a_primitive_1, a_primitive_2, a_primitive_3, a_primitive_4, a_identity, a_zero],
            
            (0,2): [a_primitive_0, a_primitive_1, a_primitive_2, a_primitive_3, a_primitive_4, a_identity, a_zero],
            
            (1,2): [a_primitive_0, a_primitive_1, a_primitive_2, a_primitive_3, a_primitive_4, a_identity, a_zero]
        }
    ]

    alpha_0 thus has specified how to create  operations of level 1 that have 3 nodes
    A, B, C specify how to create level 1 ops from level 0 ops  

    alpha_1 = 
    [
        # D
        {
            (0,1): [a_op^1_0, a_op^1_1, a_op^1_2, a_zero],
            (0,2): [a_op^1_0, a_op^1_1, a_op^1_2, a_zero],
            (1,2): [a_op^1_0, a_op^1_1, a_op^1_2, a_zero]
        },

        # E
        {
            (0,1): [a_op^1_0, a_op^1_1, a_op^1_2, a_zero],
            (0,2): [a_op^1_0, a_op^1_1, a_op^1_2, a_zero],
            (1,2): [a_op^1_0, a_op^1_1, a_op^1_2, a_zero]
        },

        # F
        {
            (0,1): [a_op^1_0, a_op^1_1, a_op^1_2, a_zero],
            (0,2): [a_op^1_0, a_op^1_1, a_op^1_2, a_zero],
            (1,2): [a_op^1_0, a_op^1_1, a_op^1_2, a_zero]
        }
    ]

    alpha_1 thus has specified how to create 3 operations of level 2 each with 3 nodes
    D, E, F specify how to create level 2 ops from level 1 ops

    alpha_2 = 
    [
        Final Architecture
        {
            (0,1): [a_op^2_0, a_op^2_1, a_op^2_2, a_zero],
            (0,2): [a_op^2_0, a_op^2_1, a_op^2_2, a_zero],
            (1,2): [a_op^2_0, a_op^2_1, a_op^2_2, a_zero]
        }
    ]

    This specifies how to use level 2 ops: D, E, F and create a final architecture

    alpha.parameters = {0: alpha_0, 1: alpha_1, 2: alpha_2}
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
                for node_a in range(0, num_nodes_at_level[i]):
                    for node_b in range(node_a + 1, num_nodes_at_level[i]):
                        if i == 0:
                            num_parameters = num_ops_at_level[i] + 2
                        else:
                            num_parameters = num_ops_at_level[i] + 1
                        assert(alpha_i[op_num][(node_a, node_b)].equal(zeros(num_parameters)))

    def test_input_validation(self):
        raise NotImplementedError

    def test_get_alpha_level(self):
        raise NotImplementedError

    def test_set_alpha_level(self):
        raise NotImplementedError

if __name__ == '__main__':
    unittest.main()

 