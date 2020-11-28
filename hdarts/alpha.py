from typing import Dict 
import unittest
import torch

class Alpha:
    '''
    This class is used to represent alpha the architecture parameters.

    Use this class as interface to:
    - initialize alpha
    - get alpha for creating a model
    - update alpha_i after a gradient update step
    '''

    def __init__(self, num_levels: int, num_nodes_at_level: Dict[int, int], num_ops_at_level: Dict[int, int]):
        '''
        - num_levels - how many levels to hierarchy, = 1 makes it equivalent to DARTs, must be >= 1
        - num_nodes_at_levels[i] specifies how many nodes the DAGs that make the operations of level i + 1 have, for the top most level this means that the dag - this dictionary must have values for i = 0 .... num_levels - 1 num_nodes_at_level[0] = number of primitive operations
        - num_ops_at_level[i] - specifies how many operations we will create for a given level - doesn't include zero and identity on primtivie level and doesn't include zero on higher levels - obviously 1 for the top most level, otherwise we would be creating multiple final architectures - hence dictionary must specify values for i = 0, ..., num_levels - 1
        '''
        # Input validation
        if (num_levels < 1):
            raise Exception("Invalid number of levels, must be >= 1")
        for i in range(0, num_levels):
            if i in num_nodes_at_level:
                if num_nodes_at_level[i] < 2:
                    raise Exception("Insufficient number of nodes at level " + str(i) + ". Must be atleast 2.")
            else:
                raise Exception(str(i) + " key missing from num_nodes_at_level")
            if i in num_ops_at_level:
                if num_ops_at_level[i] < 1:
                    raise Exception("Insufficient number of ops at level " + str(i) + ". Must be atleast 1.")
            else:
                raise Exception(str(i) + " key missing from num_ops_at_level")

        # Initialize member variables required to access parameters dict correctly to construct neural network
        self.num_levels = num_levels
        self.num_nodes_at_level = num_nodes_at_level
        self.num_ops_at_level = num_ops_at_level

        # Initialize empty dictionary that maps i -> alpha_i i.e. alpha for a given level i
        self.parameters = {}

        # Ensure num_ops_at_level at top level is 1
        num_ops_at_level[num_levels] = 1

        # Initialize alpha_i for all i < num_levels
        for i in range(0, num_levels):

            # Create a list of dictionaries, each dictionary represents the parameters for an operation at the next level
            alpha_i = [ {}  for y in range(0, num_ops_at_level[i+1])]

            # For each operation of level above i.e. dict, 
            # insert key = edge tuple
            # value = a list with size = num of operations at current level ...
            # ... since this indicates how this edge will be formed as a mix
            for dict in alpha_i:
                for node_a in range(0, num_nodes_at_level[i]):
                    for node_b in range(node_a + 1, num_nodes_at_level[i]):
                        extra_ops = 1
                        if i == 0:
                            extra_ops = 2
                        dict[(node_a, node_b)] = torch.zeros(num_ops_at_level[i] + extra_ops)
                    

            self.parameters[i] = alpha_i


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
        },

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

    alpha.parameters = [alpha_0, alpha_1, alpha_2]
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
                        assert(alpha_i[op_num][(node_a, node_b)].equal(torch.zeros(num_parameters)))

    def test_input_validation(self):
        raise NotImplementedError

if __name__ == '__main__':
    unittest.main()

 