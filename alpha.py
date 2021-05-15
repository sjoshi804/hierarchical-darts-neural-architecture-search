from torch import zeros, rand, randn
from typing import Dict 
import torch.nn as nn

class Alpha:
    '''
    This class is used to represent alpha the architecture parameters.

    Use this class as interface to:
    - initialize alpha
    - get alpha for creating a model
    - update alpha_i after a gradient update step
    '''

    def __init__(self, num_levels: int, num_nodes_at_level: Dict[int, int], num_ops_at_level: Dict[int, int], randomize=False):
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
                        # Skip creation of alpha if top level and edges don't exist
                        if (i == self.num_levels - 1) and ((node_a < 2 and node_b == 1) or (node_b == num_nodes_at_level[i] - 1)):
                            continue 

                        # Determine num of extra ops
                        if i == 0:
                            extra_ops = 1
                        else:
                            extra_ops = 0

                        # Initializing the alpha for an edge
                        # Each value in this list is a parameter
                        if i == self.num_levels - 1 or not randomize:
                            dict[(node_a, node_b)] = nn.Parameter(1e-3 * randn(num_ops_at_level[i] + extra_ops))
                        else:
                            dict[(node_a, node_b)] = nn.Parameter(rand(num_ops_at_level[i] + extra_ops))
                    
            self.parameters[i] = alpha_i

    # List of all the parameters for a given level, so that optimizer can work with them all together
    # Lexicographic ordering
    def get_alpha_level(self, num_level):

        # Gets the list of dags at a given level
        level = self.parameters[num_level]

        # Initialize an empty list to contain all the parameters
        alpha_level = []

        # Loop through all the dags 
        for dag in level:

            # Each alpha_dag will now be the lexicographic ordering of all the parameters for each of its edges
            alpha_dag = []

            for edge in sorted(dag.keys()):
                # Extend alpha_dag by the alpha parameters for this edge
                alpha_edge = dag[edge]
                alpha_dag.append(alpha_edge)

            # Extend overall alpha_level by the alpha parameters for this dag (alpha_dag)
            alpha_level.extend(alpha_dag)

        # Make alpha_level a nn.ParameterList
        return nn.ParameterList(alpha_level)