from typing import Dict 
from torch import tensor, zeros
from copy import deepcopy

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
                        dict[(node_a, node_b)] = zeros(num_ops_at_level[i] + extra_ops)
                    

            self.parameters[i] = alpha_i

    def get_alpha_level(self, num_level):
        level = self.parameters[num_level]
        alpha_level = []
        for dag in level:
            alpha_dag = []
            for edge in sorted(dag.keys()):
                alpha_edge = deepcopy(list(dag[edge]))
                alpha_dag.append(alpha_edge)
            alpha_level.append(alpha_dag)
        return tensor(alpha_level)

    def set_alpha_level(self, num_level, alpha_level):
        for dag_num in range(0, self.parameters[num_level]):
            edge_num = 0
            for node_a in range(0, self.num_nodes_at_level[num_level]):
                for node_b in range(node_a + 1, self.num_nodes_at_level[num_level]):
                    self.parameters[num_level][dag_num][(node_a, node_b)] = deepcopy(alpha_level[dag_num][edge_num])
                    edge_num += 1
        
