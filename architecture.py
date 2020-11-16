from edge import Edge

class Architecture:
    def __init__(self, num_levels, num_nodes_at_level, base_operations):
        self.num_levels = num_levels
        self.num_nodes_at_level = num_nodes_at_level
        self.base_operation = base_operations
        self.initialize()

    def initialize(self): #TODO: Must be generalized for any number of levels - JUST MAKE IT RECURSIVE
        # Loop through all possible edges for dag at level 2
        for level_2_node_a in range(self.num_nodes_at_level[2]):
            for level_2_node_b in range(level_2_node_a + 1, self.num_nodes_at_level[2]):
                level_2_sub_dag = {} # Initialize empty dict to contain all edges for the sub dag
                # Loop through all possible edges for dag at level 1
                for level_1_node_a in range(self.num_nodes_at_level[1]):
                    for level_1_node_b in range(level_1_node_a + 1, self.num_nodes_at_level[1]):
                        level_1_sub_dag = {} # Initialize empty dict to contain all edges for the sub dag
                        #Loop through all possible edges for dag at level 0
                        for level_0_node_a in range(self.num_nodes_at_level[0]):
                            for level_0_node_b in range(level_0_node_a + 1, self.num_nodes_at_level[0]):
                                level_1_sub_dag[(level_0_node_a, level_0_node_b)] = Edge((1), None, 0) # Add this edge to the sub_dag for level above
                        level_2_dag[(level_1_node_a, level_1_node_b)] = Edge((1), level_1_sub_dag, 1) # Add this edge to the sub_dag for level above
                architecure[(level_2_node_a, level_2_node_b)] = Edge((1), level_2_sub_dag, 2) # Add this edge to the sub_dag for level above
    
    #TODO:
    def get_alpha_i(self, level):
        raise NotImplementedError()

    #TODO:
    def get_operations_i(self, level):
        raise NotImplementedError()


