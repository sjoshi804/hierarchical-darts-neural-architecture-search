from edge import Edge
from copy import deepcopy 
class Architecture:
    def __init__(self, num_levels, num_nodes_at_level, base_operations):

        # Set member variables
        self.num_levels = num_levels
        self.num_nodes_at_level = num_nodes_at_level
        self.base_operation = base_operations

        # Create computational graph
        self.computational_graph = self.create_graph()

    def create_graph(self, level):
        dag = {}

        # Generate sub_dag once and use deepcopies of it across the level
        if level == 0:
            sub_dag = None
        else:
            sub_dag = create_graph(level - 1)

        for node_a in range(self.num_nodes_at_level[level]):
            for node_b in range(node_a + 1, self.num_nodes_at_level[level]):
                dag[(node_a, node_b)] = Edge((1), deepcopy(sub_dag), level)

        return dag
        
    #TODO:
    def get_alpha_i(self, level):
        raise NotImplementedError()

    #TODO:
    def get_operations_i(self, level):
        raise NotImplementedError()


