from edge import Edge
from copy import deepcopy 
import unittest
import numpy as np

# num_levels here only refers to the useful levels therefore = L - 1 (pseudocode)
class Architecture:
    def __init__(self, num_levels: int, num_nodes_at_level: dict[int, int], base_operators: list):

        # Set member variables
        self.num_levels = num_levels # L-1
        self.num_nodes_at_level = num_nodes_at_level # n_i
        self.operators = {0: base_operators}
        default_operators = ["identity",  "zero"] #TODO: fix this with correct operation representation
        self.base_operations = default_operators + base_operators
        # Create computational dag
        self.computational_dag = self.create_dag()

    def create_dag(self, level):
        dag = {}

        # Generate sub_dag and alpha once and use deepcopies of it across the level
        if level == 0:
            sub_dag = None
            alpha = [np.zeroes(len(self.base_operations)) for i in range(self.num_nodes_at_level[level])]
        else:
            sub_dag = create_dag(level - 1)
            num_edges_in_level = 1
            for i in reversed(range(level, self.num_levels)):
                num_edges_in_level *= i
            alpha = [np.zeroes(num_edges_in_level) for i in range(self.num_nodes_at_level[level])]    

        for node_a in range(self.num_nodes_at_level[level]):
            for node_b in range(node_a + 1, self.num_nodes_at_level[level]):
                dag[(node_a, node_b)] = Edge(deepcopy(alpha), deepcopy(sub_dag), level)

        return dag
    
    # Initialization of alpha for a given edge - equal weights to all ops
    # Alpha_i is a tuple with (2 + # of edges in overall dag at this level) entries
    # This describes an edge as a mix of operato
    def init_alpha_e(self, level):
        

    # Gets the vector with alpha parameters for all edges in lexicographic order
    def get_alpha_i(self, level):
        levels_to_descend = self.num_levels - level
        return self.get_all_alpha(self.computational_dag, levels_to_descend)

    def get_all_alpha(self, dag, levels_to_descend):
        if levels_to_descend == 0:
            return [dag[edge_label].alpha for edge_label in sorted(dag.keys())]
        else:
            alpha = []
            for edge_label in sorted(dag.keys()):
                alpha += get_all_alpha(dag[edge_label].sub_dag, levels_to_descend-1)
            return alpha

    #TODO: update alpha i
    def update_alpha_i(self, level):
        # updates the operators as well
        raise NotImplementedError()

class TestArchitecture(unittest.TestCase):
    def test_dummy(self):
        assert(True)

if __name__ == '__main__':
    unittest.main()