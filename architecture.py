from edge import Edge
from copy import deepcopy 
class Architecture:
    def __init__(self, num_levels, num_nodes_at_level, base_operators):

        # Set member variables
        self.num_levels = num_levels # L-1
        self.num_nodes_at_level = num_nodes_at_level # n_i
        self.operators = {0: base_operators}
        default_operators = {"identity",  "zero"} #TODO: fix this with correct operation representation
        for i in range(1, num_levels+1):
            self.operators[i] = deepcopy(default_operators)
        # Create computational dag
        self.computational_dag = self.create_dag()

    def create_dag(self, level):
        dag = {}

        # Generate sub_dag once and use deepcopies of it across the level
        if level == 0:
            sub_dag = None
        else:
            sub_dag = create_dag(level - 1)

        for node_a in range(self.num_nodes_at_level[level]):
            for node_b in range(node_a + 1, self.num_nodes_at_level[level]):
                dag[(node_a, node_b)] = Edge((1), deepcopy(sub_dag), level)

        return dag
        
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



