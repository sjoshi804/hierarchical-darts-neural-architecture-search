import torch.nn as nn

# String Constants
PREPROC_X = "preproc_x"
PREPROC_X2 = "preproc_x2"

class LearntOperation(nn.Module):
    '''
    Creates an operation given the alpha parameters on an dag and the associated base operations.
    '''
    def init(self, num_nodes, alpha_dag, base_operations, preproc_x=None, preproc_x2=None):
        self.num_nodes = num_nodes
        self.ops = nn.ParameterDict()

        # Add preprocessing to dictionary if provided
        if not preproc_x:
            self.ops[PREPROC_X] = preproc_x
        if not preproc_x2:
            self.ops[PREPROC_X2 = preproc_x2
        
        # Pick operation that corresponds to max alpha value
        for edge in alpha_dag.keys():
            op_num = int(np.argmax(alpha.parameters[level][op_num][edge].cpu().detach()))
            self.ops[str(edge)] = base_operations[op_num]
    
    def forward(self, x, x2=None):
        '''
        Iteratively compute using each edge of the dag
        '''
        output = {}

        # Apply preprocessing if applicable
        if PREPROC_X in self.ops:
        x = self.ops[PREPROC_X].forward(x)
        if PREPROC_X2 in self.ops:
        x2 = self.ops[PREPROC_X2].forward(x2)

        for node_a in range(0, self.num_nodes):
        # For a given edge, determine the input to the starting node
        if (node_a == 0): 
            # for node_a = 0, it is trivial, input of entire module / first input
            input = x
        elif (node_a == 1 and type(x2) != type(None)):
            # if top level, then x2 provided then use for second node
            input = x2
        else: 
            # otherwise it is the concatentation of the output of every edge (node, node_a)
            input = []
            for prev_node in range(0, node_a):
                edge = str((prev_node, node_a))
                if edge in output: # Ensure edge exists
                input.append(output[edge])
            input = cat(tuple(input), dim=1) 

        for node_b in range(node_a + 1, self.num_nodes):

            edge = str((node_a, node_b))

            # If edge doesn't exist, skip it
            if edge not in self.ops:
            continue
            else:       
            output[edge] = self.ops[edge].forward(input)
        
        # By extension, final output will be the concatenation of all inputs to the final node
        if type(x2) != type(None): # if top level skip input nodes
        start_node = 2
        else:
        start_node = 0
        return cat(tuple([output[str((prev_node, self.num_nodes - 1))] for prev_node in range(start_node, self.num_nodes - 1)]), dim=1)       
            