# External imports
from torch import argmax
import torch.nn as nn 

# Internal imports
from hierarchical_operation import HierarchicalOperation
from mixed_operation import MixedOperation
from model import Model


class LearntModel(nn.Module):
    '''
    - Extracts the final model learnt by HDARTS.
    - Uses weights learned during optimization to initialize the operations.

    This class takes in a Model object and converts it into a learntModel which 
    is identical to the model but with the MixedOperations that softmaxed alpha_e 
    to weight the candidate operations replaced by the arg max of the alpha_e.
    '''
    def __init__(self, model: Model):
        # Superclass constructor
        super().__init__()

        # Recursive function to replaced MixedOperations with the operation with maximum weight
        def finalize_operation(operation):

            # If mixed operation, recurse on each of its candidates if possible
            if type(operation) == MixedOperation:

                # If any operation inside is HierarchicalOperation then not base case
                is_base_case = True
                for op in operation.ops:
                    if type(op) == HierarchicalOperation:
                        is_base_case = False
                        break
                
                # If base case, return operation corresponding to argmax of alpha_e
                if is_base_case:
                    return operation.ops[argmax(operation.alpha_e)]
                # Else recurse on operation corresponding to argmax of alpha_e
                else:
                    return finalize_operation(operation.ops[argmax(operation.alpha_e)])
        
        
            # Else must be HierarchicalOperation, need to recurse on each edge and reassemble the dag
            new_ops = {}
            for node_a in range(0, operation.num_nodes):
                for node_b in range(node_a + 1, operation.num_nodes):
                    # Edge tuple
                    edge = str((node_a, node_b))
                    # Recurse on old operation to create new operation
                    new_ops[edge] = finalize_operation(operation.ops[edge])

            # Return new Hierarchical Operation constructed using these operations
            return HierarchicalOperation(operation.num_nodes, new_ops)
        
        # Finalize the top_level_op in model recursively
        model.top_level_op = finalize_operation(model.top_level_op)

        # Instantiate model member variable to register parameters / sub-modules etc.
        self.model = model

    def forward(self, x):
        return self.model(x)





