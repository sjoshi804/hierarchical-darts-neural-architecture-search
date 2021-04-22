Goal: Refactor code to create cells

- Fact: Cell = A Hierarchical Op
- Task: Create Normal / Reduction Cell classes
- Task: Create new layers in model.py for 1d convs in between
- Task: Link cells / layers appropriately
- Task: Change alpha to be alpha_normal, alpha_reduce
- Task: For reduction cells: in which all the operations adjacent
to the input nodes are of stride two.

Goal: Weight Sharing Training
- Assumption: weight sharing only at the 2nd highest level (not the level where the operation is the cell itself, but the level right below that)
- Task: Create model for search with just one operation at this 2nd highest level - equal weight for all operations
    - Action: In search, create a ModelController with just 1 op at 2nd highest level
    - Action: Modify alpha initialization to be equal in the above case (as opposed to random)
- Task: Split search into weight train and alpha train phases
    - Action: In train phase, don't obtain its alpha params and only train weights
- Task: Save operation weights -> Create new operation by loading this operation from memory
    - Action: Loop through every cell and call method to modify its structure to duplicate the operations
    - Action: Randomize the alpha of the model now - at every level but the top most
- Question: What about the Zero Operation?

Goal: Weight Sharing w/o duplication of weights
- Assumption: weight sharing only at the 2nd highest level (not the level where the operation is the cell itself, but the level right below that)
- (Failed) Idea: Modify mixed op so that in softmax bit it reuses same op (fails because while we can now allow for different alpha at this level, we can't recursively allow it)
- (Rejected) Idea: Modify the forward functions of HierarchicalOperation and MixedOperation so that they take in alpha as a parameter and accordingly compute - reject - may lead to issues with gradient computation
- (Failed) Idea: New WeightSharedMixedOp - that takes in n sets of alphas and applies them as required, Modify Hierarchical Op so that it can register a list of alpha dag params as opposed to just the one and - Need some way to modify output of mixed op without creating new ops
- Idea: MixedOp & Hierarchical Op both take in a list of alphas and the forward operation selects which one to use
