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