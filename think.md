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
- Task: Split search into weight train and alpha train phases
- Task: Save operation weights -> Create new operation by loading this operation from memory
- Task: Create supernet mdoel with fixed weights and train