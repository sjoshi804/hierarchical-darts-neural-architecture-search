Goal: Refactor code to create cells

- Fact: Cell = A Hierarchical Op
- Task: Create Normal / Reduction Cell classes
- Task: Create new layers in model.py for 1d convs in between
- Task: Link cells / layers appropriately
- Task: Change alpha to be alpha_normal, alpha_reduce
- Task: For reduction cells: in which all the operations adjacent
to the input nodes are of stride two.