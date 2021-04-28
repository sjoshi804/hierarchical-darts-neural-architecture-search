# External Imports
from copy import deepcopy
from os import write
import sys

# Internal Imports
from util import create_alpha_history_object, load_alpha,print_alpha, update_alpha_history, write_alpha_history_to_csvs

alpha_dir_path = sys.argv[1]
num_epochs = int(sys.argv[2])

# Load and print best alpha
best_alpha_normal, best_alpha_reduce = load_alpha(alpha_dir_path=alpha_dir_path)
print("Best Alpha Normal")
print_alpha(best_alpha_normal)

print("Best Alpha Reduce")
print_alpha(best_alpha_reduce)

# Construct Alpha History Object
alpha_normal_history = create_alpha_history_object(best_alpha_normal)
alpha_reduce_history = deepcopy(alpha_normal_history)

for epoch in range(num_epochs):
    alpha_normal, alpha_reduce = load_alpha(alpha_dir_path=alpha_dir_path, epoch=epoch)
    print(epoch)
    update_alpha_history(alpha_history=alpha_normal_history, alpha=alpha_normal)
    update_alpha_history(alpha_history=alpha_reduce_history, alpha=alpha_reduce)

# Write to csv
write_alpha_history_path=alpha_dir_path.replace('checkpoints_search','alpha_history')
write_alpha_history_to_csvs(alpha_history=alpha_normal_history, alpha=alpha_normal, alpha_type="normal", write_dir=write_alpha_history_path)
write_alpha_history_to_csvs(alpha_history=alpha_normal_history, alpha=alpha_normal, alpha_type="reduce", write_dir=write_alpha_history_path)
