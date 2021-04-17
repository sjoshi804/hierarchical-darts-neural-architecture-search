# External Imports
import sys

# Internal Imports
from util import load_best_alpha, print_alpha

# Constants
CHECKPOINT_ROOT_DIR = "checkpoints_search"

run_date_time = sys.argv[1]
alpha_normal, alpha_reduce = load_best_alpha(CHECKPOINT_ROOT_DIR, run_date_time)

print("Alpha Normal")
print_alpha(alpha_normal)

print("Alpha Reduce")
print_alpha(alpha_reduce)
