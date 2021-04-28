# External Imports
import sys

# Internal Imports
from util import load_alpha, print_alpha

# Constants
CHECKPOINT_ROOT_DIR = "checkpoints_search"

alpha_dir_path = sys.argv[1]
alpha_normal, alpha_reduce = load_alpha(alpha_dir_path=alpha_dir_path)

print("Alpha Normal")
print_alpha(alpha_normal)

print("Alpha Reduce")
print_alpha(alpha_reduce)
