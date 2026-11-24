"""Train matched residual-gradient routes through a set of causal output heads."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stemgenrt.joint_residual_training import main

if __name__ == "__main__":
    main()
