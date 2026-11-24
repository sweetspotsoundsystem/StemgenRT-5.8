"""Train and score the experimental causal Bass readout."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stemgenrt.peer_readout_training import main


if __name__ == "__main__":
    main()
