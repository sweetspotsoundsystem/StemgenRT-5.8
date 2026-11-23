"""Run one stage of the registered paired Bass mask-head experiment."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stemgenrt.mask_teacher_training import main

if __name__ == "__main__":
    main()
