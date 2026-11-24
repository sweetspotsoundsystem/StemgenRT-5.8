"""Fit and score the offline Bass/Other target projection."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stemgenrt.teacher_projection_calibration import main

if __name__=="__main__":
    main()
