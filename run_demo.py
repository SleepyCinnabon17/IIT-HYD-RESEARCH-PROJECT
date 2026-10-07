"""Download the verified detector and launch the downloadable demo."""

import os
import sys
from pathlib import Path

from deployment import prepare_detector

if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    print("Checking/downloading the detector. First startup needs internet.", flush=True)
    prepare_detector()
    from app import main

    raise SystemExit(main(sys.argv[1:]))
