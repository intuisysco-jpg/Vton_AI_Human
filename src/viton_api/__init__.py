"""Background Removal GPU worker — thin-venv / conda delegation sibling worker.

This package imports NO ML libraries (no torch / transformers / cv2 /
detectron2 / bitsandbytes). Every heavy stage runs as a `conda run` subprocess
in its dedicated conda env. The service only orchestrates broker I/O, DB state,
subprocess lifecycle and timeouts.
"""

__version__ = "0.1.0"
