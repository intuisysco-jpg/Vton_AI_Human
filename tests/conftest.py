"""Portable lifecycle tests; inference and GPU locking remain Linux-only."""

import sys
from types import ModuleType


if sys.platform == "win32":
    # Worker runtime tests never acquire a GPU lease. Reject accidental use
    # rather than pretending a Windows no-op provides POSIX mutual exclusion.
    fcntl = ModuleType("fcntl")

    def unavailable(*args):
        raise RuntimeError("GPU flock requires the Linux worker runtime")

    fcntl.flock = unavailable
    fcntl.LOCK_EX, fcntl.LOCK_NB, fcntl.LOCK_UN = 2, 4, 8
    sys.modules.setdefault("fcntl", fcntl)
