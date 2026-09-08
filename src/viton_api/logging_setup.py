"""Per-job log stream: merged stdout+stderr of the conda stage is appended here."""

from __future__ import annotations

from pathlib import Path


class JobLog:
    """Append-only file handle for `<workdir>/job.log`.

    The orchestrator pumps the subprocess's merged stdout/stderr into this file
    so a failed job leaves a full transcript behind in its scratch workdir.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._fh = None

    def open(self) -> "JobLog":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8", buffering=1)
        return self

    def write(self, data: str) -> None:
        if self._fh is not None and data:
            self._fh.write(data)

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.flush()
            finally:
                self._fh.close()
                self._fh = None

    def __enter__(self) -> "JobLog":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()
