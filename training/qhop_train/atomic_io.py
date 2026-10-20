"""Shared same-directory atomic replacement with bounded Windows sharing retries."""
from __future__ import annotations

import logging
import os
from pathlib import Path
import time


_SHARING_ERRORS = {5, 32, 33}
_MAX_WAIT = 2.0


def replace_with_retry(temporary, destination):
    destination = Path(destination)
    started = time.monotonic()
    delay = 0.02
    retries = 0
    last_code = None
    while True:
        try:
            os.replace(temporary, destination)
            if retries:
                logging.warning("Atomic replace succeeded after retry: operation=replace "
                                "path=%s winerror=%s retries=%d elapsed=%.3fs",
                                destination, last_code, retries, time.monotonic() - started)
            return
        except OSError as error:
            elapsed = time.monotonic() - started
            code = getattr(error, "winerror", None)
            if os.name != "nt" or code not in _SHARING_ERRORS or elapsed >= _MAX_WAIT:
                logging.error("Atomic replace failed: operation=replace path=%s "
                              "winerror=%s elapsed=%.3fs", destination, code, elapsed)
                raise
            retries += 1
            last_code = code
            time.sleep(min(delay, 0.2, _MAX_WAIT - elapsed))
            delay = min(delay * 2, 0.2)


def cleanup_temporary(path):
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        logging.exception("Could not remove temporary file: %s", path)
