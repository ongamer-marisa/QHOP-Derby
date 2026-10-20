"""Regression tests for Windows readers racing atomic training writes."""
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from qhop_train import atomic_io
from qhop_train import checkpoint
from qhop_train.control import atomic_json


def _busy(code=5):
    return PermissionError(13, "sharing conflict", None, code)


class AtomicIOTests(unittest.TestCase):
    def test_transient_contention_retries_without_rewriting_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            atomic_json(path, {"status": "old"})
            actual_replace = os.replace
            attempts = []

            def conflict_then_replace(source, destination):
                attempts.append((source, destination, Path(source).read_bytes()))
                if len(attempts) < 3:
                    raise _busy(32)
                actual_replace(source, destination)

            with patch.object(atomic_io.os, "replace", side_effect=conflict_then_replace):
                atomic_json(path, {"status": "new"})
            self.assertEqual(len(attempts), 3)
            self.assertEqual(len({str(row[0]) for row in attempts}), 1)
            self.assertEqual(len({row[2] for row in attempts}), 1)
            self.assertEqual(json.loads(path.read_text())["status"], "new")
            self.assertEqual(list(Path(directory).glob(".qhop-*")), [])

    def test_persistent_contention_preserves_old_status_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            atomic_json(path, {"status": "old"})
            with patch.object(atomic_io.os, "replace", side_effect=_busy(5)):
                with self.assertRaises(PermissionError):
                    atomic_json(path, {"status": "new"})
            self.assertEqual(json.loads(path.read_text())["status"], "old")
            self.assertEqual(list(Path(directory).glob(".qhop-*")), [])

    def test_other_error_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            with patch.object(atomic_io.os, "replace", side_effect=_busy(87)) as replace:
                with self.assertRaises(PermissionError):
                    atomic_json(path, {"status": "new"})
            self.assertEqual(replace.call_count, 1)
            self.assertEqual(list(Path(directory).glob(".qhop-*")), [])

    def test_serialization_and_cleanup_failures_preserve_original_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(atomic_io.Path, "unlink", side_effect=_busy(5)):
                with self.assertRaises(TypeError):
                    atomic_json(Path(directory) / "status.json", {"bad": object()})

    def test_checkpoint_failure_preserves_previous_file(self):
        from types import SimpleNamespace

        component = SimpleNamespace(state_dict=lambda: {"value": 1})
        ppo = SimpleNamespace(vision=component,actor=component, critic=component, optimizer=component,
                              schedule=component, normalizer=component)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            path.write_bytes(b"previous complete checkpoint")
            with patch.object(atomic_io.os, "replace", side_effect=_busy(5)):
                with self.assertRaises(PermissionError):
                    checkpoint.save(path, ppo, {"global_updates": 1}, {})
            self.assertEqual(path.read_bytes(), b"previous complete checkpoint")
            self.assertEqual(list(Path(directory).glob(".qhop-*")), [])


@unittest.skipUnless(os.name == "nt", "Windows file-sharing behavior")
class WindowsHandleTests(unittest.TestCase):
    @staticmethod
    def _open_reader(path):
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD,
            wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
            wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.CreateFileW(str(path), 0x80000000, 1, None, 3, 0x80, None)
        if handle == wintypes.HANDLE(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        return kernel, handle

    def test_replacement_waits_for_reader_then_succeeds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "status.json"
            atomic_json(path, {"status": "old"})
            kernel, handle = self._open_reader(path)
            timer = threading.Timer(0.1, kernel.CloseHandle, args=(handle,))
            timer.start()
            try:
                atomic_json(path, {"status": "new"})
            finally:
                timer.join()
            self.assertEqual(json.loads(path.read_text())["status"], "new")

    def test_replacement_times_out_with_reader_open(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "status.json"
            atomic_json(path, {"status": "old"})
            kernel, handle = self._open_reader(path)
            try:
                with self.assertRaises(PermissionError):
                    atomic_json(path, {"status": "new"})
                self.assertEqual(json.loads(path.read_text())["status"], "old")
                self.assertEqual(list(Path(directory).glob(".qhop-*")), [])
            finally:
                kernel.CloseHandle(handle)
