"""Cross-process advisory locks for cameras, microphones and real dealers."""

from __future__ import annotations

from contextlib import AbstractContextManager
import hashlib
from pathlib import Path
from threading import Lock
from typing import BinaryIO, Iterable


class ResourceBusyError(RuntimeError):
    """Raised when another runtime owns a requested live resource."""


_PROCESS_GUARD = Lock()
_PROCESS_OWNERS: set[str] = set()


def _lock_file(file: BinaryIO) -> None:
    try:
        import msvcrt

        file.seek(0)
        msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
    except ImportError:
        import fcntl

        fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(file: BinaryIO) -> None:
    try:
        import msvcrt

        file.seek(0)
        msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
    except ImportError:
        import fcntl

        fcntl.flock(file.fileno(), fcntl.LOCK_UN)


class ResourceLock(AbstractContextManager["ResourceLock"]):
    """Own one named resource until explicitly released."""

    def __init__(self, lock_root: Path, resource_id: str) -> None:
        if not resource_id.strip():
            raise ValueError("resource_id must not be blank")
        digest = hashlib.sha256(resource_id.encode("utf-8")).hexdigest()[:20]
        self.resource_id = resource_id
        self.path = lock_root / f"{digest}.lock"
        self._file: BinaryIO | None = None

    def acquire(self) -> ResourceLock:
        if self._file is not None:
            return self
        key = str(self.path.resolve())
        with _PROCESS_GUARD:
            if key in _PROCESS_OWNERS:
                raise ResourceBusyError(f"resource already in use: {self.resource_id}")
            _PROCESS_OWNERS.add(key)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            file = self.path.open("a+b")
            if file.seek(0, 2) == 0:
                file.write(b"0")
                file.flush()
            _lock_file(file)
            self._file = file
        except (OSError, IOError) as exc:
            with _PROCESS_GUARD:
                _PROCESS_OWNERS.discard(key)
            raise ResourceBusyError(
                f"resource already in use: {self.resource_id}"
            ) from exc
        return self

    def release(self) -> None:
        if self._file is None:
            return
        file = self._file
        self._file = None
        key = str(self.path.resolve())
        try:
            _unlock_file(file)
        finally:
            file.close()
            with _PROCESS_GUARD:
                _PROCESS_OWNERS.discard(key)

    def __exit__(self, *args: object) -> None:
        self.release()


class RuntimeResourceLocks(AbstractContextManager["RuntimeResourceLocks"]):
    """Acquire a set atomically in stable order and release in reverse order."""

    def __init__(self, lock_root: Path, resource_ids: Iterable[str]) -> None:
        unique_ids = sorted(set(resource_ids))
        self._locks = tuple(ResourceLock(lock_root, item) for item in unique_ids)
        self._acquired: list[ResourceLock] = []

    def acquire(self) -> RuntimeResourceLocks:
        try:
            for lock in self._locks:
                lock.acquire()
                self._acquired.append(lock)
        except Exception:
            self.release()
            raise
        return self

    def release(self) -> None:
        while self._acquired:
            self._acquired.pop().release()

    def __exit__(self, *args: object) -> None:
        self.release()


__all__ = ["ResourceBusyError", "ResourceLock", "RuntimeResourceLocks"]
