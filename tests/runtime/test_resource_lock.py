from __future__ import annotations

from pathlib import Path

import pytest

from poker_dealer.runtime import ResourceBusyError, ResourceLock, RuntimeResourceLocks


def test_resource_lock_rejects_second_owner_and_can_be_reacquired(tmp_path: Path) -> None:
    first = ResourceLock(tmp_path, "camera:local:0").acquire()
    try:
        with pytest.raises(ResourceBusyError, match="camera:local:0"):
            ResourceLock(tmp_path, "camera:local:0").acquire()
    finally:
        first.release()
    ResourceLock(tmp_path, "camera:local:0").acquire().release()


def test_resource_lock_set_rolls_back_partial_acquisition(tmp_path: Path) -> None:
    owned = ResourceLock(tmp_path, "microphone:default").acquire()
    group = RuntimeResourceLocks(tmp_path, ["camera:local:0", "microphone:default"])
    try:
        with pytest.raises(ResourceBusyError):
            group.acquire()
        ResourceLock(tmp_path, "camera:local:0").acquire().release()
    finally:
        owned.release()
