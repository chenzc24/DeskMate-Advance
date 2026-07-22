"""Hash-chained append-only hand event storage."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from .engine import HandState


@dataclass(frozen=True, slots=True)
class HandEvent:
    sequence: int
    event_id: str
    kind: str
    observed_at_ns: int
    before_version: int
    after_version: int
    accepted: bool
    payload: Mapping[str, Any]
    state_after: Mapping[str, Any]
    previous_hash: str
    event_hash: str

    def unsigned(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "event_id": self.event_id,
            "kind": self.kind,
            "observed_at_ns": self.observed_at_ns,
            "before_version": self.before_version,
            "after_version": self.after_version,
            "accepted": self.accepted,
            "payload": self.payload,
            "state_after": self.state_after,
            "previous_hash": self.previous_hash,
        }


class EventLog:
    def __init__(self) -> None:
        self.events: list[HandEvent] = []

    @staticmethod
    def _hash(unsigned: Mapping[str, Any]) -> str:
        encoded = json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def append(
        self,
        *,
        kind: str,
        event_id: str,
        before_version: int,
        accepted: bool,
        payload: Mapping[str, Any],
        state: HandState,
        observed_at_ns: int | None = None,
    ) -> HandEvent:
        from .engine import state_to_dict

        previous_hash = self.events[-1].event_hash if self.events else "0" * 64
        unsigned = {
            "sequence": len(self.events),
            "event_id": event_id,
            "kind": kind,
            "observed_at_ns": observed_at_ns or time.monotonic_ns(),
            "before_version": before_version,
            "after_version": state.state_version,
            "accepted": accepted,
            "payload": dict(payload),
            "state_after": state_to_dict(state),
            "previous_hash": previous_hash,
        }
        event = HandEvent(**unsigned, event_hash=self._hash(unsigned))
        self.events.append(event)
        return event

    def verify(self) -> None:
        previous_hash = "0" * 64
        for sequence, event in enumerate(self.events):
            if event.sequence != sequence or event.previous_hash != previous_hash:
                raise ValueError("event log sequence/hash chain is invalid")
            if event.event_hash != self._hash(event.unsigned()):
                raise ValueError("event log content hash is invalid")
            previous_hash = event.event_hash

    def recover_state(self) -> HandState:
        from .engine import state_from_dict

        self.verify()
        if not self.events:
            raise ValueError("cannot recover an empty event log")
        return state_from_dict(self.events[-1].state_after)

    def to_jsonl(self) -> str:
        return "\n".join(
            json.dumps(
                {**event.unsigned(), "event_hash": event.event_hash},
                sort_keys=True,
                ensure_ascii=False,
            )
            for event in self.events
        )

    @classmethod
    def from_jsonl(cls, text: str) -> EventLog:
        log = cls()
        for line in text.splitlines():
            if line.strip():
                log.events.append(HandEvent(**json.loads(line)))
        log.verify()
        return log


__all__ = ["EventLog", "HandEvent"]
