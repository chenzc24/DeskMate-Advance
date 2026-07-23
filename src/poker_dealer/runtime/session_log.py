"""Hash-chained audit and independent checks for a multi-hand session."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping

from poker_dealer.domain import SEAT_ORDER, Seat


@dataclass(frozen=True, slots=True)
class SessionAuditEvent:
    sequence: int
    observed_at_ns: int
    kind: str
    payload: Mapping[str, object]
    previous_hash: str
    event_hash: str

    def unsigned(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "observed_at_ns": self.observed_at_ns,
            "kind": self.kind,
            "payload": dict(self.payload),
            "previous_hash": self.previous_hash,
        }


class SessionEventLog:
    def __init__(self) -> None:
        self.events: list[SessionAuditEvent] = []

    @staticmethod
    def _hash(value: Mapping[str, object]) -> str:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def append(
        self,
        kind: str,
        payload: Mapping[str, object],
        *,
        observed_at_ns: int | None = None,
    ) -> SessionAuditEvent:
        if not kind.strip():
            raise ValueError("session event kind is required")
        previous_hash = self.events[-1].event_hash if self.events else "0" * 64
        unsigned: dict[str, object] = {
            "sequence": len(self.events),
            "observed_at_ns": (
                time.monotonic_ns() if observed_at_ns is None else observed_at_ns
            ),
            "kind": kind,
            "payload": dict(payload),
            "previous_hash": previous_hash,
        }
        event = SessionAuditEvent(
            **unsigned, event_hash=self._hash(unsigned)  # type: ignore[arg-type]
        )
        self.events.append(event)
        return event

    def verify(self) -> None:
        previous_hash = "0" * 64
        previous_time = -1
        for sequence, event in enumerate(self.events):
            if event.sequence != sequence or event.previous_hash != previous_hash:
                raise ValueError("session event sequence/hash chain is invalid")
            if event.observed_at_ns < previous_time:
                raise ValueError("session event timestamps are not monotonic")
            if event.event_hash != self._hash(event.unsigned()):
                raise ValueError("session event content hash is invalid")
            previous_hash = event.event_hash
            previous_time = event.observed_at_ns

    def to_jsonl(self) -> str:
        self.verify()
        return "\n".join(
            json.dumps(
                {**event.unsigned(), "event_hash": event.event_hash},
                sort_keys=True,
                ensure_ascii=False,
            )
            for event in self.events
        )

    @classmethod
    def from_jsonl(cls, text: str) -> SessionEventLog:
        log = cls()
        for line in text.splitlines():
            if line.strip():
                log.events.append(SessionAuditEvent(**json.loads(line)))
        log.verify()
        return log

    @classmethod
    def from_path(cls, path: str | Path) -> SessionEventLog:
        return cls.from_jsonl(Path(path).read_text(encoding="utf-8"))


class SessionEventWriter:
    """Exclusively create and incrementally sync one ignored session JSONL."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("x", encoding="utf-8", newline="\n")
        self._written = 0

    def sync(self, log: SessionEventLog) -> None:
        log.verify()
        for event in log.events[self._written :]:
            self._stream.write(
                json.dumps(
                    {**event.unsigned(), "event_hash": event.event_hash},
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
            )
            self._written += 1
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()

    def __enter__(self) -> SessionEventWriter:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class SessionLogCheck:
    passed: bool
    session_id: str | None
    hands_started: int
    hands_closed: int
    ended: bool
    issues: tuple[str, ...]


def check_session_log(
    log: SessionEventLog, *, verify_hand_logs: bool = False
) -> SessionLogCheck:
    issues: list[str] = []
    try:
        log.verify()
    except ValueError as exc:
        return SessionLogCheck(False, None, 0, 0, False, (str(exc),))
    if not log.events or log.events[0].kind != "session_started":
        return SessionLogCheck(
            False, None, 0, 0, False, ("session_started_missing",)
        )
    started = log.events[0].payload
    session_id = str(started.get("session_id", "")) or None
    try:
        button = Seat(str(started["button"]))
        stacks = _stacks(started["stacks"])
    except (KeyError, TypeError, ValueError) as exc:
        return SessionLogCheck(
            False, session_id, 0, 0, False, (f"invalid_session_start:{exc}",)
        )
    active_hand: str | None = None
    table_cleared = True
    ended = False
    hand_ids: set[str] = set()
    hands_started = 0
    hands_closed = 0
    for event in log.events[1:]:
        payload = event.payload
        if ended:
            issues.append(f"event_after_session_end:{event.sequence}")
            continue
        if event.kind == "hand_started":
            hand_id = str(payload.get("hand_id", ""))
            if active_hand is not None or not table_cleared:
                issues.append(f"hand_started_outside_ready_state:{hand_id}")
            if not hand_id or hand_id in hand_ids:
                issues.append(f"duplicate_or_empty_hand_id:{hand_id}")
            if payload.get("button") != button.value:
                issues.append(f"hand_button_discontinuity:{hand_id}")
            try:
                if _stacks(payload["starting_stacks"]) != stacks:
                    issues.append(f"hand_stack_discontinuity:{hand_id}")
            except (KeyError, TypeError, ValueError):
                issues.append(f"hand_starting_stacks_invalid:{hand_id}")
            active_hand = hand_id
            table_cleared = False
            hand_ids.add(hand_id)
            hands_started += 1
        elif event.kind == "hand_closed":
            hand_id = str(payload.get("hand_id", ""))
            if active_hand != hand_id:
                issues.append(f"hand_closed_without_matching_start:{hand_id}")
            terminal = str(payload.get("terminal_phase", ""))
            expected_button = (
                button
                if terminal == "voided"
                else SEAT_ORDER[(SEAT_ORDER.index(button) + 1) % len(SEAT_ORDER)]
                if terminal == "settled"
                else None
            )
            if expected_button is None:
                issues.append(f"invalid_terminal_phase:{hand_id}:{terminal}")
            elif payload.get("button_after") != expected_button.value:
                issues.append(f"button_after_invalid:{hand_id}")
            if payload.get("button_before") != button.value:
                issues.append(f"button_before_invalid:{hand_id}")
            try:
                closed_stacks = _stacks(payload["stacks"])
                if sum(closed_stacks.values()) != sum(stacks.values()):
                    issues.append(f"hand_ledger_not_conserved:{hand_id}")
                stacks = closed_stacks
            except (KeyError, TypeError, ValueError):
                issues.append(f"hand_closed_stacks_invalid:{hand_id}")
            if not payload.get("hand_log_sha256"):
                issues.append(f"hand_log_hash_missing:{hand_id}")
            if payload.get("hand_log_check_passed") is not True:
                issues.append(f"hand_log_check_not_passed:{hand_id}")
            if verify_hand_logs:
                _verify_hand_log_reference(payload, hand_id, terminal, issues)
            if expected_button is not None:
                button = expected_button
            active_hand = None
            hands_closed += 1
        elif event.kind == "table_cleared":
            if active_hand is not None or table_cleared:
                issues.append(f"table_clear_outside_terminal_boundary:{event.sequence}")
            table_cleared = True
        elif event.kind == "stack_adjusted":
            if active_hand is not None:
                issues.append(f"stack_adjusted_during_hand:{event.sequence}")
            try:
                seat = Seat(str(payload["seat"]))
                amount = int(payload["amount_units"])
                stacks[seat] += amount
                if stacks[seat] < 0 or stacks[seat] != int(payload["stack_after"]):
                    raise ValueError("stack_after mismatch")
            except (KeyError, TypeError, ValueError) as exc:
                issues.append(f"invalid_stack_adjustment:{event.sequence}:{exc}")
        elif event.kind == "session_ended":
            if active_hand is not None or not table_cleared:
                issues.append("session_ended_before_safe_boundary")
            try:
                if _stacks(payload["final_stacks"]) != stacks:
                    issues.append("session_final_stacks_mismatch")
            except (KeyError, TypeError, ValueError):
                issues.append("session_final_stacks_invalid")
            ended = True
        elif event.kind not in {"recovery_decision"}:
            issues.append(f"unknown_session_event:{event.kind}")
    if active_hand is not None:
        issues.append(f"active_hand_at_log_end:{active_hand}")
    if hands_started != hands_closed:
        issues.append("session_hand_count_mismatch")
    if not ended:
        issues.append("session_not_ended")
    return SessionLogCheck(
        not issues,
        session_id,
        hands_started,
        hands_closed,
        ended,
        tuple(issues),
    )


def _stacks(value: object) -> dict[Seat, int]:
    if not isinstance(value, Mapping):
        raise TypeError("stacks must be an object")
    stacks = {Seat(str(seat)): int(amount) for seat, amount in value.items()}
    if set(stacks) != set(SEAT_ORDER) or min(stacks.values()) < 0:
        raise ValueError("stacks must contain four non-negative balances")
    return stacks


def _verify_hand_log_reference(
    payload: Mapping[str, object],
    hand_id: str,
    terminal: str,
    issues: list[str],
) -> None:
    from .event_log import RuntimeEventLog, check_runtime_hand_log

    raw_path = payload.get("hand_log_path")
    if not isinstance(raw_path, str) or not raw_path:
        issues.append(f"hand_log_path_missing:{hand_id}")
        return
    path = Path(raw_path)
    if not path.is_file():
        issues.append(f"hand_log_file_missing:{hand_id}")
        return
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != payload.get("hand_log_sha256"):
        issues.append(f"hand_log_hash_mismatch:{hand_id}")
        return
    try:
        checked = check_runtime_hand_log(
            RuntimeEventLog.from_path(path),
            allow_voided=terminal == "voided",
        )
    except (OSError, TypeError, ValueError) as exc:
        issues.append(f"hand_log_unreadable:{hand_id}:{exc}")
        return
    if not checked.passed:
        issues.append(f"hand_log_recheck_failed:{hand_id}")


__all__ = [
    "SessionAuditEvent",
    "SessionEventLog",
    "SessionEventWriter",
    "SessionLogCheck",
    "check_session_log",
]
