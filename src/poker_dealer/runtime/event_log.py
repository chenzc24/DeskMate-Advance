"""Append-only runtime evidence log and independent settled-hand checker."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import IO, Any, Mapping

from poker_dealer.domain import HandPhase
from poker_dealer.game import EventLog, HandEvent, settle_showdown, state_from_dict


@dataclass(frozen=True, slots=True)
class RuntimeLogRecord:
    sequence: int
    record_type: str
    kind: str
    observed_at_ns: int
    payload: Mapping[str, Any]
    previous_hash: str
    record_hash: str

    def unsigned(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "record_type": self.record_type,
            "kind": self.kind,
            "observed_at_ns": self.observed_at_ns,
            "payload": self.payload,
            "previous_hash": self.previous_hash,
        }


class RuntimeEventWriter:
    """Write evidence and copied engine events without overwriting prior runs."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.records: list[RuntimeLogRecord] = []
        self._engine_events_written = 0
        self._file: IO[str] | None = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = path.open("x", encoding="utf-8", newline="\n")

    @staticmethod
    def _hash(unsigned: Mapping[str, Any]) -> str:
        encoded = json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def emit(
        self,
        kind: str,
        *,
        observed_at_ns: int,
        payload: Mapping[str, object],
    ) -> None:
        self._append("runtime_evidence", kind, observed_at_ns, payload)

    def sync_engine(self, engine_log: EventLog) -> None:
        while self._engine_events_written < len(engine_log.events):
            event = engine_log.events[self._engine_events_written]
            self._append(
                "engine_event",
                event.kind,
                event.observed_at_ns,
                {"event": {**event.unsigned(), "event_hash": event.event_hash}},
            )
            self._engine_events_written += 1

    def _append(
        self,
        record_type: str,
        kind: str,
        observed_at_ns: int,
        payload: Mapping[str, object],
    ) -> None:
        if not kind.strip() or observed_at_ns < 0:
            raise ValueError("runtime log kind and timestamp must be valid")
        previous_hash = self.records[-1].record_hash if self.records else "0" * 64
        unsigned = {
            "sequence": len(self.records),
            "record_type": record_type,
            "kind": kind,
            "observed_at_ns": observed_at_ns,
            "payload": dict(payload),
            "previous_hash": previous_hash,
        }
        record = RuntimeLogRecord(**unsigned, record_hash=self._hash(unsigned))
        self.records.append(record)
        if self._file is not None:
            self._file.write(
                json.dumps(
                    {**record.unsigned(), "record_hash": record.record_hash},
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
            )
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> RuntimeEventWriter:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class RuntimeEventLog:
    def __init__(self, records: tuple[RuntimeLogRecord, ...]) -> None:
        self.records = records

    @classmethod
    def from_jsonl(cls, text: str) -> RuntimeEventLog:
        records = tuple(
            RuntimeLogRecord(**json.loads(line))
            for line in text.splitlines()
            if line.strip()
        )
        value = cls(records)
        value.verify()
        return value

    @classmethod
    def from_path(cls, path: Path) -> RuntimeEventLog:
        return cls.from_jsonl(path.read_text(encoding="utf-8"))

    def verify(self) -> None:
        previous_hash = "0" * 64
        for sequence, record in enumerate(self.records):
            if record.sequence != sequence or record.previous_hash != previous_hash:
                raise ValueError("runtime log sequence/hash chain is invalid")
            if record.record_hash != RuntimeEventWriter._hash(record.unsigned()):
                raise ValueError("runtime log content hash is invalid")
            previous_hash = record.record_hash
        self.engine_log().verify()

    def engine_log(self) -> EventLog:
        log = EventLog()
        for record in self.records:
            if record.record_type != "engine_event":
                continue
            raw = record.payload.get("event")
            if not isinstance(raw, Mapping):
                raise ValueError("engine-event runtime record is malformed")
            log.events.append(HandEvent(**dict(raw)))
        if not log.events:
            raise ValueError("runtime log has no engine events")
        return log

    def evidence(self, kind: str | None = None) -> tuple[RuntimeLogRecord, ...]:
        return tuple(
            record
            for record in self.records
            if record.record_type == "runtime_evidence"
            and (kind is None or record.kind == kind)
        )


@dataclass(frozen=True, slots=True)
class HandLogCheck:
    passed: bool
    hand_id: str | None
    phase: str | None
    engine_events: int
    evidence_records: int
    issues: tuple[str, ...]


def check_runtime_hand_log(
    log: RuntimeEventLog, *, require_settled: bool = True
) -> HandLogCheck:
    """Recompute invariants instead of trusting a claimed winner or balance."""

    issues: list[str] = []
    try:
        log.verify()
        engine_log = log.engine_log()
    except ValueError as exc:
        return HandLogCheck(False, None, None, 0, 0, (str(exc),))
    events = engine_log.events
    for previous, current in zip(events, events[1:]):
        if current.before_version != previous.after_version:
            issues.append(
                f"state_version_discontinuity:{previous.sequence}->{current.sequence}"
            )
    initial = state_from_dict(events[0].state_after)
    final = engine_log.recover_state()
    if require_settled and final.phase is not HandPhase.SETTLED:
        issues.append(f"hand_not_settled:{final.phase.value}")
    if initial.total_units() != final.total_units():
        issues.append("digital_ledger_not_conserved")
    visible_cards = tuple(final.board) + tuple(
        card for cards in final.hole_cards.values() for card in cards
    )
    if len(visible_cards) != len(set(visible_cards)):
        issues.append("duplicate_card_identity")
    issued = {
        event.event_id
        for event in events
        if event.kind == "dealer_command_issued"
    }
    completed_events = [
        event for event in events if event.kind == "dealer_command_completed"
    ]
    acknowledged = {
        str(event.payload.get("command_id"))
        for event in completed_events
    }
    if issued != acknowledged:
        issues.append("dealer_command_ack_set_mismatch")
    issued_by_id = {
        event.event_id: event
        for event in events
        if event.kind == "dealer_command_issued"
    }
    previous_device_version = -1
    for event in completed_events:
        command_id = str(event.payload.get("command_id"))
        issued_event = issued_by_id.get(command_id)
        if issued_event is None:
            continue
        if (
            event.payload.get("command") != issued_event.payload.get("command")
            or event.payload.get("target_slot")
            != issued_event.payload.get("target_slot")
        ):
            issues.append(f"dealer_ack_correlation_mismatch:{command_id}")
        version = int(event.payload.get("device_state_version", -1))
        if version <= previous_device_version:
            issues.append(f"dealer_device_version_not_monotonic:{command_id}")
        previous_device_version = version
        evidence = event.payload.get("sensor_evidence")
        required_evidence = {
            "homed",
            "at_target",
            "deck_present",
            "exit_pulses",
            "interlock_closed",
            "emergency_stop",
        }
        if not isinstance(evidence, dict) or not required_evidence.issubset(evidence):
            issues.append(f"dealer_ack_sensor_evidence_missing:{command_id}")
    if final.pending_command_id is not None:
        issues.append("pending_dealer_command_at_log_end")
    if final.phase is HandPhase.SETTLED and final.board and final.hole_cards:
        recomputed = settle_showdown(
            final.pots,
            final.board,
            final.hole_cards,
            final.button,
        )
        if dict(recomputed.awards) != dict(final.awards):
            issues.append("showdown_awards_do_not_match_recomputed_result")
    return HandLogCheck(
        passed=not issues,
        hand_id=final.hand_id,
        phase=final.phase.value,
        engine_events=len(events),
        evidence_records=len(log.evidence()),
        issues=tuple(issues),
    )


__all__ = [
    "HandLogCheck",
    "RuntimeEventLog",
    "RuntimeEventWriter",
    "RuntimeLogRecord",
    "check_runtime_hand_log",
]
