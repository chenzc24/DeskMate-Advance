from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from poker_dealer.domain import (
    ActionEvidenceState,
    DealerAckStatus,
    DealerCommandType,
    DealerDeviceState,
    DealerErrorCode,
    DealerTargetSlot,
    ObservationStatus,
    PlayerActionType,
    Rank,
    Seat,
    Suit,
    VisionSlot,
)


ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = ROOT / "configs" / "contracts"


def load_json(relative: str) -> object:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def validate(instance_path: str, schema_name: str) -> None:
    schema = load_json(f"configs/contracts/{schema_name}")
    instance = load_json(instance_path)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(instance)


def test_all_json_files_parse_and_all_schemas_are_valid() -> None:
    for path in ROOT.glob("configs/**/*.json"):
        json.loads(path.read_text(encoding="utf-8"))
    for path in CONTRACTS.glob("*.schema.json"):
        Draft202012Validator.check_schema(
            json.loads(path.read_text(encoding="utf-8"))
        )


def test_core_rules_layout_decisions_and_walkthroughs_validate() -> None:
    validate("configs/game/core_v1.json", "core_rules.schema.json")
    validate("configs/table/logical_layout_v1.json", "table_layout.schema.json")
    validate("configs/contracts/stage0_decisions.json", "stage0_decisions.schema.json")
    validate("configs/game/stage0_walkthroughs.json", "walkthroughs.schema.json")


def test_contract_examples_validate() -> None:
    pairs = {
        "configs/contracts/examples/action_observation.candidate.json": "action_observation.schema.json",
        "configs/contracts/examples/card_observation.confirmed.json": "card_observation.schema.json",
        "configs/contracts/examples/dealer_command.rotate.json": "dealer_message.schema.json",
        "configs/contracts/examples/dealer_ack.failed.json": "dealer_message.schema.json",
        "configs/contracts/examples/player_action.raise.json": "player_action.schema.json",
        "configs/contracts/examples/hand_snapshot.preflop.json": "hand_snapshot.schema.json",
    }
    for instance, schema in pairs.items():
        validate(instance, schema)


def test_layout_ids_match_python_domain_enums_exactly() -> None:
    layout = load_json("configs/table/logical_layout_v1.json")
    mechanical = {item["target_id"] for item in layout["mechanical_targets"]}
    vision = {item["slot_id"] for item in layout["vision_slots"]}
    action_seats = {item["seat_id"] for item in layout["action_regions"]}
    action_regions = {item["region_id"] for item in layout["action_regions"]}
    assert mechanical == {item.value for item in DealerTargetSlot}
    assert vision == {item.value for item in VisionSlot}
    assert action_seats == {item.value for item in Seat}
    assert action_regions == {f"{item.value}_action" for item in Seat}


def test_machine_schemas_match_python_vocabulary_exactly() -> None:
    rules = load_json("configs/game/core_v1.json")
    assert set(rules["table"]["seats_clockwise"]) == {item.value for item in Seat}
    assert set(rules["betting"]["actions_semantics_frozen"]) == {
        item.value for item in PlayerActionType
    }

    dealer = load_json("configs/contracts/dealer_message.schema.json")["$defs"]
    assert set(dealer["target"]["enum"]) == {
        item.value for item in DealerTargetSlot
    }
    assert set(dealer["command"]["properties"]["command"]["enum"]) == {
        item.value for item in DealerCommandType
    }
    ack_properties = dealer["ack"]["properties"]
    assert set(ack_properties["status"]["enum"]) == {
        item.value for item in DealerAckStatus
    }
    assert set(ack_properties["device_state"]["enum"]) == {
        item.value for item in DealerDeviceState
    }
    assert set(ack_properties["error_code"]["oneOf"][0]["enum"]) == {
        item.value for item in DealerErrorCode
    }

    card = load_json("configs/contracts/card_observation.schema.json")["$defs"]
    assert set(card["vision_slot"]["enum"]) == {
        item.value for item in VisionSlot
    }
    assert set(card["card"]["properties"]["rank"]["enum"]) == {
        item.value for item in Rank
    }
    assert set(card["card"]["properties"]["suit"]["enum"]) == {
        item.value for item in Suit
    }
    card_status = load_json("configs/contracts/card_observation.schema.json")
    assert set(card_status["properties"]["status"]["enum"]) == {
        item.value for item in ObservationStatus
    }

    action = load_json("configs/contracts/action_observation.schema.json")["$defs"]
    assert set(action["seat"]["enum"]) == {item.value for item in Seat}
    assert set(action["action"]["enum"]) == {
        item.value for item in PlayerActionType
    }
    assert set(action["evidence_state"]["enum"]) == {
        item.value for item in ActionEvidenceState
    }
    assert action["evidence_state"]["enum"] == rules["behavior_perception"]["evidence_states"]
    assert action["action"]["enum"] == rules["behavior_perception"]["semantic_candidates"]

    snapshot = load_json("configs/contracts/hand_snapshot.schema.json")
    assert snapshot["$defs"]["slot_lifecycle"]["enum"] == rules["table_scene"]["slot_lifecycle"]


def test_decision_register_is_complete_and_honest_about_open_evidence() -> None:
    register = load_json("configs/contracts/stage0_decisions.json")
    decisions = register["decisions"]
    assert {item["id"] for item in decisions} == {
        f"S0-{number:02d}" for number in range(1, 23)
    }
    assert len({item["id"] for item in decisions}) == len(decisions)
    assert any(item["status"] == "evidence_required" for item in decisions)
    for item in decisions:
        if item["status"] != "frozen":
            assert item["blocks"], item["id"]
    status_counts = {
        status: sum(item["status"] == status for item in decisions)
        for status in ("frozen", "partially_frozen", "evidence_required")
    }
    assert status_counts == {
        "frozen": 5,
        "partially_frozen": 12,
        "evidence_required": 5,
    }


def test_walkthrough_matrix_covers_required_rule_and_fault_cases() -> None:
    walkthroughs = load_json("configs/game/stage0_walkthroughs.json")["scenarios"]
    assert len(walkthroughs) == 18
    assert len({item["id"] for item in walkthroughs}) == len(walkthroughs)
    tags = {tag for item in walkthroughs for tag in item["tags"]}
    required = {
        "fold",
        "check_through",
        "raise_cap",
        "all_in",
        "tie",
        "vision_unknown",
        "duplicate_card",
        "dealer_jam",
        "timeout",
        "duplicate_ack",
        "misdeal",
        "side_pot",
        "position_rotation",
        "attention_gating",
        "action_confirmation",
        "scene_lifecycle",
        "ledger_audit",
    }
    assert required <= tags


def test_fixed_limit_core_defaults_are_internally_consistent() -> None:
    rules = load_json("configs/game/core_v1.json")
    assert rules["table"]["players"] == 4
    assert rules["betting"]["product_decision_status"] == "confirmed_core_v1"
    assert rules["betting"]["structure"] == "fixed_limit"
    assert rules["session_defaults"]["initial_button_policy"] == "explicit_per_session"
    assert rules["betting"]["numeric_values_status"] == "configurable_defaults"
    assert rules["blinds_defaults"]["small_blind_units"] == 1
    assert rules["blinds_defaults"]["big_blind_units"] == 2
    assert rules["betting"]["small_bet_units_default"] == rules["blinds_defaults"]["big_blind_units"]
    assert rules["betting"]["big_bet_units_default"] == 2 * rules["betting"]["small_bet_units_default"]
    assert rules["betting"]["max_full_bets_per_street_default"] == 4
    assert rules["session_defaults"]["starting_stack_units"] >= rules["session_defaults"]["minimum_stack_to_start_hand_units"]
    assert rules["betting"]["main_and_side_pots_required"] is True
    assert rules["behavior_perception"]["acting_seat_authority"] == "deterministic_game_state_machine"
    assert rules["behavior_perception"]["model_output_contract"] == "player_action_observation_evidence_only"
    identity = rules["behavior_perception"]
    assert identity["face_identity_allowed"] is True
    assert identity["face_identity_scope"] == "session_only_explicit_enrollment"
    assert identity["face_embedding_persistence"] == "memory_only_never_serialized"
    assert identity["face_identity_authority"] == "verification_only_never_selects_acting_seat"
    assert rules["table_scene"]["phase_driven_roi_activation"] is True
    assert rules["ledger"]["physical_chip_recognition"] is False
    assert rules["ledger"]["append_only_hand_log_required"] is True


def test_schemas_reject_guessing_or_ambiguous_physical_messages() -> None:
    action_schema = load_json("configs/contracts/action_observation.schema.json")
    bad_action = load_json("configs/contracts/examples/action_observation.candidate.json")
    bad_action["evidence_state"] = "ambiguous"
    with pytest.raises(ValidationError):
        Draft202012Validator(action_schema).validate(bad_action)

    card_schema = load_json("configs/contracts/card_observation.schema.json")
    bad_card = load_json("configs/contracts/examples/card_observation.confirmed.json")
    bad_card["card"] = None
    with pytest.raises(ValidationError):
        Draft202012Validator(card_schema).validate(bad_card)

    face_down = load_json("configs/contracts/examples/card_observation.confirmed.json")
    face_down["status"] = "face_down"
    face_down["card"] = None
    Draft202012Validator(card_schema).validate(face_down)
    face_down["card"] = {"rank": "A", "suit": "spades"}
    with pytest.raises(ValidationError):
        Draft202012Validator(card_schema).validate(face_down)

    dealer_schema = load_json("configs/contracts/dealer_message.schema.json")
    bad_command = load_json("configs/contracts/examples/dealer_command.rotate.json")
    bad_command["target_slot"] = None
    with pytest.raises(ValidationError):
        Draft202012Validator(dealer_schema).validate(bad_command)

    removed_burn_target = load_json("configs/contracts/examples/dealer_command.rotate.json")
    removed_burn_target["target_slot"] = "burn_tray"
    with pytest.raises(ValidationError):
        Draft202012Validator(dealer_schema).validate(removed_burn_target)

    bad_ack = load_json("configs/contracts/examples/dealer_ack.failed.json")
    bad_ack["status"] = "succeeded"
    with pytest.raises(ValidationError):
        Draft202012Validator(dealer_schema).validate(bad_ack)

    good_dispense = load_json("configs/contracts/examples/dealer_ack.failed.json")
    good_dispense["status"] = "succeeded"
    good_dispense["device_state"] = "ready"
    good_dispense["error_code"] = None
    good_dispense["reason"] = None
    good_dispense["sensor_evidence"]["exit_pulses"] = 1
    Draft202012Validator(dealer_schema).validate(good_dispense)
    good_dispense["sensor_evidence"]["exit_pulses"] = 0
    with pytest.raises(ValidationError):
        Draft202012Validator(dealer_schema).validate(good_dispense)


def test_fault_walkthroughs_never_settle_or_advance_button() -> None:
    walkthroughs = load_json("configs/game/stage0_walkthroughs.json")["scenarios"]
    fault_tags = {"vision_unknown", "duplicate_card", "dealer_jam", "timeout"}
    for scenario in walkthroughs:
        if fault_tags.intersection(scenario["tags"]):
            assert scenario["expected"]["terminal_phase"] == "paused_recovery"
            assert scenario["expected"]["next_button"] is None
