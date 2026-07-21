from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from poker_dealer.game import run_random_hands, run_walkthroughs


ROOT = Path(__file__).resolve().parents[2]


def test_all_18_stage0_walkthroughs_execute_and_match_expected() -> None:
    results = run_walkthroughs(ROOT / "configs/game/stage0_walkthroughs.json")
    assert len(results) == 18
    assert all(result.passed for result in results), {
        result.scenario_id: result.mismatches
        for result in results
        if not result.passed
    }


def test_10000_seeded_legal_hands_settle_without_invariant_failure() -> None:
    summary = run_random_hands(10_000, seed=20260721)
    assert summary.hands == 10_000
    assert summary.actions > 10_000
    assert summary.showdowns > 0
    assert summary.folds > 0


def test_no_device_demo_cli_completes_a_visible_four_player_hand() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/game/demo_stage1.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["betting_product_status"] == "candidate_pending_confirmation"
    assert result["final_snapshot"]["phase"] == "settled"
    assert result["final_snapshot"]["pot_units"] == 0
    assert len(result["final_snapshot"]["confirmed_cards"]) == 13
    assert result["event_count"] > 20
    assert len(result["event_log_tail_hash"]) == 64
