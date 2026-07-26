"""Finite table-route planner for the four-player I-shaped layout."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum

from poker_dealer.domain import RobotPoseNode, Seat, TableRole, role_for_seat


class CarApiAction(StrEnum):
    """Actions exposed by cocino_car's formal robotics HTTP facade."""

    FOLLOW_LINE_TO_END = "follow_line_to_end"
    FACE_TURN = "face_turn"
    LINE_RECENTER = "line_recenter"
    PRESET_TURN = "preset_turn"
    FOLLOW_LINE_TO_BOARD = "follow_line_to_board"


class TurnDirection(StrEnum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"


@dataclass(frozen=True, slots=True)
class RoutePrimitive:
    action: CarApiAction
    start_pose: RobotPoseNode
    target_pose: RobotPoseNode
    direction: TurnDirection | None = None
    degrees: int | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.start_pose is self.target_pose:
            raise ValueError("route primitive must change pose")
        if self.action in {
            CarApiAction.FACE_TURN,
            CarApiAction.LINE_RECENTER,
            CarApiAction.PRESET_TURN,
        } and self.direction is None:
            raise ValueError(f"{self.action.value} requires a direction")
        if self.action is CarApiAction.PRESET_TURN and self.degrees not in {90, 180}:
            raise ValueError("preset_turn requires 90 or 180 degrees")
        if self.action is not CarApiAction.PRESET_TURN and self.degrees is not None:
            raise ValueError("degrees are only valid for preset_turn")


def player_pose(button: Seat, seat: Seat) -> RobotPoseNode:
    """Resolve a fixed seat to its physical role pose for this hand."""

    return {
        TableRole.BUTTON: RobotPoseNode.INIT_BUTTON,
        TableRole.SMALL_BLIND: RobotPoseNode.END_SMALL_BLIND,
        TableRole.BIG_BLIND: RobotPoseNode.END_BIG_BLIND,
        TableRole.UNDER_THE_GUN: RobotPoseNode.INIT_UTG,
    }[role_for_seat(button, seat)]


def hole_deal_pose(button: Seat, seat: Seat) -> RobotPoseNode:
    """Return the line-facing pose used by the left-side dispenser."""

    return {
        TableRole.BUTTON: RobotPoseNode.INIT_TO_END,
        TableRole.SMALL_BLIND: RobotPoseNode.END_TO_END,
        TableRole.BIG_BLIND: RobotPoseNode.END_TO_INIT,
        TableRole.UNDER_THE_GUN: RobotPoseNode.INIT_TO_INIT,
    }[role_for_seat(button, seat)]


def _edge(
    action: CarApiAction,
    start: RobotPoseNode,
    target: RobotPoseNode,
    *,
    direction: TurnDirection | None = None,
    degrees: int | None = None,
    note: str = "",
) -> RoutePrimitive:
    return RoutePrimitive(action, start, target, direction, degrees, note)


_EDGES: tuple[RoutePrimitive, ...] = (
    _edge(
        CarApiAction.FOLLOW_LINE_TO_END,
        RobotPoseNode.INIT_TO_END,
        RobotPoseNode.END_TO_END,
        note="init to end",
    ),
    _edge(
        CarApiAction.FOLLOW_LINE_TO_END,
        RobotPoseNode.END_TO_INIT,
        RobotPoseNode.INIT_TO_INIT,
        note="end to init; API name describes endpoint detection, not direction",
    ),
    _edge(
        CarApiAction.FOLLOW_LINE_TO_BOARD,
        RobotPoseNode.INIT_TO_END,
        RobotPoseNode.BOARD_TO_END,
        note="required board marker stop; absent from cocino_car API v1.0",
    ),
    _edge(
        CarApiAction.FOLLOW_LINE_TO_END,
        RobotPoseNode.BOARD_TO_END,
        RobotPoseNode.END_TO_END,
        note="board to end",
    ),
    _edge(
        CarApiAction.PRESET_TURN,
        RobotPoseNode.END_TO_END,
        RobotPoseNode.END_TO_INIT,
        direction=TurnDirection.RIGHT,
        degrees=180,
        note=(
            "hole route after SB delivery: right toward BB, then right "
            "back to the line; no player operation"
        ),
    ),
    _edge(
        CarApiAction.FACE_TURN,
        RobotPoseNode.INIT_TO_END,
        RobotPoseNode.INIT_BUTTON,
        direction=TurnDirection.LEFT,
    ),
    _edge(
        CarApiAction.FACE_TURN,
        RobotPoseNode.INIT_TO_END,
        RobotPoseNode.INIT_UTG,
        direction=TurnDirection.RIGHT,
    ),
    _edge(
        CarApiAction.FACE_TURN,
        RobotPoseNode.INIT_TO_INIT,
        RobotPoseNode.INIT_UTG,
        direction=TurnDirection.LEFT,
    ),
    _edge(
        CarApiAction.FACE_TURN,
        RobotPoseNode.INIT_TO_INIT,
        RobotPoseNode.INIT_BUTTON,
        direction=TurnDirection.RIGHT,
    ),
    _edge(
        CarApiAction.FACE_TURN,
        RobotPoseNode.END_TO_END,
        RobotPoseNode.END_SMALL_BLIND,
        direction=TurnDirection.LEFT,
    ),
    _edge(
        CarApiAction.FACE_TURN,
        RobotPoseNode.END_TO_END,
        RobotPoseNode.END_BIG_BLIND,
        direction=TurnDirection.RIGHT,
    ),
    _edge(
        CarApiAction.FACE_TURN,
        RobotPoseNode.END_TO_INIT,
        RobotPoseNode.END_BIG_BLIND,
        direction=TurnDirection.LEFT,
    ),
    _edge(
        CarApiAction.FACE_TURN,
        RobotPoseNode.END_TO_INIT,
        RobotPoseNode.END_SMALL_BLIND,
        direction=TurnDirection.RIGHT,
    ),
    _edge(
        CarApiAction.LINE_RECENTER,
        RobotPoseNode.INIT_BUTTON,
        RobotPoseNode.INIT_TO_END,
        direction=TurnDirection.RIGHT,
    ),
    _edge(
        CarApiAction.LINE_RECENTER,
        RobotPoseNode.INIT_UTG,
        RobotPoseNode.INIT_TO_END,
        direction=TurnDirection.LEFT,
    ),
    _edge(
        CarApiAction.LINE_RECENTER,
        RobotPoseNode.END_SMALL_BLIND,
        RobotPoseNode.END_TO_INIT,
        direction=TurnDirection.LEFT,
    ),
    _edge(
        CarApiAction.LINE_RECENTER,
        RobotPoseNode.END_BIG_BLIND,
        RobotPoseNode.END_TO_INIT,
        direction=TurnDirection.RIGHT,
    ),
    _edge(
        CarApiAction.PRESET_TURN,
        RobotPoseNode.END_SMALL_BLIND,
        RobotPoseNode.END_BIG_BLIND,
        direction=TurnDirection.RIGHT,
        degrees=180,
        note="implemented as two right 90-degree API calls",
    ),
    _edge(
        CarApiAction.PRESET_TURN,
        RobotPoseNode.INIT_UTG,
        RobotPoseNode.INIT_BUTTON,
        direction=TurnDirection.RIGHT,
        degrees=180,
        note="implemented as two right 90-degree API calls",
    ),
)


class TableRoutePlanner:
    """Find the shortest legal directed path between canonical table poses."""

    def __init__(self, edges: tuple[RoutePrimitive, ...] = _EDGES) -> None:
        self.edges = edges
        self._outgoing: dict[RobotPoseNode, list[RoutePrimitive]] = {}
        for edge in edges:
            self._outgoing.setdefault(edge.start_pose, []).append(edge)

    def plan(
        self, start: RobotPoseNode, target: RobotPoseNode
    ) -> tuple[RoutePrimitive, ...]:
        if start is RobotPoseNode.UNKNOWN or target is RobotPoseNode.UNKNOWN:
            raise ValueError("route planning requires known canonical poses")
        if start is target:
            return ()
        queue = deque([(start, ())])
        visited = {start}
        while queue:
            pose, path = queue.popleft()
            for edge in self._outgoing.get(pose, ()):
                candidate = path + (edge,)
                if edge.target_pose is target:
                    return candidate
                if edge.target_pose not in visited:
                    visited.add(edge.target_pose)
                    queue.append((edge.target_pose, candidate))
        raise ValueError(f"no legal table route from {start.value} to {target.value}")


__all__ = [
    "CarApiAction",
    "RoutePrimitive",
    "TableRoutePlanner",
    "TurnDirection",
    "hole_deal_pose",
    "player_pose",
]
