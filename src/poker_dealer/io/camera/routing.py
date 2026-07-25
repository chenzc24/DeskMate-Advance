"""State-directed routing between player and table cameras."""

from __future__ import annotations

from enum import StrEnum

from .adapter import CameraRead, OpenCVCamera


class CameraRoute(StrEnum):
    """Perception-owned camera roles; models never select these routes."""

    PLAYER = "player"
    TABLE = "table"


class RoutedOpenCVCamera:
    """Expose two cameras through the existing single-camera read contract."""

    def __init__(
        self,
        *,
        player_camera: OpenCVCamera,
        table_camera: OpenCVCamera,
        initial_route: CameraRoute = CameraRoute.PLAYER,
    ) -> None:
        self.player_camera = player_camera
        self.table_camera = table_camera
        self._active_route = CameraRoute(initial_route)

    @property
    def active_route(self) -> CameraRoute:
        return self._active_route

    @property
    def active_camera(self) -> OpenCVCamera:
        if self._active_route is CameraRoute.PLAYER:
            return self.player_camera
        return self.table_camera

    @property
    def is_open(self) -> bool:
        return all(camera.is_open for camera in self._unique_cameras())

    @property
    def network_reconnects(self) -> int:
        return self.active_camera.network_reconnects

    def select_route(self, route: CameraRoute | str) -> None:
        self._active_route = CameraRoute(route)

    def open(self) -> RoutedOpenCVCamera:
        opened_here: list[OpenCVCamera] = []
        try:
            for camera in self._unique_cameras():
                if camera.is_open:
                    continue
                camera.open()
                opened_here.append(camera)
        except Exception:
            for camera in reversed(opened_here):
                camera.close()
            raise
        return self

    def read(self) -> CameraRead:
        return self.active_camera.read()

    def close(self) -> None:
        for camera in reversed(self._unique_cameras()):
            camera.close()

    def _unique_cameras(self) -> tuple[OpenCVCamera, ...]:
        if self.player_camera is self.table_camera:
            return (self.table_camera,)
        return (self.table_camera, self.player_camera)


__all__ = ["CameraRoute", "RoutedOpenCVCamera"]
