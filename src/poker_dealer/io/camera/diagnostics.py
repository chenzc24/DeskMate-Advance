"""Bounded camera probing and non-recording live preview helpers."""

from __future__ import annotations

import cv2

from .adapter import (
    CameraConfig,
    CameraError,
    CameraReadStatus,
    OpenCVCamera,
)


def probe_camera_indices(
    indexes: list[int],
    *,
    backend: str,
    width: int,
    height: int,
    fps: float,
) -> list[dict[str, object]]:
    """Read one frame per index and return reports without saving images."""

    reports: list[dict[str, object]] = []
    for index in indexes:
        config = CameraConfig(
            device_index=index,
            source_id=f"camera-{index}",
            backend=backend,
            width=width,
            height=height,
            fps=fps,
            disconnect_after_failures=1,
        )
        try:
            with OpenCVCamera(config) as camera:
                result = camera.read()
                report: dict[str, object] = camera.negotiated_properties()
                report.update(
                    {
                        "read_status": result.status.value,
                        "observed_at_ns": result.observed_at_ns,
                        "error": result.reason,
                    }
                )
        except CameraError as error:
            report = {
                "device_index": index,
                "backend": backend,
                "read_status": "unavailable",
                "error": str(error),
            }
        reports.append(report)
    return reports


def run_camera_preview(config: CameraConfig, *, title: str) -> int:
    """Show live frames until Q, Escape, or window close; never record them."""

    try:
        with OpenCVCamera(config) as camera:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            while True:
                result = camera.read()
                if result.status is CameraReadStatus.OK and result.frame is not None:
                    display = result.frame.image.copy()
                    cv2.putText(
                        display,
                        (
                            f"camera={config.device_index}  "
                            f"{result.frame.width}x{result.frame.height}"
                        ),
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(title, display)
                elif result.status is CameraReadStatus.DISCONNECTED:
                    return 2

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    return 0
                if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    return 0
    except CameraError as error:
        print(error)
        return 1
    finally:
        cv2.destroyAllWindows()
