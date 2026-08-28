"""OpenCV developer tool for homography calibration and scene-area drawing.

Controls:
  Calibration mode: click camera image, then the matching scene point.
  P: switch between calibration and polygon modes.
  H: compute/recompute homography and print its quality.
  U: undo the latest pending point, pair, or polygon vertex.
  R: reset the current mode's points.
  S: save matrix and YAML config (requires calibration and polygon).
  Q or Esc: quit.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from material_usage_perception.geometry import (
    CalibrationPointPair,
    compute_camera_to_scene_homography,
    validate_normalized_polygon,
)


class CalibrationUi:
    def __init__(
        self,
        camera_image: np.ndarray,
        scene_image: np.ndarray,
        *,
        camera_id: str,
        scene_id: str,
        area_id: str,
        output_dir: Path,
        max_display_height: int = 720,
    ) -> None:
        self.camera_original = camera_image
        self.scene_original = scene_image
        self.camera_id = camera_id
        self.scene_id = scene_id
        self.area_id = area_id
        self.output_dir = output_dir
        self.mode = "calibration"
        self.pairs: list[CalibrationPointPair] = []
        self.pending_camera: tuple[float, float] | None = None
        self.polygon: list[tuple[float, float]] = []
        self.result = None

        camera_scale = min(1.0, max_display_height / camera_image.shape[0])
        scene_scale = min(1.0, max_display_height / scene_image.shape[0])
        camera_resized = cv2.resize(
            camera_image, None, fx=camera_scale, fy=camera_scale
        )
        scene_resized = cv2.resize(scene_image, None, fx=scene_scale, fy=scene_scale)
        self.camera_content_size = (camera_resized.shape[1], camera_resized.shape[0])
        self.scene_content_size = (scene_resized.shape[1], scene_resized.shape[0])
        height = max(camera_resized.shape[0], scene_resized.shape[0])
        self.camera_display = self._pad_height(camera_resized, height)
        self.scene_display = self._pad_height(scene_resized, height)
        self.camera_display_width = self.camera_display.shape[1]

    @staticmethod
    def _pad_height(image: np.ndarray, height: int) -> np.ndarray:
        return cv2.copyMakeBorder(
            image,
            0,
            height - image.shape[0],
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(30, 30, 30),
        )

    def run(self) -> None:
        window = "Spatial calibration (camera | scene)"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, self.on_mouse)
        while True:
            cv2.imshow(window, self.render())
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("p"):
                self.mode = "polygon" if self.mode == "calibration" else "calibration"
                print(f"Mode: {self.mode}")
            elif key == ord("h"):
                self.compute()
            elif key == ord("u"):
                self.undo()
            elif key == ord("r"):
                self.reset_current_mode()
            elif key == ord("s"):
                self.save()
        cv2.destroyAllWindows()

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if x < self.camera_display_width:
            if y >= self.camera_content_size[1]:
                return
            point = (
                x / self.camera_content_size[0],
                y / self.camera_content_size[1],
            )
            if self.mode != "calibration":
                print("Polygon vertices must be selected on the scene image.")
                return
            self.pending_camera = point
            print(f"Camera point selected: {point}")
            return

        scene_x = x - self.camera_display_width
        if scene_x >= self.scene_display.shape[1]:
            return
        if y >= self.scene_content_size[1]:
            return
        point = (
            scene_x / self.scene_content_size[0],
            y / self.scene_content_size[1],
        )
        if self.mode == "polygon":
            self.polygon.append(point)
            print(f"Polygon vertex {len(self.polygon)}: {point}")
        elif self.pending_camera is None:
            print("Select the corresponding camera point first.")
        else:
            self.pairs.append(CalibrationPointPair(self.pending_camera, point))
            self.pending_camera = None
            self.result = None
            print(f"Calibration pair {len(self.pairs)} completed.")

    def compute(self) -> None:
        try:
            self.result = compute_camera_to_scene_homography(
                self.pairs,
                camera_image_size=self.camera_size,
                scene_image_size=self.scene_size,
            )
        except ValueError as exc:
            print(f"Calibration failed: {exc}")
            return
        quality = self.result.quality
        print(
            f"Calibration ready: {quality.inlier_count}/{quality.point_count} inliers, "
            f"mean error={quality.mean_reprojection_error_px:.2f}px, "
            f"max error={quality.max_reprojection_error_px:.2f}px"
        )

    @property
    def camera_size(self) -> tuple[float, float]:
        return float(self.camera_original.shape[1]), float(self.camera_original.shape[0])

    @property
    def scene_size(self) -> tuple[float, float]:
        return float(self.scene_original.shape[1]), float(self.scene_original.shape[0])

    def undo(self) -> None:
        if self.mode == "polygon" and self.polygon:
            self.polygon.pop()
        elif self.pending_camera is not None:
            self.pending_camera = None
        elif self.pairs:
            self.pairs.pop()
            self.result = None

    def reset_current_mode(self) -> None:
        if self.mode == "polygon":
            self.polygon.clear()
        else:
            self.pairs.clear()
            self.pending_camera = None
            self.result = None

    def save(self) -> None:
        if self.result is None:
            self.compute()
        if self.result is None:
            return
        try:
            validate_normalized_polygon(self.polygon)
        except ValueError as exc:
            print(f"Cannot save polygon: {exc}")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        matrix_name = f"{self.camera_id}_camera_to_scene.npy"
        matrix_path = self.output_dir / matrix_name
        config_path = self.output_dir / f"{self.camera_id}_spatial.yaml"
        np.save(matrix_path, self.result.matrix)

        scene_w, scene_h = self.scene_size
        config = {
            "camera": {"camera_id": self.camera_id},
            "scene": {"scene_id": self.scene_id},
            "projection": {
                "direction": "camera_to_scene",
                "matrix_path": matrix_name,
                "source_frame_size": list(self.camera_size),
                "calibration_source_size": list(self.camera_size),
                "calibration_target_size": list(self.scene_size),
                "scene_size": list(self.scene_size),
                "point_pairs": [
                    {
                        "camera_normalized": list(pair.camera_normalized),
                        "scene_normalized": list(pair.scene_normalized),
                    }
                    for pair in self.pairs
                ],
                "quality": asdict(self.result.quality),
            },
            "area": {
                "area_id": self.area_id,
                "coordinate_system": "scene",
                "polygon": [[x * scene_w, y * scene_h] for x, y in self.polygon],
                "polygon_normalized": [list(point) for point in self.polygon],
            },
        }
        with config_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
        print(f"Saved matrix: {matrix_path}")
        print(f"Saved config: {config_path}")

    def render(self) -> np.ndarray:
        camera = self.camera_display.copy()
        scene = self.scene_display.copy()
        for index, pair in enumerate(self.pairs, start=1):
            self._draw_point(
                camera, pair.camera_normalized, index, (0, 255, 0), self.camera_content_size
            )
            self._draw_point(
                scene, pair.scene_normalized, index, (0, 255, 0), self.scene_content_size
            )
        if self.pending_camera is not None:
            self._draw_point(
                camera,
                self.pending_camera,
                len(self.pairs) + 1,
                (0, 255, 255),
                self.camera_content_size,
            )

        polygon_pixels = [
            self._display_point(point, self.scene_content_size) for point in self.polygon
        ]
        for index, pixel in enumerate(polygon_pixels, start=1):
            cv2.circle(scene, pixel, 5, (255, 0, 255), -1)
            cv2.putText(scene, str(index), pixel, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        if len(polygon_pixels) >= 2:
            cv2.polylines(scene, [np.asarray(polygon_pixels)], len(polygon_pixels) >= 3, (255, 0, 255), 2)

        combined = np.hstack((camera, scene))
        status = f"mode={self.mode} pairs={len(self.pairs)} polygon={len(self.polygon)} | P mode H compute U undo R reset S save Q quit"
        cv2.rectangle(combined, (0, 0), (combined.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(combined, status, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        return combined

    @staticmethod
    def _display_point(
        normalized: tuple[float, float], content_size: tuple[int, int]
    ) -> tuple[int, int]:
        return int(normalized[0] * content_size[0]), int(normalized[1] * content_size[1])

    def _draw_point(
        self,
        image: np.ndarray,
        normalized: tuple[float, float],
        label: int,
        color: tuple[int, int, int],
        content_size: tuple[int, int],
    ) -> None:
        pixel = self._display_point(normalized, content_size)
        cv2.circle(image, pixel, 6, color, -1)
        cv2.putText(image, str(label), (pixel[0] + 7, pixel[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenCV homography and area calibration tool")
    parser.add_argument("--camera-image", required=True)
    parser.add_argument("--scene-image", required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--scene-id", default="scene_01")
    parser.add_argument("--area-id", default="pouring_zone_01")
    parser.add_argument("--output-dir", default="config/generated")
    args = parser.parse_args()

    camera_image = cv2.imread(args.camera_image)
    scene_image = cv2.imread(args.scene_image)
    if camera_image is None or scene_image is None:
        print("Failed to load camera or scene image.", file=sys.stderr)
        raise SystemExit(2)
    CalibrationUi(
        camera_image,
        scene_image,
        camera_id=args.camera_id,
        scene_id=args.scene_id,
        area_id=args.area_id,
        output_dir=Path(args.output_dir),
    ).run()


if __name__ == "__main__":
    main()
