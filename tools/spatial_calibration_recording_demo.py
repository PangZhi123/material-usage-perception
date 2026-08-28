"""PyCharm-ready demo for feature-point calibration and work-area selection.

Run this file directly. It uses the first frame of the repository's test video
as the camera reference image, opens the existing calibration UI, and records
the UI window to ``outputs/spatial_calibration_demo.mp4``.

Controls:
    Left click: select points
    P: switch between feature-point calibration and work-area selection
    H: compute the homography
    U: undo
    R: reset the points in the current mode
    S: save the calibration and work area
    Q or Esc: finish and close the recording
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Make direct execution in PyCharm independent of its working-directory setting.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from opencv_spatial_calibration import CalibrationUi  # noqa: E402


TEST_VIDEO = PROJECT_ROOT / "data" / "test" / "cam2_5min_clip.mp4"
SCENE_IMAGE = PROJECT_ROOT / "data" / "test" / "scene-1.png"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "spatial_calibration_demo"
RECORDING_PATH = PROJECT_ROOT / "outputs" / "spatial_calibration_demo.mp4"
RECORDING_FPS = 10.0
RECORDING_MAX_WIDTH = 1280
WINDOW_WIDTH = 1500
WINDOW_HEIGHT = 650
CHINESE_FONT = Path("C:/Windows/Fonts/msyh.ttc")
INSTRUCTION_PANEL_HEIGHT = 170


def read_first_video_frame(video_path: Path):
    capture = cv2.VideoCapture(str(video_path))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"无法读取测试视频首帧：{video_path}")
    return frame


class RecordingCalibrationUi(CalibrationUi):
    """Existing calibration UI with its displayed frames encoded to MP4."""

    def __init__(self, *args, recording_path: Path, recording_fps: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.recording_path = recording_path
        self.recording_fps = recording_fps
        self.feedback = "请从步骤 1 开始：在左侧摄像头图像上点击一个特征点。"
        self.chinese_font = ImageFont.truetype(str(CHINESE_FONT), 21)
        self.chinese_font_small = ImageFont.truetype(str(CHINESE_FONT), 18)

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        pair_count = len(self.pairs)
        polygon_count = len(self.polygon)
        pending_camera = self.pending_camera
        super().on_mouse(event, x, y, flags, param)

        if self.mode == "calibration":
            if pending_camera is None and self.pending_camera is not None:
                self.feedback = "✓ 已选择左图特征点，请在右图点击它的对应位置。"
            elif len(self.pairs) > pair_count:
                if len(self.pairs) < 4:
                    self.feedback = f"✓ 第 {len(self.pairs)} 对特征点已完成，还需至少 {4 - len(self.pairs)} 对。"
                else:
                    self.feedback = f"✓ 已完成 {len(self.pairs)} 对特征点。可按 H 计算标定，或继续添加点。"
            elif x >= self.camera_display_width and self.pending_camera is None:
                self.feedback = "⚠ 请先点击左图特征点，再点击右图对应位置。"
        elif len(self.polygon) > polygon_count:
            if len(self.polygon) < 3:
                self.feedback = f"✓ 已选择 {len(self.polygon)} 个区域顶点，还需至少 {3 - len(self.polygon)} 个。"
            else:
                self.feedback = f"✓ 工作区域已有 {len(self.polygon)} 个顶点。完成后按 S 保存。"

    def compute(self) -> None:
        super().compute()
        if self.result is None:
            self.feedback = "⚠ 标定计算失败，请检查是否有至少四对、不共线的特征点。"
        else:
            quality = self.result.quality
            self.feedback = (
                f"✓ 标定完成：内点 {quality.inlier_count}/{quality.point_count}，"
                f"平均误差 {quality.mean_reprojection_error_px:.2f}px。按 P 选择工作区域。"
            )

    def undo(self) -> None:
        super().undo()
        self.feedback = "↶ 已撤销上一步操作。"

    def reset_current_mode(self) -> None:
        mode_name = "工作区域" if self.mode == "polygon" else "特征点"
        super().reset_current_mode()
        self.feedback = f"↻ 已清空当前的{mode_name}，请重新选择。"

    def save(self) -> None:
        config_path = self.output_dir / f"{self.camera_id}_spatial.yaml"
        previous_mtime = config_path.stat().st_mtime_ns if config_path.exists() else None
        super().save()
        current_mtime = config_path.stat().st_mtime_ns if config_path.exists() else None
        if current_mtime is not None and current_mtime != previous_mtime:
            self.feedback = f"✓ 保存成功：标定矩阵和工作区域已写入 {self.output_dir}"
        else:
            self.feedback = "⚠ 尚未保存：请先完成标定，并选择至少三个有效的区域顶点。"

    def render(self) -> np.ndarray:
        frame = super().render()
        panel = np.full(
            (INSTRUCTION_PANEL_HEIGHT, frame.shape[1], 3), (245, 245, 245), dtype=np.uint8
        )
        panel_rgb = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(panel_rgb)
        draw = ImageDraw.Draw(image)

        mode_name = "特征点标定" if self.mode == "calibration" else "工作区域选取"
        summary = (
            f"当前模式：{mode_name}    特征点对：{len(self.pairs)}    "
            f"区域顶点：{len(self.polygon)}"
        )
        if self.mode == "calibration":
            steps = "步骤 1：点击左图特征点  →  步骤 2：点击右图对应点  →  重复至少 4 对  →  按 H 计算  →  按 P 切换"
        else:
            steps = "步骤 3：只在右图依次点击工作区域边界（至少 3 点）  →  按 S 保存  →  按 Q 结束录像"

        draw.text((15, 10), summary, font=self.chinese_font, fill=(20, 20, 20))
        draw.text((15, 50), steps, font=self.chinese_font_small, fill=(35, 35, 35))
        draw.rounded_rectangle(
            (12, 91, frame.shape[1] - 12, 153), radius=8, fill=(225, 240, 255)
        )
        draw.text((25, 106), f"操作反馈：{self.feedback}", font=self.chinese_font, fill=(0, 75, 145))
        panel = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        return np.vstack((frame, panel))

    def run(self) -> None:
        window = "Spatial calibration demo (camera | scene)"
        self.recording_path.parent.mkdir(parents=True, exist_ok=True)

        first_frame = self.render()
        display_height, display_width = first_frame.shape[:2]
        recording_scale = min(1.0, RECORDING_MAX_WIDTH / display_width)
        recording_size = (
            int(display_width * recording_scale),
            int(display_height * recording_scale),
        )
        writer = cv2.VideoWriter(
            str(self.recording_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.recording_fps,
            recording_size,
        )
        if not writer.isOpened():
            raise RuntimeError(f"无法创建录像文件：{self.recording_path}")

        # On Windows, a window created by a PyCharm child process can initially
        # appear behind the IDE. Show one frame first, then size, position and
        # briefly raise it so the demo is immediately visible.
        cv2.startWindowThread()
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.imshow(window, first_frame)
        cv2.resizeWindow(window, WINDOW_WIDTH, WINDOW_HEIGHT)
        cv2.moveWindow(window, 40, 40)
        cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
        cv2.setMouseCallback(window, self.on_mouse)
        cv2.waitKey(100)
        print("演示已启动，操作过程会自动录制。")
        print("先在左图点一个特征点，再在右图点对应位置；至少选择四对。")
        print("按 P 切换到工作区域模式，在右图依次点击多边形顶点。")
        print("如果窗口仍被遮挡，请按 Alt+Tab，选择 Spatial calibration demo。")

        frame_period = 1.0 / self.recording_fps
        next_record_time = time.monotonic()
        try:
            while True:
                frame = self.render()
                cv2.imshow(window, frame)

                now = time.monotonic()
                # Keep the saved video close to real time, including pauses while
                # the presenter explains the current screen.
                while now >= next_record_time:
                    recorded_frame = cv2.resize(
                        frame, recording_size, interpolation=cv2.INTER_AREA
                    )
                    writer.write(recorded_frame)
                    next_record_time += frame_period

                key = cv2.waitKey(10) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("p"):
                    self.mode = (
                        "polygon" if self.mode == "calibration" else "calibration"
                    )
                    if self.mode == "polygon":
                        self.feedback = "✓ 已切换到工作区域模式，请只在右图点击区域边界顶点。"
                    else:
                        self.feedback = "✓ 已切换到特征点模式，请先点击左图，再点击右图对应点。"
                    print(f"当前模式：{self.mode}")
                elif key == ord("h"):
                    self.compute()
                elif key == ord("u"):
                    self.undo()
                elif key == ord("r"):
                    self.reset_current_mode()
                elif key == ord("s"):
                    self.save()
        finally:
            writer.release()
            cv2.destroyAllWindows()
            print(f"录像已保存：{self.recording_path}")


def main() -> None:
    if not TEST_VIDEO.is_file():
        raise FileNotFoundError(f"缺少测试视频：{TEST_VIDEO}")
    if not SCENE_IMAGE.is_file():
        raise FileNotFoundError(f"缺少场景图：{SCENE_IMAGE}")

    camera_image = read_first_video_frame(TEST_VIDEO)
    scene_image = cv2.imread(str(SCENE_IMAGE))
    if scene_image is None:
        raise RuntimeError(f"无法读取场景图：{SCENE_IMAGE}")

    RecordingCalibrationUi(
        camera_image,
        scene_image,
        camera_id="cam_02_demo",
        scene_id="scene_01",
        area_id="pouring_zone_demo",
        output_dir=OUTPUT_DIR,
        recording_path=RECORDING_PATH,
        recording_fps=RECORDING_FPS,
    ).run()


if __name__ == "__main__":
    main()
