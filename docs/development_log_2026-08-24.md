# 开发记录（2026-08-24）

## 本轮目标

在没有 GPU 的条件下，从重新标定、旧检测结果空间回放、车辆作业状态机三个方向继续单摄像头离线 MVP。

## 已完成

### 1. 无 GPU 空间回放

新增 `pipeline/legacy_spatial_replay.py`，直接读取旧 `tracking_results_2.json` 中的罐车 `bbox` 和 ByteTrack `track_id`，结合源视频、俯视图和空间配置完成：

- 摄像头画面检测框和底边中心绘制；
- 俯视图映射点、车辆轨迹和计量区域绘制；
- 区域内外状态着色；
- 逐帧空间观测 JSON Lines 输出；
- 调试回放 MP4 输出；
- 车辆访问事件 JSON 输出。

该流程不调用 YOLO，不需要 GPU。旧跟踪脚本的置信度有历史错误，因此仅将旧 `bbox` 和 `track_id` 作为可信输入。

### 2. 车辆作业状态机

新增纯时间状态机：

```text
OUTSIDE -> ENTERING -> INSIDE -> LEAVING -> OUTSIDE
```

支持：

- 连续进入确认；
- 最短停留判定；
- 连续离开确认；
- 短时跟踪丢失容忍；
- 边界回入取消离场；
- `completed` 有效访问；
- `pass_through` 短时经过；
- 视频结束时的 `incomplete` 未闭合访问。

状态键为 `camera_id + processing_session_id + track_id + area_id`，所有阈值均使用视频时间秒数。

### 3. 重新标定准备

现有 OpenCV 双图工具已经能够重新选择 6～10 对地面点、计算 RANSAC 单应矩阵、检查误差、绘制区域并保存正式 YAML。重新标定需要人工点击，代码侧无需 GPU。

建议输出正式配置：

```text
config/generated/cam_02_camera_to_scene.npy
config/generated/cam_02_spatial.yaml
```

## 验证

- 新增状态机测试 6 项通过；
- 无 OpenCV 依赖的既有测试 7 项通过；
- 合计 13 项测试通过；
- 23 个 Python 文件 AST 语法检查通过；
- 当前 Codex 临时 Python 不包含 OpenCV 和 PyYAML，因此未在该运行时生成真实回放视频；应在项目本地 Python 3.10 环境执行。

## 下一步操作

1. 在本地 Python 3.10 环境运行标定工具，生成正式 cam2 空间配置；
2. 先用 `--max-frames 500` 做短回放，检查画面、矩阵方向和轨迹；
3. 短回放正确后运行完整五分钟视频；
4. 人工核对 `cam2_vehicle_visits.json`，调整进入、停留、离开和丢失阈值；
5. 状态机稳定后开发车辆容量和物料用量估算。
