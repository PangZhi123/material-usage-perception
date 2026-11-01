# 开发记录（2026-08-24）

## 本轮目标

在没有 GPU 的条件下，从重新标定、旧检测结果空间回放、车辆作业状态机三个方向继续单摄像头离线 MVP；在此基础上补齐在线车辆身份修复、完整五分钟回放验证与物料用量估算。

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

### 3. 在线车辆身份修复

#### 背景

500 帧旧跟踪结果中，同一辆红色车辆出现 7 个原始 ByteTrack ID。系统后续需要接入 RTSP 相机，因此不能使用依赖完整视频的离线轨迹聚类。

#### 在线轨迹拼接器

新增 `identity/online_track_stitcher.py`，只使用当前帧和有限历史缓存，根据以下信息实时判断新 raw ID 是否属于最近丢失的车辆：

- 摄像头、流会话和施工区域；
- 时间间隔；
- 俯视图位置与速度预测位置；
- 摄像头图像位置；
- 车辆类别；
- 同帧是否仍存在原 raw ID。

输出稳定 `vehicle_visit_id`，同时保留 `source_track_id`、是否发生拼接及拼接分数。

#### 同帧重复车辆抑制

旧数据中的 raw ID 30 和 50 与主车辆框同时存在，并且框大部分被主框包含，属于重复检测而非 ID 切换。在线处理器根据 IoU、包含率、已有轨迹连续性、置信度和框面积保留一个车辆观测。

#### 统一实时处理核心

新增 `pipeline/frame_processor.py`：

```text
逐帧空间观测
 -> 同帧重复抑制
 -> 在线 ID 修复
 -> vehicle_visit_id
 -> 作业状态机
 -> 实时事件
```

旧 JSON、本地视频和未来 RTSP 相机均调用同一个 `FrameProcessor.process_frame()`，核心业务逻辑不依赖输入来源，也不读取未来帧。

#### 状态机身份升级

状态机改为按：

```text
camera_id + processing_session_id + vehicle_visit_id + area_id
```

管理访问状态。事件保存全部 `source_track_ids`，便于审计 ID 拼接结果。

### 4. 重新标定准备

现有 OpenCV 双图工具已经能够重新选择 6～10 对地面点、计算 RANSAC 单应矩阵、检查误差、绘制区域并保存正式 YAML。重新标定需要人工点击，代码侧无需 GPU。

建议输出正式配置：

```text
config/generated/cam_02_camera_to_scene.npy
config/generated/cam_02_spatial.yaml
```

### 5. 完整五分钟验证

使用旧 ByteTrack JSON 运行 7500 帧 CPU 回放，不调用 YOLO。

结果：

- 原始车辆观测 515 条；
- 原始 raw ID 7 个；
- 在线 ID 拼接 4 次；
- 同帧重复观测抑制 20 条；
- 最终稳定 `vehicle_visit_id` 1 个；
- 车辆从视频开始就在区域内，确认离开时间为第 500 帧、23 秒；
- 区域停留 23 秒，低于默认 30 秒最短作业阈值；
- 输出 1 个 `pass_through`，无 `completed` 和 `incomplete` 事件。

完整调试视频：`outputs/cam2_online_identity_replay_full.mp4`。

### 6. 物料估算

新增 `estimation/material_usage.py`：

- 方量来源优先级：实际装载量 > 车辆容量 > 默认容量；
- 默认公式：`capacity_m3 × load_factor × unload_factor`；
- 只有 `completed` 事件计量；
- `pass_through` 和 `incomplete` 明确输出 0；
- 累计器按相机和区域聚合；
- 同一 `vehicle_visit_id` 不能重复计量。

新增 `pipeline/material_from_events.py`，可以在不重跑视频的情况下更换容量参数并重新计算已有事件。

当前旧样本的物料结果：

- 有效车辆数 0；
- 忽略事件数 1；
- 估计用量 0 m³。

该结果符合当前 30 秒最短作业规则。是否将 23 秒车辆视为真实作业，需要现场业务依据，不能仅为了产生用量而降低阈值。

## 验证汇总

- 状态机相关新增测试 6 项通过，加无 OpenCV 依赖的既有测试 7 项，当时合计 13 项；
- 加入在线身份与处理管线后累计 29 项测试通过；
- 500 帧真实旧数据：514 条原始观测、7 个原始 raw ID、4 次 ID 切换成功拼接、20 条同帧重复观测被抑制、最终只生成 1 个稳定 `vehicle_visit_id`；
- 完成物料估算与完整回放后合计 33 项单元测试通过；
- 34 个 Python 文件 AST 语法检查通过；
- 完整回放、车辆事件文件和物料累计文件均已生成；
- 当前 Codex 临时 Python 不包含 OpenCV 和 PyYAML，因此未在该运行时生成真实回放视频；应在项目本地 Python 3.10 环境执行。

500 帧验证回放：`outputs/cam2_online_identity_replay_500.mp4`。

## 下一步操作

1. 在本地 Python 3.10 环境运行标定工具，生成正式 cam2 空间配置；
2. 先用 `--max-frames 500` 做短回放，检查画面、矩阵方向和轨迹；
3. 短回放正确后运行完整五分钟视频；
4. 人工核对 `cam2_vehicle_visits.json`，调整进入、停留、离开和丢失阈值；
5. 人工抽查不同时间段，校准在线拼接距离和时间阈值；
6. 实现 RTSP `FrameSource` 和每相机 `CameraWorker`；
7. 增加最新帧队列、断线重连、流会话 ID、现场时间和处理延迟指标；
8. 获得至少包含一次真实卸料过程的视频或人工事件真值，校准最短停留时间及进入、离开确认阈值；
9. 定义工作相机输入配置和结构化结果输出接口；
10. 在有推理设备时接入 YOLO + ByteTrack 的实时相机帧；
11. 状态机稳定后根据业务数据接入车辆实际装载量或车辆容量。
