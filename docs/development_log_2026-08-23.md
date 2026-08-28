# 开发交接记录（2026-08-23）

## 今日目标

建立单摄像头离线 MVP 的前半段算法链路，并补齐开发验证所需的坐标标定与关键区域绘制能力。

## 已完成

### 1. 旧 ML 原型梳理

- 确认旧工程包含自训练 YOLO 权重、六路现场视频、ByteTrack 结果、单应矩阵、俯视图和关键区域配置。
- 模型 `best.pt` 类别映射已读取确认，`transporter` 的类别 ID 为 `8`。
- 旧统计方法按区域内出现帧数和吨/小时估算，没有按车辆作业事件计量；新工程不沿用该统计逻辑。
- 发现旧跟踪脚本错误地为同帧所有检测框写入 `boxes.conf[0]`。
- 发现旧单应矩阵文件名方向与实际用法不一致，并存在前后隐藏缩放。

### 2. 标准车辆观测模型

实现 `BoundingBox` 和 `TrackObservation`，统一记录：

- 摄像头和处理会话；
- 视频帧号和视频时间；
- `track_id`；
- 车辆类别；
- 检测框和逐目标置信度；
- 检测框底边中心地面点。

旧跟踪 JSON 读取器仅保留为测试与数据分析辅助，不作为生产主链路输入。

### 3. 正式检测与跟踪链路

实现 YOLO + ByteTrack 视频适配器：

```text
视频 -> best.pt 检测 transporter -> ByteTrack -> TrackObservation
```

- 只请求 `transporter` 类别；
- 每个目标使用自己的置信度；
- 视频时间由帧号和 FPS 产生；
- 空帧也保留在逐帧输出中；
- 离线管线使用 JSON Lines 流式写出。

真实 cam2 首帧已成功检测并跟踪到一辆罐车。ByteTrack 缺失依赖 `lap>=0.5.12` 已加入 `requirements.txt` 并安装。

### 4. 空间映射与区域判断

实现：

```text
检测框底边中心 -> camera_to_scene 单应矩阵 -> 俯视图坐标 -> 多边形区域判断
```

- 单应矩阵必须明确声明为 `camera_to_scene`；
- 显式记录原图、矩阵输入、矩阵输出和场景图尺寸；
- 支持任意多边形，边界点视为区域内；
- cam2 旧矩阵的隐藏缩放已通过示例 YAML 显式表达。

### 5. 标定与区域开发工具

实现后端标定算法：

- 使用 `[0,1]` 归一化点对；
- 至少四对点；
- 拒绝重复和共线点；
- 使用 RANSAC 计算矩阵；
- 输出内点数、逐点误差、平均误差和最大误差；
- 校验矩阵有限且可逆；
- 校验区域范围、面积、重复顶点和自相交。

实现 OpenCV 双图工具：

- 左侧选择摄像头地面特征点；
- 右侧选择对应俯视图点；
- 计算并显示标定质量；
- 在俯视图绘制关键区域；
- 保存 `.npy` 矩阵和视频管线可直接读取的 YAML。

### 6. 不同分辨率收尾

已区分三种尺寸：显示尺寸、标定参考图尺寸、运行视频帧尺寸。

运行时自动读取视频 FPS、宽度和高度。若运行视频与标定图宽高比一致但分辨率不同，车辆点先按运行分辨率归一化，再映射到矩阵标定尺寸。若宽高比不同则拒绝运行，因为裁剪、旋转或黑边无法通过简单缩放可靠修复。

已验证同一个相对点在 `2560x1440` 与 `1920x1080` 下产生完全一致的场景坐标。

## 当前主链路状态

```text
视频
 -> YOLO 罐车检测
 -> ByteTrack 车辆跟踪
 -> 检测框底边中心
 -> 分辨率归一化
 -> camera_to_scene 映射
 -> 关键区域内外判断
 -> 尚未实现：车辆作业状态机
 -> 尚未实现：单车用量估算
 -> 尚未实现：事件与累计统计
```

## 验证结果

- 全部 16 项单元测试通过。
- 源码和工具脚本 `compileall` 通过。
- 真实模型、真实 cam2 视频首帧、ByteTrack 冒烟测试通过。
- 标定工具输出到运行时空间配置的保存/加载回环测试通过。
- 尚未完整运行五分钟视频并人工核对全程映射轨迹。

## 关键文件

- `src/material_usage_perception/domain/models.py`
- `src/material_usage_perception/tracker/yolo_bytetrack.py`
- `src/material_usage_perception/geometry/calibration.py`
- `src/material_usage_perception/geometry/spatial.py`
- `src/material_usage_perception/pipeline/offline_video.py`
- `tools/opencv_spatial_calibration.py`
- `config/cam2_spatial.example.yaml`

## 明天建议从这里继续

### 第一项：实际操作标定工具

使用 cam2 参考图和俯视图重新选择 6～10 对地面对应点，绘制真实关键区域，生成 `config/generated/cam_02_spatial.yaml`。重点检查平均与最大重投影误差，并在俯视图上人工观察若干测试点是否落位准确。

### 第二项：完整视频空间回放

运行 cam2 五分钟视频，输出逐帧空间观测。建议增加一段调试视频可视化：左侧绘制检测框和地面点，右侧绘制映射点、轨迹和关键区域，以人工检查矩阵方向、区域和跟踪稳定性。

### 第三项：车辆作业状态机

空间映射确认后，实现按 `camera_id + processing_session_id + track_id + area_id` 管理的状态机：

```text
OUTSIDE -> ENTERING -> INSIDE -> LEAVING -> COMPLETED
```

覆盖进入确认、最短停留、离开确认、短时漏检、边界抖动、经过车辆和视频结束收尾。

## 已知限制和注意事项

- 单应矩阵只适用于近似同一地面平面；高处点不能正确映射，因此车辆使用检测框底边中心。
- 运行视频如果被裁剪、旋转、加黑边或改变视角，需要重新标定。
- 旧 cam2 矩阵只适合作为兼容验证，建议用新工具重新标定。
- 目标检测和单应映射只能识别车辆进入并停留，不能直接测量罐内真实方量；方量仍需运输单、车辆容量或默认容量。
- 当前目录未被 Git 识别为仓库，今日修改没有提交记录。

## 常用命令

运行测试：

```powershell
$env:PYTHONPATH="src"
python -m unittest discover -s tests -p "test_*.py" -v
```

启动 OpenCV 标定工具：

```powershell
$env:PYTHONPATH="src"
python tools\opencv_spatial_calibration.py `
  --camera-image "config\generated\cam2_reference_frame.png" `
  --scene-image "data\test\scene-1.png" `
  --camera-id cam_02 `
  --scene-id scene_01 `
  --area-id pouring_zone_01 `
  --output-dir config\generated
```

运行离线检测、跟踪与空间判断：

```powershell
$env:PYTHONPATH="src"
python -m material_usage_perception.pipeline.offline_video `
  --video "data\test\cam2_5min_clip.mp4" `
  --model "models\best.pt" `
  --output "outputs\cam_02_tracking.jsonl" `
  --camera-id cam_02 `
  --spatial-config "config\generated\cam_02_spatial.yaml" `
  --confidence 0.5
```
