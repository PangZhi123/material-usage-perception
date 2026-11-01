# 物料使用情况智能感知后端

本工程用于从施工现场视频中检测、跟踪混凝土罐车，判断车辆是否进入关键施工区域，并将一次完整的车辆作业过程转换为混凝土使用事件与累计用量。

当前阶段聚焦后端算法。关键点标注、关键区域划定等交互由前端完成，后端接收配置并执行坐标转换、区域判断、事件识别和物料估算。

## 2026-09-01 更新说明

本次更新完善了本地历史视频接入及注册流程：

- 新增 `POST /api/material-perception/v1/videos`，供页面端以
  `multipart/form-data` 上传 MP4；上传完成并通过解码校验后返回 `videoId`。
- 新增 `POST /api/material-perception/v1/video-registrations`，用于登记算法服务器
  允许目录内已有的 MP4，直接读取原文件，不产生额外视频副本。
- 上传过程采用分块写入并同步计算 SHA-256；相同视频重复上传会复用已有文件，
  相同 `cameraId` 下还会复用原 `videoId`。
- 视频元数据与注册关系持久化到 SQLite，并同步加入运行时注册表。注册成功后无需
  修改 YAML 或重启服务，即可将返回的 `videoId` 用于 `VIDEO_FILE` 分析任务；服务
  重启后原 `videoId` 仍然有效。
- 创建历史视频任务时，任务的 `cameraId` 必须与视频注册时的 `cameraId` 一致。
- 当前采用可靠的“上传完成后分析”流程，尚未对未上传完整的普通 MP4 边传边分析。

### 接口一：上传并注册页面端历史视频

适用于视频位于测试人员电脑或需要由前端页面选择文件的场景。请求使用
`multipart/form-data`，当前仅支持 MP4：

```powershell
curl.exe -X POST "http://192.168.10.32:8001/api/material-perception/v1/videos" `
  -F "file=@D:/test/history.mp4;type=video/mp4" `
  -F "cameraId=CAM_TEST_1" `
  -F "capturedAt=2026-09-01T08:00:00+08:00"
```

字段说明：

- `file`：必填，待上传的 MP4 文件；
- `cameraId`：必填，视频所属摄像头标识，后续创建任务时必须保持一致；
- `capturedAt`：选填，带时区的 ISO 8601 采集时间。

成功时 HTTP 状态码为 `201`，重点读取响应中的 `data.videoId`：

```json
{
  "success": true,
  "code": "0",
  "data": {
    "videoId": "video_0123456789abcdef",
    "cameraId": "CAM_TEST_1",
    "status": "READY",
    "deduplicated": false
  }
}
```

用相同 `cameraId` 重复上传内容完全相同的文件时，接口返回原 `videoId`，并将
`deduplicated` 设为 `true`，不会再次保存一份视频。

### 接口二：注册算法服务器本地已有视频

适用于 MP4 已经位于算法后端服务器的场景。此接口只登记文件路径，不上传、不复制。
启动后端前先设置服务器允许访问的视频根目录：

```powershell
$env:MATERIAL_PERCEPTION_LOCAL_VIDEO_ROOT="D:\shared-videos"
```

假设待测文件为 `D:\shared-videos\project-a\history.mp4`，请求中的
`relativePath` 应填写相对于上述根目录的路径：

```powershell
curl.exe -X POST `
  "http://192.168.10.32:8001/api/material-perception/v1/video-registrations" `
  -H "Content-Type: application/json" `
  -d '{"cameraId":"CAM_TEST_1","relativePath":"project-a/history.mp4","capturedAt":"2026-09-01T08:00:00+08:00"}'
```

成功时同样返回 HTTP `201` 和 `data.videoId`。`relativePath` 不能是绝对路径，
也不能通过 `..` 越过允许根目录；后端进程必须具有原文件读取权限。由于分析时直接
读取原文件，注册后不要移动或删除该 MP4。

### 使用返回的 videoId 创建历史分析任务

上述任一接口成功后，无需修改 YAML 或重启后端，直接调用任务接口：

```powershell
curl.exe -X POST "http://192.168.10.32:8001/api/material-perception/v1/tasks" `
  -H "Content-Type: application/json" `
  -d '{"configurationId":"cfg_example","inputType":"VIDEO_FILE","videoId":"video_0123456789abcdef","options":{"persistTrajectory":true,"pushIntervalMs":1000}}'
```

其中 `configurationId` 必须替换为针对该 `videoId` 重新上传场景图、标定并划定
工作区域后生成的专属配置 ID，不能复用相机或其他视频的旧配置；`videoId` 必须
原样使用接口返回值，不能填写文件名。也可以打开
`http://<后端IP>:<端口>/docs`，依次通过 Swagger 测试“上传/注册”和“创建任务”。

测试人员可参考
[历史视频上传注册接口测试说明](docs/历史视频上传注册接口测试说明.txt)
完成上传、去重、重启持久化及历史分析任务验收。

## 核心原则

- 统计对象是“一辆车的一次有效作业事件”，不是目标出现的帧数。
- 使用目标跟踪 ID 区分车辆，避免逐帧重复计量。
- 使用检测框底边中心作为车辆地面位置。
- 关键施工区域支持多边形，坐标优先使用归一化形式。
- 通过进入、停留、离开状态机抑制漏检和边界抖动。
- 混凝土量优先采用运输单实际装载量，其次采用车辆档案容量，最后使用默认容量估算。
- MVP 阶段一个施工区域只设置一个计量摄像头，避免多摄像头重复计量。

## 计划中的处理链路

```text
前端标定/区域配置 -> 视频输入 -> YOLO 检测 -> ByteTrack 跟踪
-> 地面点提取 -> 坐标映射 -> 多边形区域判断 -> 在线ID修复
-> 稳定vehicle_visit_id -> 车辆状态机
-> 有效卸料事件 -> 单车用量估计 -> 事件明细与累计统计
```

## 工程结构

```text
material-usage-perception/
├── models/best.pt                  # 随交付包提供的 YOLO 模型
├── data/test/                      # 联调视频、跟踪结果、场景图和标定矩阵
├── config/                         # 本地开发配置与示例配置
├── docs/                           # 设计与接口文档
├── src/material_usage_perception/
│   ├── api/                        # FastAPI 接口
│   ├── detector/                   # 罐车检测
│   ├── tracker/                    # 多目标跟踪
│   ├── geometry/                   # 坐标转换与区域判断
│   ├── domain/                     # 领域模型与车辆状态机
│   ├── estimation/                 # 物料用量估计
│   ├── pipeline/                   # 离线/实时处理管线
│   └── repositories/               # 配置与事件持久化
├── tests/                          # 单元测试与集成测试
├── requirements.txt
└── pyproject.toml
```

详细方案见 [docs/material_usage_algorithm_design.md](docs/material_usage_algorithm_design.md)。  
开发日志按日期归档在 [docs/开发日志/](docs/开发日志/README.md)。

## 开发环境约定

工程不内置 `.venv`，开发时直接使用本机已有 Python 环境。迁移到其他设备后再安装依赖：

```bash
pip install -r requirements.txt
```

本交付包在 `models/` 和 `data/test/` 中附带模型及联调数据。它们默认仍被
`.gitignore` 排除，避免误提交大文件；复制或压缩整个工程目录时需要确认这些目录
已被包含。检测结果和运行产物写入 `outputs/`，不应作为源代码提交。

## 第一阶段开发目标

先实现单摄像头离线闭环：输入视频、前端区域配置、模型权重和车辆容量配置，输出逐车作业事件及累计混凝土方量。离线验证稳定后，再接入 FastAPI 和 RTSP 实时流。

## 当前可运行能力

已实现离线视频的 YOLO 罐车检测与 ByteTrack 跟踪，输出逐帧 JSON Lines：

```bash
python -m material_usage_perception.pipeline.offline_video \
  --video path/to/video.mp4 \
  --model path/to/best.pt \
  --output outputs/cam_02_tracking.jsonl \
  --camera-id cam_02 \
  --spatial-config config/cam2_spatial.example.yaml \
  --output-events outputs/cam_02_events.json \
  --output-material outputs/cam_02_material.json \
  --minimum-dwell-seconds 60 \
  --confidence 0.5
```

该入口现在执行完整离线闭环，而不只输出检测框：空间映射后的观测会继续经过
在线 ID 修复、逐车作业状态机和物料估算。`--minimum-dwell-seconds` 是后台可设置
的最短工作时长。同一区域同一时刻存在多辆罐车时，每个 ByteTrack 轨迹分别维护
`vehicle_visit_id` 和状态，不会因为共享施工区域而合并。

在未安装工程包时，先将 `src` 加入 `PYTHONPATH`，或以可编辑模式安装工程。

### OpenCV 标定与区域工具

开发验证阶段可以使用双图 OpenCV 工具标注摄像头—俯视图对应点，并在俯视图绘制计量区域：

```bash
python tools/opencv_spatial_calibration.py \
  --camera-image path/to/camera_reference.png \
  --scene-image path/to/scene_topview.png \
  --camera-id cam_02 \
  --scene-id scene_01 \
  --area-id pouring_zone_01 \
  --output-dir config/generated
```

标定模式下先点击左侧摄像头点，再点击右侧对应场景点，至少完成四对。快捷键：`P` 切换标定/区域模式，`H` 计算矩阵，`U` 撤销，`R` 重置当前模式，`S` 保存，`Q` 退出。保存的 YAML 可直接传给离线管线的 `--spatial-config`。

### 无 GPU 的旧跟踪结果空间回放

旧 ByteTrack JSON 已包含车辆框和 `track_id`，空间回放无需再次运行 YOLO：

```powershell
$env:PYTHONPATH="src"
python -m material_usage_perception.pipeline.legacy_spatial_replay `
  --video "data\test\cam2_5min_clip.mp4" `
  --tracking-json "data\test\tracking_results_2.json" `
  --scene-image "data\test\scene-1.png" `
  --spatial-config "config\cam2_spatial.example.yaml" `
  --output-video "outputs\cam2_legacy_spatial_replay.mp4" `
  --output-jsonl "outputs\cam2_legacy_spatial.jsonl" `
  --output-events "outputs\cam2_vehicle_visits.json" `
  --output-material "outputs\cam2_material_usage.json" `
  --camera-id cam_02
```

左侧显示旧检测框、车辆底边中心和区域状态，右侧显示俯视图映射点、轨迹及计量区域。回放同时通过车辆状态机输出 `completed`、`pass_through` 和 `incomplete` 访问事件。旧脚本的置信度存在历史写入问题，回放只将 `bbox` 和 `track_id` 作为可信输入。

离线视频、旧 JSON 和未来 RTSP 相机共用 `pipeline.FrameProcessor`。该处理器逐帧执行同帧重复框抑制、在线 ID 切换修复和作业状态机，不使用未来帧或全视频聚类，能够直接用于实时流。

### 物料用量输出规则

车辆事件按以下优先级解析方量：实际装载量、车辆档案容量、默认容量。默认计算为 `容量 × 装载系数 × 卸料系数`。只有 `completed` 作业事件进入累计，`pass_through` 和 `incomplete` 输出审计记录但用量为 0。

已有事件文件可以单独重新估算，不需要重跑视频：

```powershell
$env:PYTHONPATH="src"
python -m material_usage_perception.pipeline.material_from_events `
  --input-events "outputs\cam2_online_identity_events_full.json" `
  --output-material "outputs\cam2_material_usage_full.json" `
  --default-capacity-m3 12 `
  --default-load-factor 1 `
  --default-unload-factor 0.95
```

### 人工设置与单视频后台测试

启动最小测试后台：

```powershell
$env:PYTHONPATH="src"
$env:MATERIAL_PERCEPTION_MODEL_PATH="models\best.pt"
$env:MATERIAL_PERCEPTION_MODEL_VERSION="yolov8-ml-1-20250511"
$env:MATERIAL_PERCEPTION_CONFIDENCE="0.5"
$env:MATERIAL_PERCEPTION_CORS_ORIGINS="http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173"
uvicorn material_usage_perception.api.app:app --host 127.0.0.1 --port 8000
```

浏览器打开 `http://127.0.0.1:8000/docs`，可人工调用：

- `GET /settings`：查看当前算法参数；
- `PUT /settings`：设置最短工作时长、进出确认、漏检容忍、ID 拼接及容量参数；
- `POST /analysis/offline`：使用现有视频、跟踪 JSON 和空间配置生成事件及用量；
- `GET /health`：后台健康检查。

正式 V1 后端接口已经开始按 2026-08-25 接口规范落地：

- `POST /api/material-perception/v1/scenes`：上传 PNG/JPEG 场景图，返回尺寸、摘要和 `sceneId`；
- `POST /api/material-perception/v1/configurations`：提交特征点、工作区域、物料和状态机参数，返回不可变配置版本、单应矩阵及标定质量；
- `POST /api/material-perception/v1/cameras`：登记或更新 `cameraId` 对应的 RTSP/RTMP 地址；
- `POST /api/material-perception/v1/videos`：流式上传 MP4、哈希去重、持久注册并返回 `videoId`；
- `POST /api/material-perception/v1/video-registrations`：登记允许目录内已有的本地 MP4，不复制文件；
- `POST /api/material-perception/v1/tasks`：创建 `VIDEO_FILE` 历史视频任务或 `CAMERA_STREAM` 实时任务；
- `PATCH /api/material-perception/v1/tasks/{taskId}`：幂等停止任务；
- `GET /api/material-perception/v1/events`：按时间、相机、区域和状态分页查询事件；
- `GET /api/material-perception/v1/statistics`：查询完成、路过、未完成事件和正式累计方量；
- `GET /api/material-perception/v1/trajectories/{vehicleVisitId}`：查询可抽样的相机/场景轨迹；
- 业务错误使用规范中的 `MP0101`～`MP0105`，HTTP 响应采用统一包络。

配置请求允许传入 `cameraFrameWidth` 和 `cameraFrameHeight`；省略时使用接口示例的
1920×1080。实时任务开始后，检测框归一化和空间投影使用相机实际返回的帧尺寸。
接口规范规定的最短有效作业时间默认值为 60 秒，后端默认值已与其统一，用户配置
仍可覆盖。

正式任务的模型路径属于部署配置，不写入前端配置或接口响应。启动前设置：

```powershell
$env:MATERIAL_PERCEPTION_MODEL_PATH="models\best.pt"
$env:MATERIAL_PERCEPTION_MODEL_VERSION="yolov8-ml-1-20250511"
$env:MATERIAL_PERCEPTION_CONFIDENCE="0.5"
$env:MATERIAL_PERCEPTION_CAMERA_REGISTRY="config\camera_registry.yaml"
```

参考文件为 `config/backend.env.example`。`GET /health` 会返回模型是否已配置及
`model_version`，但不会暴露模型绝对路径。

当前后端默认将场景、配置、视频元数据、任务、事件和轨迹保存为工程目录下的 JSON 文件，
不依赖 SQLite。默认根目录为 `outputs/backend_data/`，可通过 `create_app(storage_root=...)`
指定其他目录。文件采用临时文件写入后替换，降低写入中断造成 JSON 损坏的风险。

文件布局如下：

```text
outputs/backend_data/
├── configurations/<configurationId>.json
├── scenes/<sceneId>.json
├── videos/<videoId>.json
└── tasks/
    ├── index.json
    └── <taskId>/
        ├── task.json
        ├── events.json
        └── trajectories/<vehicleVisitId>.json
```

前端可通过 `GET /files/data/tasks/index.json` 获取任务索引，再读取任务目录中的
`task.json`、`events.json` 和轨迹文件。配置、场景及视频元数据分别位于对应目录，
通过 `GET /files/data/configurations/<configurationId>.json` 等路径读取。场景图片仍通过
`/files/scenes/` 访问。历史视频和实时摄像头任务都由后台线程执行，
`PATCH /tasks/{taskId}` 可请求停止并释放视频源。

实时相机可以通过环境变量指向 YAML 注册表，也可以在服务启动后调用：

```http
POST /api/material-perception/v1/cameras
Content-Type: application/json

{
  "cameraId": "CAM_TEST_1",
  "streamUrl": "rtsp://username:password@192.168.10.100:554/Streaming/Channels/102"
}
```

随后使用相同的 `cameraId` 和对应配置创建任务：

```json
{
  "cameraId": "CAM_TEST_1",
  "configurationId": "cfg_xxx",
  "inputType": "CAMERA_STREAM",
  "options": {"persistTrajectory": true, "pushIntervalMs": 1000}
}
```

RTSP 默认通过 OpenCV FFmpeg 后端并优先使用 TCP；读取线程只向算法提供最新帧，
断线后自动重连，避免推理速度低于码流帧率时不断积压旧帧。

### 历史视频上传与本地注册

前端上传视频：

```http
POST /api/material-perception/v1/videos
Content-Type: multipart/form-data

file=@history.mp4
cameraId=CAM_TEST_1
capturedAt=2026-09-01T08:00:00+08:00
```

后端分块写入文件并同步计算 SHA-256，不会把整个视频加载到内存。相同内容、相同
`cameraId` 重复上传时返回原有 `videoId`；相同内容用于另一个相机时新建注册记录，
但复用同一个物理文件。上传记录保存在 SQLite 中，重启后仍可解析 `videoId`。

若视频已经位于服务器允许目录中，可只注册、不复制。首先设置：

```powershell
$env:MATERIAL_PERCEPTION_LOCAL_VIDEO_ROOT="D:\shared-videos"
```

然后提交相对于该目录的路径：

```json
POST /api/material-perception/v1/video-registrations

{
  "cameraId": "CAM_TEST_1",
  "relativePath": "project-a/history.mp4",
  "capturedAt": "2026-09-01T08:00:00+08:00"
}
```

接口禁止绝对路径和越过允许根目录的路径。上传及本地注册成功后，必须先用返回的
`videoId` 上传专属场景图、重新标定和划定工作区域，再使用新生成的
`configurationId` 调用 `/api/material-perception/v1/tasks`。当前版本在文件上传完成且通过
解码校验后才允许开始分析，尚不对未完成的 MP4 边上传边分析。

### 当前阶段范围限制

当前版本明确以“单工作区域、单监控视频”为目标：

- 每个配置必须且只能提交一个 `workArea`，否则返回 `MP0104`；
- 任务输入支持 `VIDEO_FILE` 和 `CAMERA_STREAM`；
- 同一摄像头同时只允许一个 `QUEUED/RUNNING/STOPPING` 任务；
- 同一摄像头的第二个并发任务返回 `MP0203`；
- 暂不处理多区域身份共享、多视频并行和跨摄像头去重。

这些限制位于接口和仓储层，后续扩展时可以逐步放开，不影响现有单车状态机和用量
估算算法。

例如只把最短工作时长改为 23 秒：

```json
{
  "state": {"minimum_dwell_seconds": 23.0},
  "stitch": {},
  "material": {}
}
```

也可不启动 HTTP 服务，直接对现有跟踪结果进行无界面验收：

```powershell
$env:PYTHONPATH="src"
python -m material_usage_perception.pipeline.offline_analysis `
  --video "data\test\cam2_5min_clip.mp4" `
  --tracking-json "data\test\tracking_results_2.json" `
  --spatial-config "config\cam2_spatial.example.yaml" `
  --camera-id cam_02 `
  --minimum-dwell-seconds 23 `
  --output "outputs\cam2_manual_acceptance.json"
```

不使用 ReID 时，多车区分依赖 ByteTrack ID、同帧身份互斥、位置和运动连续性。
车辆长时间完全遮挡后交叉换位属于该方案的已知边界，结果中保留全部原始
`source_track_ids`，便于后台人工复核。

## 移交与前端联调说明

建议从工程根目录启动后端，所有示例路径均以工程根目录为基准。接收方先安装依赖，
设置上述环境变量，再启动 Uvicorn。`GET /health` 返回 `model_status=ready` 表示模型
路径有效。Swagger 地址为 `http://127.0.0.1:8000/docs`。

浏览器跨域访问由 `MATERIAL_PERCEPTION_CORS_ORIGINS` 控制，多个来源用英文逗号分隔。
默认允许本机前端常用的 3000 和 5173 端口；部署到其他主机或端口时必须显式加入
实际前端 Origin。不要在携带凭据的部署中把允许来源配置为 `*`。

交付包附带以下联调数据：

- `models/best.pt`：YOLO 模型权重；
- `data/test/cam2_5min_clip.mp4`：5 分钟测试视频；
- `data/test/tracking_results_2.json`：已有跟踪结果，可用于无 GPU 快速联调；
- `data/test/scene-1.png`：场景俯视图；
- `data/test/homography_sence-1_to_cam2_clip.npy`：示例空间映射矩阵。

当前整合边界如下：

- 仅支持 `VIDEO_FILE`、单工作区域及单个活动任务，暂不支持实时相机流；
- `POST /tasks` 当前同步处理视频。5 分钟视频可能长时间占用 HTTP 请求，优先使用
  随包提供的 `trackingJsonPath` 做接口联调；
- `videoPath` 和 `trackingJsonPath` 是后端进程所在机器可读取的路径，不是浏览器本地路径；
- WebSocket 实时推送、断线续传、`SNAPSHOT` 和异步 worker 尚未实现；
- `vehicleOverrides`、统计查询的 `materialType`、轨迹查询的 `coordinateType` 以及
  `pushIntervalMs` 已进入接口模型，但当前版本尚未完整应用；
- 本地 JSON 结果保存在 `outputs/backend_data/`。该目录包含任务与现场配置数据，移交前按需筛选；
- 模型、视频和现场数据体积较大且可能涉及内部资料，传输前应确认接收方及传输渠道权限。

快速验证命令：

```powershell
$env:PYTHONPATH="src"
python -m unittest discover -s tests -p "test_*.py" -v
```
