# 物料使用情况智能感知后端开发进度

更新时间：2026-08-28  
接口基线：`物料使用情况智能感知前后端接口规范-0825.docx` V1.0

## 1. 当前结论

当前后端已完成场景上传、空间标定、历史视频异步分析、WebSocket 实时结果推送、
车辆作业事件识别、物料估算、事件查询、统计查询和轨迹查询，可开始与按照同一份
接口规范开发的前端进行历史视频模式联调。

尚未完成的主要功能是 `CAMERA_STREAM` 实时相机输入适配和 MySQL 正式仓储实现。
鉴权按当前安排暂不开发。上述未完成项不阻塞使用随包历史视频进行前端功能联调，
但会阻塞正式实时相机验收和生产部署。

## 2. 已完成的算法闭环

当前处理链路如下：

```text
场景图与标定配置
→ 视频源解析
→ YOLO 检测或已有 tracking JSON 回放
→ ByteTrack 轨迹
→ 地面锚点提取
→ 单应坐标转换
→ 工作区域判断
→ 在线 ID 修复
→ 车辆作业状态机
→ 用量估算
→ 事件与轨迹保存
→ HTTP 查询及 WebSocket 推送
```

已实现的算法能力包括：

- 使用检测框底边中心作为车辆地面锚点；
- 摄像头坐标到场景坐标的单应矩阵转换；
- 归一化及像素多边形区域校验；
- ByteTrack 原始 ID 到稳定 `vehicleVisitId` 的在线拼接；
- 同帧重复车辆框抑制；
- `OUTSIDE / ENTERING / INSIDE / LEAVING` 区域状态；
- `ACTIVE / COMPLETED / PASS_THROUGH / INCOMPLETE / REJECTED` 事件状态；
- 仅 `COMPLETED` 事件进入正式物料累计；
- 车辆容量覆盖、用户默认容量和系统默认容量优先级；
- 事件、轨迹、配置版本和模型版本关联。

## 3. HTTP 接口完成情况

| 接口 | 状态 | 说明 |
|---|---|---|
| `POST /api/material-perception/v1/scenes` | 已完成 | 上传 PNG/JPEG 场景图，返回尺寸、摘要及 `sceneId` |
| `POST /api/material-perception/v1/configurations` | 已完成 | 校验特征点和区域，计算单应矩阵及重投影质量 |
| `POST /api/material-perception/v1/tasks` | 历史视频已完成 | 异步创建任务并立即返回 `taskId` |
| `PATCH /api/material-perception/v1/tasks/{taskId}` | 已完成 | 可取消运行任务，重复停止幂等 |
| `GET /api/material-perception/v1/events` | 已完成 | 时间、相机、区域、状态过滤及分页 |
| `GET /api/material-perception/v1/statistics` | 已完成 | 用量累计、状态计数、材料过滤及时间分组 |
| `GET /api/material-perception/v1/trajectories/{vehicleVisitId}` | 已完成 | 抽样、增量分页和坐标类型控制 |
| `GET /health` | 已完成 | 检查服务和模型配置状态 |

正式任务请求只接收 `cameraId` 或 `videoId`，不再向前端暴露服务器本地
`videoPath` 和 `trackingJsonPath`。历史视频由 `config/video_registry.yaml` 将
`videoId` 映射到部署环境可读取的资源。

## 4. 异步任务能力

任务生命周期已经从同步 HTTP 执行改为后台异步执行，状态包括：

```text
QUEUED → RUNNING → COMPLETED
                 → FAILED
                 → STOPPING → STOPPED
```

当前具备：

- 创建任务后立即返回；
- 后台独立执行视频分析；
- 每帧检查取消令牌；
- 停止时结束帧读取并将活动访问处理为 `INCOMPLETE`；
- 同一相机活动任务互斥；
- 不同相机任务可独立认领；
- 任务异常保存为 `FAILED` 并发布错误消息；
- 任务事件总线和有界历史缓冲。

## 5. WebSocket 完成情况

已实现：

```text
/ws/material-perception/v1/tasks/{taskId}
```

支持以下消息：

- `FRAME_RESULT`：帧信息、检测框、锚点、场景坐标、轨迹增量和车辆状态；
- `EVENT_UPDATE`：活动事件及最终作业事件；
- `HEARTBEAT`：空闲连接心跳；
- `ERROR`：任务或数据保存错误；
- `SNAPSHOT`：历史缓冲不足时恢复最新状态。

实时结果已包含：

- `frame.seq / capturedAt / width / height`；
- `trackId / vehicleVisitId / className`；
- `bboxPx / bboxNorm`；
- `anchorPx`；
- `scenePointPx / scenePointNorm`；
- `trajectoryDelta`；
- `areaId / zoneState / eventStatus`；
- `confidence / estimatedVolumeM3`。

`pushIntervalMs` 已按照视频源时间实际节流。被节流帧中的轨迹点会合并到下一次
推送，不会直接丢弃。客户端可携带 `lastFrameSeq` 和 `lastTrajectorySeq` 请求增量
重放；缓冲不足时服务端发送 `SNAPSHOT`。

## 6. 配置和物料估算

配置提交已经支持：

- 至少四组且不共线的特征点对；
- 特征点重复、越界及退化校验；
- 单应矩阵计算；
- 内点数量、平均误差和最大误差；
- 多边形顶点、重复、自相交、越界和面积校验；
- 单工作区域限制；
- 系统默认值与用户配置值来源标记；
- `vehicleOverrides` 唯一性和容量范围校验。

车辆容量匹配优先使用上层提供的 `vehicleKey`。旧 tracking JSON 可提供
`vehicleKey` 或 `plateNumber`。在尚未接入车辆识别模块时，也可以显式使用
`vehicleVisitId`、原始 track ID 或 `track:<id>` 作为联调匹配键。

当前用量规则为：

```text
实际装载量
> 车辆覆盖容量
> 用户提交的默认容量
> 系统默认容量
```

只有 `COMPLETED` 事件计入累计，其他事件用量为 0。

## 7. 事件、统计和轨迹查询

### 事件查询

- 时间范围采用 `[startTime, endTime)`；
- 时间必须显式携带时区；
- 最长查询范围为 31 天；
- 支持相机、区域和多事件状态过滤；
- 每页最多 100 条；
- 使用开始时间和 `vehicleVisitId` 稳定倒序排列。

事件证据当前包括：

```json
{
  "minConfidence": 0.9,
  "boundaryCrossing": true,
  "stateTransitions": 4,
  "enterFrame": 0,
  "exitFrame": 100,
  "sourceTrackCount": 1
}
```

### 统计查询

- 仅累计 `COMPLETED` 事件；
- 分别返回完成、路过、未完成和活动事件数量；
- 支持相机、区域及 `materialType` 过滤；
- 支持 `NONE / HOUR / DAY` 分组。

### 轨迹查询

- 使用 `fromSeq` 和 `limit` 增量分页；
- 返回 `nextSeq`；
- 支持 `samplingMs` 抽样；
- `coordinateType=PIXEL` 返回像素坐标；
- `coordinateType=NORMALIZED` 返回归一化场景坐标；
- `coordinateType=BOTH` 返回全部坐标。

## 8. 保存和数据一致性

业务层已通过 `BackendRepository` 与具体数据库实现解耦。当前提供 Memory 和 SQLite
开发实现，后续 MySQL 仓储只需实现相同接口，无需修改 API、任务或算法管线。

已经实现的数据一致性规则：

- 任务最终状态、最终事件和轨迹通过统一仓储操作保存；
- SQLite 开发实现使用单事务提交；
- 任一事件冲突会回滚整次任务结果保存；
- 事件使用 `(taskId, vehicleVisitId)` 幂等更新；
- 轨迹使用 `(taskId, vehicleVisitId, seq)` 幂等写入；
- 最终事件不会被后续重复 `ACTIVE` 消息降级；
- 两个不同最终状态不能静默互相覆盖；
- 重复帧和重复保存不会造成重复计量；
- 配置版本只新增，不原地覆盖；
- 事件完成及物料估算在同一个事件记录中保存后才推送给前端。

保存和查询失败使用 `MP0401`；查询范围、页大小和轨迹上限错误使用 `MP0402`。

## 9. 当前验证结果

截至 2026-08-28：

- 54 项单元及集成测试全部通过；
- Python 源码编译检查通过；
- 已覆盖场景上传、配置标定、异步历史任务、事件保存、统计、轨迹和 WebSocket；
- 已覆盖 WebSocket 帧结果、活动事件、最终事件、节流轨迹合并和历史缓冲截断；
- 已覆盖事务回滚、事件状态冲突、幂等保存和 `MP0401/MP0402`；
- 随包测试视频可读取，共 7500 帧，25 FPS；
- 随包单应矩阵可读取，尺寸为 3×3。

测试命令：

```powershell
$env:PYTHONPATH="src"
python -m unittest discover -s tests -p "test_*.py" -v
```

## 10. 当前可进行的前端联调

现在可以联调：

1. 上传场景图；
2. 提交特征点、区域和物料配置；
3. 使用 `videoId` 创建历史视频任务；
4. 连接 WebSocket 绘制检测框、位置和轨迹；
5. 接收活动和最终事件；
6. 停止任务；
7. 查询事件、统计和轨迹；
8. 验证断线重放和 `SNAPSHOT`。

随包示例视频 ID：

```text
video_test_cam2
```

本地启动所需环境变量参考 `config/backend.env.example`。默认模型为
`models/best.pt`，视频注册表为 `config/video_registry.yaml`。

## 11. 尚未完成及待确认事项

### 11.1 实时相机输入

`CAMERA_STREAM` 请求模型已经具备，但尚未实现相机源适配器。目前缺少：

- 根据 `cameraId` 获取视频流或帧队列；
- 实时帧持续读取；
- 断流重连；
- 相机分辨率获取；
- 停止时释放相机资源。

历史视频与实时相机将复用同一算法、任务和 WebSocket 输出结构。

### 11.2 MySQL 正式仓储

仓储协议、事务边界、分页查询、统计和幂等规则已经完成，但尚未开发 MySQL 适配器。
后续需要补充：

- MySQL 版本及连接参数；
- 数据库、账号和权限；
- 连接池参数；
- 表结构和索引；
- 数据库迁移工具；
- MySQL 并发和事务测试。

### 11.3 暂缓项

- Bearer Token 鉴权和数据权限；
- 人工用量调整及审核接口；
- 多区域、多视频并行和跨摄像头车辆去重。

### 11.4 前后端需要确认的契约细节

- 相机原始画面尺寸最终由相机管理模块还是配置接口提供；
- `SNAPSHOT` 的正式字段是否采用当前实现；
- 多车同时存在时是否将 `lastTrajectorySeq` 调整为按 `vehicleVisitId` 提交；
- `materialType` 是否只使用 `CONCRETE`，还是增加强度等级字段；
- `REJECTED` 的具体业务触发条件；
- 轨迹不存在是否需要新增独立错误码，避免复用 `MP0204`。

## 12. 后续建议顺序

1. 前端立即开始历史视频模式联调；
2. 固化一套场景、配置、任务、WebSocket、事件和轨迹联调样例；
3. 明确相机管理模块输入接口并实现 `CAMERA_STREAM`；
4. 确认 MySQL 环境后实现正式仓储和迁移；
5. 补充生产鉴权；
6. 完成完整接口契约测试和最终交付文档同步。
