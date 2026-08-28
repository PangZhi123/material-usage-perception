# 完整回放与物料估算开发记录（2026-08-24）

## 完整五分钟验证

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

## 物料估算

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

## 验证

- 全部 33 项单元测试通过；
- 34 个 Python 文件 AST 语法检查通过；
- 完整回放、车辆事件文件和物料累计文件均已生成。

## 下一步

1. 获得至少包含一次真实卸料过程的视频或人工事件真值；
2. 校准最短停留时间及进入、离开确认阈值；
3. 定义工作相机输入配置和结构化结果输出接口；
4. 在有推理设备时接入 YOLO + ByteTrack 的实时相机帧；
5. 根据业务数据接入车辆实际装载量或车辆容量。
