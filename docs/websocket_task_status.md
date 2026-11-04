# WebSocket 任务结束通知

地址：`/ws/material-perception/v1/tasks/{taskId}`。

任务完成、停止或失败时，服务端先发送完该连接待发送的结果，再发送以下消息，最后以关闭码 `1000` 正常关闭连接：

```json
{
  "type": "TASK_STATUS",
  "taskId": "task_example",
  "taskStatus": "COMPLETED",
  "endedAt": "2026-11-04T10:00:00+08:00",
  "error": null
}
```

`taskStatus` 为 `COMPLETED`（分析完成，结果已保存）、`STOPPED`（已停止）或 `FAILED`（失败，原因见 `error`）。`endedAt` 在旧记录中可能为空。

业务前端收到 `TASK_STATUS` 后按 `taskStatus` 显示结束提示并退出运行状态。不要将车辆 `EVENT_UPDATE.event.status` 当作整个任务状态，也不要仅凭连接断开判断分析成功。意外断线时应重连或查询任务接口。

重新连接已结束任务（包括服务重启后恢复的任务）时也会收到终态通知，前端应允许重复通知并按 taskId 更新状态。现有帧、事件、心跳消息格式不变。

仓库自带测试页面已增加终态提示并在收到通知后停止状态轮询。部署更新后需重启后端并刷新页面；独立业务前端需增加上述消息处理。
