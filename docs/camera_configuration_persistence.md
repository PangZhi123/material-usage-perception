# 相机与配置保存、删除

此前相机 POST 登记仅保存在内存，重启会丢失；标定配置已保存，但测试页面列表只依赖浏览器 localStorage，因此换浏览器、清缓存或变更站点地址后无法找回列表。旧测试页面没有后端删除接口。

现在默认文件仓库将相机连接信息保存到数据目录的 `.private/cameras.json`，启动时恢复，覆盖 YAML 中同 ID 的初始登记。删除记录也持久化，防止 YAML 相机重启后重新出现。此目录不能通过 `/files/data` 下载。列表接口不返回流地址，更新地址仍通过 POST。

接口（统一前缀 `/api/material-perception/v1`）：

- `GET /cameras`：服务端相机列表，返回 `data.items`。
- `POST /cameras`：登记或更新相机，持久化成功才返回成功。
- `DELETE /cameras/{cameraId}`：删除服务端登记；新实时任务无法再使用该登记。已有任务继续使用启动时的连接信息，不级联删除标定、视频或历史结果。
- `GET /configurations`：配置列表，返回 `data.items`，不含已删除配置。
- `DELETE /configurations/{configurationId}`：持久化标记为 DELETED，详情返回 404、新任务不可使用。保留历史任务配置快照和结果，已启动任务继续执行。

测试页面加载时从服务端恢复相机、配置列表，删除按钮先调用后端，成功后再刷新列表。独立业务前端同样需要接入上述 GET 和 DELETE，不能仅清理本地页面记录。

部署时设置 `MATERIAL_PERCEPTION_STORAGE_ROOT` 为固定、可写的持久目录；默认 `outputs/backend_data`。容器需要挂载持久卷，重新部署不可清空该目录。既有默认目录的数据需迁移到新目录后再切换环境变量。流地址可能含密码，应限制数据目录文件权限。

持久化相机适用于默认注册器；显式注入的自定义注册器需实现 list_sources/delete 并自行提供持久化。标定配置列表和删除同时支持内存、文件和 SQLite 仓库。
