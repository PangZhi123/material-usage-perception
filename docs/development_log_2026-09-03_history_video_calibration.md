# 历史视频分析独立标定约束

更新时间：2026-09-03

## 目标

历史视频不得直接复用摄像头或其他历史视频已有的场景、标定和工作区域配置。
每次上传（或注册）一个新视频后，前端必须完成以下流程：

上传/注册历史视频并取得 videoId
→ 为该 videoId 重新上传场景图
→ 使用该场景图重新标定并划定一个工作区域
→ 生成绑定该 videoId 的新 configurationId
→ 使用 videoId + configurationId 创建 VIDEO_FILE 任务

## 接口约束

- POST /api/material-perception/v1/scenes 增加必填表单字段 videoId；场景图与视频绑定。
- POST /api/material-perception/v1/configurations 在历史视频模式下提交 videoId；后端校验场景图和配置均属于该视频。
- 创建 VIDEO_FILE 任务时，配置必须显式绑定同一个 videoId；未绑定、绑定其他视频或使用旧相机配置时返回 MP0202。
- CAMERA_STREAM 保持使用未绑定 videoId 的相机配置。
- 视频内容去重返回同一个 videoId 时，可继续使用该视频已有的专属配置；不同 videoId 之间不得复用。

## 前端要求

上传成功后必须清空之前选择的 sceneId、configurationId、标定点和工作区域，完成场景图上传、标定及工作区域划定前禁用“开始分析”。

