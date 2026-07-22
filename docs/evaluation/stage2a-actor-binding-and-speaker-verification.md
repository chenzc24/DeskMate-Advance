# Stage 2A：玩家归属与会话声纹验证

状态：`development_feasibility_only`。本页描述玩家动作进入游戏状态机之前的归属门控；它不改变游戏规则，也不允许模型选择行动席位。

## 解决的两个问题

多人同时入镜时，不再使用“检测器返回的第一只手”。当前纵向链路为：

```text
HandEngine.acting_seat（唯一注意力来源）
  -> 已注册人脸匹配当前席位
  -> 人脸框关联目标人体 Pose track
  -> 目标人体左右手腕关联最多四只候选手
  -> 只让目标人体的手势进入时序确认
  -> ActorBinding + PlayerActionObservation
  -> 确定性合法性检查与原子状态/账本提交
```

绑定包含 `session_id/hand_id/state_version/focus_seat/player_id/person_track_id/camera_epoch`，默认租约为 2500 ms。短时遮挡或多脸只会让租约自然过期；识别到另一名已注册玩家、摄像头重连、状态版本变化或租约超时会立即失效。邻座的手、无法唯一关联到手腕的手以及两只目标手给出冲突动作都不会成为候选。

语音增加会话级说话人验证，但声纹只回答“这段话是否来自已经绑定的玩家”，不能决定轮到谁：

```text
绑定玩家说 fold/check/call/bet/raise
  -> Vosk 英文 ASR + 同一段音频的 speaker embedding
  -> 内存注册库匹配绑定 player_id
  -> pending action
  -> 同一声纹说 confirm：提交 observation
     同一声纹说 cancel：撤销 pending
     不同/未知/过短声纹：拒绝
```

## UI 验收流程

使用 `--consent-confirmed` 启动后：

1. 按 `1`–`4` 选择注册席位，按 `E` 完成人脸注册。
2. 保持该角色，按 `V` 开始声纹注册。屏幕逐段显示 `LISTENING 1/3` 到 `3/3`，本人按提示将七个英文动作词在一口气内读完；每段会明确显示并播报 accepted 或 retry。三段接纳后自动结束，再按一次 `V` 可随时取消。单个动作词过短，不能作为有效声纹样本。
3. 四名玩家完成人脸注册后按 `S` 开局；声纹是语音功能的前置条件，但不阻塞纯手势模式开局。
4. 当前行动者通过人脸与人体绑定后，说动作词，再说 `confirm`；说 `cancel` 可撤销。
5. `C` 仍是明确标记为 operator/UI override 的人工确认，不会伪装成声纹验证。
6. `X` 在允许的阶段清空人脸库和声纹库；退出进程也会清空。音频、视频和两类 embedding 均不写盘。

结构化日志会记录 `actor_binding_*`、`speaker_enrollment_*`、`speaker_verification`、`speech_confirmation_state` 和最终拒绝/接纳原因，但不会记录声纹向量。

## 模型与开放 Gate

- 手/姿态：MediaPipe Gesture Recognizer + Pose Landmarker Lite，均为 development evidence。
- 语音：`vosk-model-small-en-us-0.15`。
- 声纹：`vosk-model-spk-0.4`，tree SHA-256 为 `bfb2d247c774f5e721607923e14ee1d91282fcb158c9d634ecae7d14cae28639`。
- 当前相似度 `0.70`、第二名 margin `0.08`、最少三段注册音频和最少 40 speaker frames 都只是可配置默认值。

进入 candidate 前仍需真人 held-out participant/session 数据，至少报告 false match、false non-match、跨玩家泄漏、多人手错归属、拒识、噪声/距离/机器人电机噪声、重放攻击以及 P95 确认延迟。真实机器人 `rotate_to` ACK 仍未接入；软件只使用模拟 ACK，并在 ACK 后要求新帧和视觉稳定，不能据此声称物理联调完成。
