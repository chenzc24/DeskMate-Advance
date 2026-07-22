# Live Runtime 软件验证记录

日期：2026-07-22

物理运动：无；所有 Dealer ACK 来自显式 `SimulatedDealerAdapter`。

## 完整手牌 Replay

执行：

```powershell
.\.venv\Scripts\python.exe scripts\runtime\run_hand.py `
  --profile laptop --mode replay `
  --session-id final-validation-20260722 --hand-id complete-hand-v2 `
  --log-jsonl runs/runtime/validation/complete-hand-replay-20260722-v2.jsonl
```

结果：

- `phase=settled`
- `steps=111`
- `state_version=56`
- `engine_events=149`
- `evidence_records=53`
- `log_check_passed=true`
- 发牌次数遵循无烧牌流程：8 张 Hole + 5 张 Board。

随后以第一份日志中的 exact Identity/Action/Card observations 作为输入，在
`robot_camera` Profile 的无设备 Replay 中重新执行：

```powershell
.\.venv\Scripts\python.exe scripts\runtime\run_hand.py `
  --profile robot_camera --mode replay `
  --replay-log runs/runtime/validation/complete-hand-replay-20260722-v2.jsonl `
  --log-jsonl runs/runtime/validation/exact-recorded-replay-20260722-v3.jsonl
```

第二次同样以 `settled` 结束，并通过独立 Checker。Replay 模式不打开机器人 MJPEG、
Laptop 摄像头或麦克风。

## Live 资产与组合检查

两个非物理 Profile 的 `live-preflight` 均成功校验：

- YuNet/SFace identity assets；
- MediaPipe gesture 与 pose assets；
- LGD 52-class card ONNX/classes；
- Vosk English model；
- Vosk speaker model（本场声纹模板只在内存中，未保存音频或 embedding）；
- 所有运行时下载关闭，frames/audio 不保存。

Face、Gesture、Pose、Card 模型也在不打开设备的情况下完成同一
`LivePerceptionSession` 构造/释放检查。

最终仓库测试为 `284 passed`；runtime Profile Schema、全部 config JSON、相对文档
链接、Python compileall 与 `git diff --check` 均通过。

## 尚未关闭的物理/数据 Gate

- Laptop 和机器人目标桌面几何均标记 `target_geometry_validated=false`。
- 机器人摄像头的 Face/Hand/Card ROI 与 calibration 尚未实测冻结。
- Hole 牌占用与背面朝向没有 admitted model；开发 Live 只能使用显式操作员确认，
  因此不能计为 Gate 2B 或 I4.4 通过。
- 未执行四名真人、目标摄像头长运行或实体 Dealer 测试。

因此当前证明的是软件纵向闭环、日志/Replay 和双 Profile 组合正确，不是目标桌面或
真实机器人验收。
