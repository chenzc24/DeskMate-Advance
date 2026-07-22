# Stage 2A：验收前基础设施与后训练准备

状态：`prepared_not_executed / prepared_not_trained`。本包把四人验收前可以完成的 Part A 工作全部落地，但不生成真人指标、不冻结阈值、不录取模型，也不修改 Part B 或 Robotics 的独占实现。

## 1. 环境与资产预检

预检只读检查 Python/依赖版本、五个 development 模型/支持资产的 hash、协议、模型清单、磁盘、摄像头单帧和麦克风输入；不保存媒体：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\perception\preflight_part_a_acceptance.py `
  --camera-index 0 --camera-backend dshow --speech-device 1 `
  --output runs\stage2a_four_player_acceptance\preflight-20260722.json
```

`--skip-devices` 只适用于 CI/离线复核，不能替代明日现场设备检查。输出文件使用 exclusive create，防止覆盖旧证据。

## 2. 匿名 session 与同意记录

不使用姓名、学号、邮箱等真实身份。四名参与者分别确认后，操作员创建本地 ignored 记录：

```powershell
.\.venv\Scripts\python.exe scripts\perception\prepare_four_player_session.py `
  --session-group fourp-20260722-01 `
  --operator-code OP01 `
  --seat-a-code P01 --seat-b-code P02 --seat-c-code P03 --seat-d-code P04 `
  --all-consent-confirmed `
  --lighting office-even --camera-distance-cm 85
```

记录位于 `runs/stage2a_four_player_acceptance/<session-group>/session_record.json`，不会进入 Git。四个 participant code 必须不同；runner 在启动真人注册前再次验证四席同意与隐私字段。

## 3. 单 Case 与九 Case 汇总

每个 Case 使用同一个 `--session-group`，但生成唯一 attempt/session/hand：

```powershell
.\.venv\Scripts\python.exe scripts\perception\run_four_player_acceptance.py FPA-01 `
  --session-group fourp-20260722-01 `
  --index 0 --backend dshow --speech-device 1 --consent-confirmed
```

退出 live UI 后，按 runner 打印的 JSONL 路径记录现场观察；以下只是格式示例：

```powershell
.\.venv\Scripts\python.exe scripts\perception\record_acceptance_case_observation.py `
  <attempt-dir>\FPA-01.jsonl `
  --operator-code OP01 --observed-result observed_pass `
  --handedness-used right --camera-distance-cm 85 --lighting office-even `
  --speech-used yes --gesture-used yes --failure-category none
```

该记录补充机器日志无法证明的实际动作、左右手、距离、光照、使用模态和失败类别。它必须与 JSONL 同目录；缺失、上下文不一致或操作员标记失败时，batch 不会把机器 PASS 当作整体 PASS。

全部执行后汇总：

```powershell
.\.venv\Scripts\python.exe scripts\perception\summarize_four_player_acceptance.py `
  --session-group fourp-20260722-01
```

汇总器不删除失败尝试，也不静默选择最新结果。状态含义：

- `COMPLETE_PASS`：九 Case 均有 PASS 且没有失败尝试。
- `COMPLETE_WITH_RETRIES`：九 Case 均最终有 PASS，同时保留至少一次 FAIL。
- `INCOMPLETE`：仍缺 Case。
- `FAIL`：某 Case 已执行但没有 PASS，或证据无法解析/跨 session。

## 4. 离线安全 replay

无需摄像头、参与者或机器人即可验证大量 `no_action`、ambiguous、occluded、unknown、out-of-ROI、非当前席、stale、低置信、非法和重复 observation 均不改变状态/账本，之后正确动作仍能恢复：

```powershell
.\.venv\Scripts\python.exe scripts\perception\replay_part_a_safety.py `
  --no-action-events 10000 `
  --output runs\stage2a_four_player_acceptance\part-a-safety-replay.json
```

它证明 deterministic gate，不证明真实模型不会把普通动作错误分类为 candidate。

## 5. 行为数据 manifest 与 split

源 manifest schema 为 [`stage2a_action_source_manifest.schema.json`](../../configs/evaluation/stage2a_action_source_manifest.schema.json)。每条记录必须包含：

- 匿名 participant/session、seat、camera、lighting、label、duration；
- `data/raw/...` 下的只读相对路径、原始 bytes 和 SHA-256；
- `contains_identity_media=true`、`git_tracked=false`；
- 五动作以及 no-action/cancelled/ambiguous/occluded 负样本。

解析并按 participant 整体分配 train/validation/test：

```powershell
.\.venv\Scripts\python.exe scripts\perception\prepare_action_dataset_manifest.py `
  data\work\action-source-v1.json `
  --verify-files `
  --seed stage2a-action-split-v1 `
  --output data\work\action-resolved-v1.json
```

同一 participant 的所有 session、同一 session 和完全重复字节不得跨 split。工具不移动或修改原始媒体。

## 6. Landmark 派生与 TCN

只有明日证据表明 canned gesture 的时序/误触问题无法靠交互或规则解决时才执行。准备好的配置是 [`action_tcn_v1.json`](../../configs/training/action_tcn_v1.json)，状态固定为 `prepared_not_trained`。

从 resolved manifest 生成 ignored landmark view：

```powershell
.\.venv\Scripts\python.exe scripts\perception\extract_action_landmarks.py `
  data\work\action-resolved-v1.json
```

派生层使用已哈希的 MediaPipe Hand Landmarker，只在恰好一只手时输出 21×3、腕点平移和尺度归一化的 63 维特征；无人手或多手帧只产生无效 mask，不猜测。

先做无训练检查：

```powershell
.\.venv\Scripts\python.exe scripts\perception\train_action_tcn.py --check-config
.\.venv\Scripts\python.exe scripts\perception\train_action_tcn.py `
  data\work\action_landmarks_v1\view_manifest.json --dry-run
```

真正训练需要显式安装可选依赖 `.[training]`。训练器实现 participant-safe view 检查、32 帧窗口、class weighting、early stopping、held-out confusion/per-label P/R/F1、development checkpoint、TorchScript export 和 reference/export 数值一致性。输出始终在 ignored `runs/`，不会自动修改 `models/manifest.yaml` 或晋升模型状态。

## 7. 模型清单与剩余 Gate

`models/manifest.yaml` 已登记：

- 现有 MediaPipe canned gesture、Vosk English、YuNet、SFace：`development`；
- MediaPipe Hand Landmarker：只作为离线派生 supporting asset；
- `player-action-landmark-tcn@untrained-v1`：无权重、无指标、`development`。

TCN 的 blockers 明确保留：四人失败证据、source/view manifest hash、训练与 export hash、held-out 指标、false acceptance/跨席/取消/延迟和 target-camera 迁移。任何离线 accuracy 都不能单独移除这些 blocker。

## 8. 轨道隔离

本包未实现牌面 ROI/classifier、机器人 transport 或物理运动。Part B 可以由其 owner 独立推进 fixed-ROI/牌角/分类准备；Robotics 可以独立推进 protocol/mock。Part A 只通过冻结 schema、state version 和模拟 ACK 与它们汇合。

最终交接字段、必备指标、bundle 禁止内容和状态晋升条件见 [Part A Handoff Checklist](stage2a-part-a-handoff-checklist.md)。当前模板保持 `handoff_not_ready`，直到真人与 target-camera 证据真正存在。
