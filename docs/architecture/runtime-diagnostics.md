# Runtime 全程诊断包

## 目标与边界

正式入口 `scripts/runtime/run_hand.py` 提供可选的 Diagnostics 旁路。它观察同一个
Runtime、状态机、感知端口和 Dealer 端口，不提供第二套 Debug 规则，不改变模型阈值、
动作合法性、账本、ACK 或安全门槛，也不能让 `robot_hardware` 绕过禁用状态。

Diagnostics 用于回答现场运行的因果问题；原有 hand/session JSONL 仍是牌局审计权威：

- Diagnostics：启动、配置、设备、阶段、异常、每步/端口耗时和 stdout/stderr；
- hand log：原始 observation、Engine event、Dealer command/ACK 和最终状态；
- session log：跨手筹码、Button、清台、恢复、重买和结束边界。

## 启动

开发 Laptop 全程运行：

```powershell
.\.venv\Scripts\python.exe scripts\runtime\run_hand.py `
  --profile laptop --mode live `
  --session-id field-debug-001 --hand-id hand `
  --button seat_a --max-hands 20 `
  --consent-confirmed --web-console --announcer windows `
  --registration-timeout-seconds 900 `
  --diagnostics
```

无设备验证 Diagnostics 本身：

```powershell
.\.venv\Scripts\python.exe scripts\runtime\run_hand.py `
  --profile laptop --mode replay `
  --session-id diagnostics-replay-001 --hand-id hand-001 `
  --diagnostics
```

默认目录为 `runs/diagnostics/<profile>/<run-id>/`，最终 CLI JSON 会返回绝对
`diagnostics_path`。可以用 `--diagnostics-dir runs/diagnostics/<new-id>` 固定目录；目录
必须位于 ignored `runs/` 且不能已经存在。`--diagnostics-max-records` 和
`--diagnostics-max-mib` 分别限制每条 JSONL 和每个 stdio 文件，默认是 100,000 条、
32 MiB。

## 运行包

```text
<run-id>/
├─ manifest.json
├─ runtime.jsonl
├─ metrics.jsonl
├─ stdout.log
├─ stderr.log
├─ configs/
│  ├─ runtime_profile.json
│  └─ ...
├─ session.jsonl
├─ hands/
│  ├─ hand-001.jsonl
│  └─ ...
└─ summary.json
```

- `manifest.json`：运行参数、Python/平台/进程、隐私声明、容量上限，以及 Profile、
  Core 和感知配置原文件/脱敏快照的字节数和 SHA-256。
- `configs/`：运行开始时生成的 JSON 配置脱敏快照，使整个目录复制到另一台机器后
  仍可验证；URL/敏感字段仅保留短 hash，原文件 SHA-256 仍写入 manifest。
- `runtime.jsonl`：带独立 SHA-256 链的启动、preflight、hand loop、错误和结束事件。
  每条立即 flush。
- `metrics.jsonl`：带独立 SHA-256 链的 Runtime step、相机读取/帧间隔、身份、动作、
  牌面、视觉稳定和 Dealer 调用耗时；同时记录起止 CPU time、线程数，并在已安装
  `psutil` 时记录 RSS。
- `stdout.log` / `stderr.log`：终端输出的有界副本；URL 在持久化副本中被哈希替代。
- `session.jsonl` / `hands/*.jsonl`：原有权威审计日志；Diagnostics 不复制或改写其
  内容，只把默认路径放进同一个运行包。
- `summary.json`：退出码、成功/失败、事件计数、所有 metric 的 min/P50/P95/P99/max、
  第一个 error/critical 事件、其前序因果窗口、运行结果和审计文件 SHA-256。

Runtime step 与端口 metric 使用共同上下文：`session_id`、`hand_id`、`step`、
`state_version`、`acting_seat`、`hand_phase`、`part_a_phase`、`part_b_phase` 和
`camera_epoch`。定位问题时先看 `summary.first_failure` 和 `causal_context`，再用这些
字段关联 `runtime.jsonl`、`metrics.jsonl` 与对应 hand log。

## 独立检查

```powershell
.\.venv\Scripts\python.exe scripts\runtime\check_diagnostics.py `
  runs/diagnostics/laptop/<run-id>
```

Checker 独立重算两条 Diagnostics hash chain，重新 hash 配置快照和 hand/session
引用，并直接重跑单手与整场 Checker；同时检查必需文件、Schema version、run ID、
退出状态和持久化 URL 泄漏。`passed=true` 说明运行包结构与审计检查完整且未被修改，
但真实硬件/目标相机 Gate 仍需各自实测证据。

若进程被受控异常终止，`uncaught_exception` 会保存错误类型、原因和 traceback，
`summary.json` 仍会生成。若断电或进程被强制杀死，已 flush 的 JSONL 仍可读取，
但缺少 summary 时 Checker 会明确报告不完整，不能误判为通过。

## 隐私与证据

Diagnostics 永不保存帧、音频或 embedding，manifest 固定声明三者为 false。URL 和
带敏感名称的字段只持久化短 SHA-256 标识。若后续确需保存故障图片，必须另行设计
显式开关、ROI/时间范围、参与者同意、保留期和删除流程；不能把媒体暗中加入本运行包。
