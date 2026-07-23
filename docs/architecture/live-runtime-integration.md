# Live 感知接入统一状态机

## 结论与边界

当前已经建立一条共享的软件路径：

```text
RuntimeProfile
  -> 单一 FrameSource / 可选 Microphone / DealerPort
  -> RegistrationSource 冻结四人 Roster
  -> SessionRuntime（一次注册、Button/筹码/清台/重买/结束）
     -> HandRuntime 1..N
       -> Part B: rotate ACK -> dispense ACK -> CardObservation
       -> Part A: rotate ACK -> visual settle -> identity -> ActionEvidence
       -> Showdown: confirmed slots -> deterministic evaluator -> ledger
  -> 每手 runtime log + 跨手 session log -> independent checkers
```

Laptop 与机器人 MJPEG 使用同一个 `HandRuntimeLoop` 和
`LivePerceptionSession`。Profile 只选择设备、模型配置和 calibration ID；不允许按
Profile 复制规则、下注逻辑、Part A 或 Part B。

这条路径的开发版本可运行，但不能描述为 Gate 2B/4 已通过。现有牌模型只识别活动
固定 ROI 中的正面牌身份；13 槽占用、Hole 背面朝向和目标桌面几何尚未得到模型与
实测标定。开发 Live 模式因此要求操作员按 `F` 明确确认 Hole 牌背面状态，并在日志
中标为 `not_gate_2b_model_evidence`。默认不提供该开关时，Live 启动会在打开设备前
拒绝运行。

## 端口与权威

| 端口 | 代码 | 唯一输出/作用 |
| --- | --- | --- |
| Frame | `runtime/ports.py` | 一次读取一个 `FramePacket`；当前阶段只有一个消费者入口 |
| Registration | `LivePerceptionSession.acquire_roster()` | 四名 consented participant 的 `FrozenSessionRoster` |
| Identity | `observe_identity()` | 只核验状态机已选中的 `focus_seat` |
| Action | `observe_action()` | 语音/手势与 actor binding；只产生 evidence |
| Card | `observe_card()` | 当前 Part B step 指定槽位的 `CardObservation` |
| Dealer | `robotics/dealer/port.py` | `DealerCommand -> DealerAck`，不暴露马达参数 |
| Event | `RuntimeEventWriter` | 原始 evidence 与 Engine 事件的独占、Hash-chain JSONL |

`HandRuntimeLoop` 是唯一调度者。它每次只读取一个共享帧，并根据
`HandRuntime.part_a/part_b` 选择当前允许调用的源。模型看不到“下一个行动者选择”
接口，也不能调用 game reducer、账本或 Dealer。

正式玩家动作路径是：

```text
FaceIdentityObservation MATCHED
  -> pose/person track
  -> expiring ActorBinding
  -> target-attributed gesture and/or current-player speaker verification
  -> English speech confirmation when speech is the only modality
  -> PlayerActionObservation
  -> attribution/state version/current seat/legal action gates
  -> HandEngine atomic action + ledger commit
```

按钮不产生 fold/check/call/bet/raise。启用语音时，每名玩家在人脸注册后还要完成三份
本场声纹样本；声纹模板只存在内存，退出时清零，不记录音频或 embedding。英语命令只有
在声纹匹配状态机当前 `player_id` 后才会进入融合窗口，其他玩家、unknown 或 ambiguous
语音直接丢弃。纯语音候选还需同一声纹说 `confirm`，或由操作员按 `C` 确认；与当前玩家
手势一致时可作为双模态 evidence。Backspace/X 只取消候选。直接按钮下注仍隔离在
`poker_dealer.pilots`。

## Runtime 模式

### 1. 只读 Profile 检查

```powershell
.\.venv\Scripts\python.exe scripts\runtime\run_hand.py --profile laptop --mode preflight
.\.venv\Scripts\python.exe scripts\runtime\run_hand.py --profile robot_camera --mode preflight
```

不打开设备。

### 2. Live 模型资产检查

```powershell
.\.venv\Scripts\python.exe scripts\runtime\run_hand.py --profile laptop --mode live-preflight
.\.venv\Scripts\python.exe scripts\runtime\run_hand.py --profile robot_camera --mode live-preflight
```

读取配置并校验 Face、Gesture、Pose、Card、Vosk English 与 Vosk speaker 的本地
资产 Hash，不打开摄像头或麦克风。输出会继续报告
`target_geometry_validated: false`。

### 3. 完整软件 Replay

```powershell
.\.venv\Scripts\python.exe scripts\runtime\run_hand.py `
  --profile laptop --mode replay `
  --session-id replay-001 --hand-id hand-001 `
  --log-jsonl runs/runtime/replay-001/hand-001.jsonl

.\.venv\Scripts\python.exe scripts\runtime\check_hand_log.py `
  runs/runtime/replay-001/hand-001.jsonl
```

Replay 使用精确的 domain observations、模拟 Dealer ACK 和正式
`require_actor_binding/require_visual_settle` 路径完成 Hole、四轮下注、五张 Board、
Showdown 和结算。可以把第一份日志作为 `--replay-log` 输入，证明同一 evidence
再次产生相同最终状态。

连续 20 手软件资格测试使用同一个 Roster、持续筹码和轮转 Button；每手仍有独立日志：

```powershell
.\.venv\Scripts\python.exe scripts\runtime\run_hand.py `
  --profile laptop --mode replay --max-hands 20 `
  --session-id replay-20-001 `
  --session-log-jsonl runs/runtime/validation/replay-20-001/session.jsonl

.\.venv\Scripts\python.exe scripts\runtime\check_session_log.py `
  runs/runtime/validation/replay-20-001/session.jsonl
```

### 4. 开发 Live 模式

```powershell
.\.venv\Scripts\python.exe scripts\runtime\run_hand.py `
  --profile laptop --mode live `
  --session-id laptop-live-001 --hand-id hand --button seat_a --max-hands 20 `
  --consent-confirmed --development-operator-face-down `
  --speech-device 1 --max-seconds 900 --max-steps 20000
```

机器人摄像头只替换 Profile：

```powershell
.\.venv\Scripts\python.exe scripts\runtime\run_hand.py `
  --profile robot_camera --mode live `
  --session-id robot-cam-live-001 --hand-id hand-001 --button seat_a `
  --consent-confirmed --development-operator-face-down `
  --speech-device 1 --max-seconds 900 --max-steps 20000
```

注册按 Button、Small Blind、Big Blind、UTG 顺序进行：`E`/Enter 开始当前角色
人脸采样；启用 speech 时随后按 UI 提示让该玩家说三次闭集命令，完成本场声纹注册后
才转到下一个角色。四人完成后按 `S` 冻结 Roster。游戏中 Gesture 或经当前玩家声纹
绑定/确认的英语语音产生动作；`C` 确认 pending speech，Backspace/X 取消；`F` 是上述
Hole 牌开发 fallback；`Q`/Esc 安全退出并进入暂停日志。使用 `--disable-speech` 时只做
人脸注册，不创建声纹库。

每手结算后必须先人工收回全部牌，再按 `C` 确认清台。清台后 `S` 开始下一手，
`X`/Backspace 结束整场；低于最小起手筹码时用 `N/P` 选玩家、`C` 执行有审计的
重买。暂停恢复时 `S` 只在状态一致且无冲突槽后重试，`N/P` 选择冲突槽、`C` 确认
该槽已空，`X` 作废本手且 Button 不轮转。达到 `--max-hands` 后，最后一次清台确认
直接安全结束整场。

Laptop Profile 已使用独立的 13 槽开发配置。可从固定机位截图交互生成新的、仍未
验收的配置：

```powershell
.\.venv\Scripts\python.exe scripts\perception\calibrate_card_slots.py `
  --image data/work/laptop-table.png `
  --output configs/perception/card_slots_laptop_local.json `
  --calibration-id laptop-local-001
```

工具逐槽要求框选并强制 13 槽不重叠，但始终写入
`target_geometry_validated=false`；只有真实桌面评估才能改变 Gate 状态。

## Replay 与日志校验

`RuntimeEventWriter` 把两类记录放在同一个独占 JSONL 中：

- `runtime_evidence`：Identity、Action、Card 以及设备状态；不含 embedding、图像或
  音频。
- `engine_event`：Engine 自己的 Hash-chain 事件，包括命令、ACK、动作、槽位、
  street 和结算。

外层和 Engine 内层分别有 SHA-256 链。`check_hand_log.py` 还会重新检查：

- state version 连续性；
- 数字筹码守恒；
- 重复牌；
- issued command 与 successful ACK 集合；
- 结束时不存在 pending command；
- 从最终 board、hole cards 和 pots 重新运行 evaluator，验证 awards。

`SessionEventWriter` 另存一条跨手 SHA-256 链。`check_session_log.py` 会重新打开每手
日志核验文件 SHA-256 和单手 Checker，并检查手牌 ID 唯一、起止筹码连续、Button
轮转（作废手不轮转）、清台门、调整账目和安全结束。声明“某手已通过”本身不被信任。

## 双路并行

Laptop 本机摄像头和机器人 MJPEG URL 使用不同资源 ID，可以运行两个独立进程。
如果两路都使用同一个 Laptop 麦克风，第二路会失败；应给其中一路关闭 speech 或
配置不同输入设备。两路必须使用不同 session/hand ID 和日志目录。真实 Dealer 始终
只能由一个进程持有。

## Robotics Handoff

本仓库只冻结 `DealerPort`、语义 target、command correlation、ACK 证据验证、超时、
幂等和暂停策略。Robotics 负责 transport/firmware、角度与运动、homing、feeder、
传感器、interlock、E-stop 和 watchdog。`robot_hardware` 当前仍 `enabled: false`，
任何 live/replay 命令都不能把它降级成模拟硬件并继续。

只有 Robotics 提供 protocol/firmware/calibration/safety release 后，才新增真实
Adapter 并执行 Stage 3 的 mock、bench、低速和 operator-witnessed Gate。本次软件
集成没有授权任何物理运动。
