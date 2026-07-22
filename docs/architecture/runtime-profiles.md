# 双 Runtime Profile 与统一启动边界

## 当前结论

Laptop 和机器人摄像头不再被视为两套产品代码。二者使用同一个
`LiveHandApplication -> HandRuntime -> HandEngine` 组合根，只替换外部依赖。

| Profile | 摄像头 | Dealer | 用途 | 当前可启动性 |
| --- | --- | --- | --- | --- |
| `laptop` | 本机 OpenCV 摄像头 | `SimulatedDealerAdapter` | 单机模型与交互测试 | 开发 Live 入口可运行；真人/牌槽 Gate 未验收 |
| `robot_camera` | 机器人 HTTP MJPEG | `SimulatedDealerAdapter` | 目标视角实测，不驱动机构 | 同一开发 Live 入口可运行；目标标定未冻结 |
| `robot_hardware` | 机器人 HTTP MJPEG | 真实 Dealer 端口 | 最终实体闭环 | 明确不可用并返回非零状态 |

Profile 只决定摄像头、控制输入、语音设备、Dealer 适配器和日志目录，不复制规则、
Part A、Part B、账本或状态机。三个 Profile 的机器可读定义位于
`configs/runtime/`，结构由 `configs/contracts/runtime_profile.schema.json` 约束。

## 启动方法

下面的配置检查不打开摄像头、麦克风或 Dealer：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\runtime\run_hand.py --profile laptop --check-config
.\.venv\Scripts\python.exe scripts\runtime\run_hand.py --profile robot_camera --check-config
.\.venv\Scripts\python.exe scripts\runtime\run_hand.py --profile robot_hardware --check-config
```

前两个命令应返回 `ready: true`。第三个命令应返回 `ready: false` 和退出码 2；
这不是自动降级，而是尚未完成协议、固件和安全放行时的预期行为。

只测试摄像头链路时：

```powershell
.\.venv\Scripts\python.exe scripts\runtime\run_hand.py --profile laptop --camera-smoke-frames 30
.\.venv\Scripts\python.exe scripts\runtime\run_hand.py --profile robot_camera --camera-smoke-frames 30
```

可用 `--camera-index` 覆盖 Laptop 设备号，或用 `--stream-url` 覆盖机器人摄像头
地址。两者互斥。`--disable-speech` 可在不测试语音的一路释放麦克风资源。

## 并行与资源规则

进程启动前会独占它使用的资源：本机摄像头按 device index 上锁，MJPEG 按 URL
上锁，启用语音时按 microphone device 上锁，真实 Dealer 按 device ID 上锁。
因此 Laptop 摄像头和机器人 MJPEG 可以并行，但两路若都选择同一个 Laptop 麦克风，
第二路会拒绝启动。需要同时跑两路时，应给它们配置不同麦克风，或给不测语音的一路
加 `--disable-speech`。

运行日志按 Profile、session 和 hand 隔离在 ignored `runs/runtime/...` 下，使用独占
创建，不覆盖已有证据。

## 代码职责

- `runtime/profile.py`：解析并验证依赖选择，不访问设备。
- `runtime/live_hand_app.py`：唯一应用装配根，持有 camera、Dealer、资源锁和
  `HandRuntime` 创建入口。
- `runtime/hand_runtime.py`：唯一完整手牌协调门面；规则权威仍是 `HandEngine`。
- `robotics/dealer/port.py`：语义 Dealer 端口，只接收 `DealerCommand` 并返回
  `DealerAck`。
- `robotics/dealer/adapters.py`：显式模拟器和 fail-closed 真实适配器占位。
- `scripts/runtime/run_hand.py`：唯一正式 Profile 入口。
- `pilots/button_betting.py` 与 `scripts/pilots/`：按钮直连账本的窄测试，不属于正式
  动作输入链路。

## 仍未完成的边界

当前统一入口已经接入共享 Frame、Face/Gesture/English Speech/Card 端口、完整
Replay、追加日志和 `HandRuntime` event loop。两个非物理 Profile 都提供开发 Live
模式，但输出仍保持 `full_live_hand_integrated: false`：原因不是缺少调度入口，而是
13 槽目标几何以及 Hole 牌占用/背面朝向模型尚未实测冻结。开发模式只允许显式的
操作员 Hole 确认，并标为非 Gate evidence；不得据此声称 I4.4 已通过。

真实机器人闭环仍缺少发布后的协议/传输实现、firmware ACK、homing、互锁、急停、
卡牌在位/卡堵传感器、watchdog 和现场操作员安全放行。在这些 Gate 完成前，
`robot_hardware` 必须持续拒绝启动；本次工作没有授权或触发任何物理运动。
