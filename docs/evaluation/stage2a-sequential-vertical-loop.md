# Stage 2A：顺序式身份—多模态动作纵向闭环

状态：`development_feasibility_only`。本闭环把此前分别可运行的人脸、手势、英文语音和游戏状态机接到同一个 Laptop runtime；机器人动作仍由模拟器代替。

```text
HandEngine.acting_seat
  -> semantic rotate_to(seat)
  -> SimulatedDealer succeeded ACK + at_target
  -> session face identity matched
  -> open state-owned gesture + English speech window
  -> 500 ms bounded multimodal fusion
  -> ActionPromoter temporal/confidence gate
  -> deterministic legality + atomic ledger/state commit
  -> next acting_seat
```

## 多模态当前状态

此前已经存在手势/语音 observation 和静态融合函数，但三个关键部分尚未连上：身份通过前关闭动作输入、首个模态候选的等待窗口、动作提交后自动重置上下文并转向下一玩家。本 runtime 已补齐这些门控。

- 两个模态在 1.5 s 新鲜度范围内给出相同动作：立即形成 `multimodal_agreement`。
- 两者给出不同动作：形成 `ambiguous`，不提交，保持当前玩家。
- 只有一个模态给出候选：等待 500 ms 后形成 `single_modality_candidate`。
- 身份未通过时，麦克风块会被丢弃，手势不会进入动作 adapter。
- 每次 `state_version/focus_seat` 改变后，手势时序、Vosk 解码窗口、融合缓存和身份确认均重置。
- `unknown/seat_mismatch` 只保持身份窗口；非法、低置信或过期动作只保持动作窗口。

## Laptop 操作

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\perception\live_sequential_part_a.py --index 0 --backend dshow --speech-device 1 --consent-confirmed --max-seconds 900
```

机器人 MJPEG 视频流模式使用同一套人脸、手势、语音和状态机，仅替换相机输入：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\perception\live_sequential_part_a.py --stream-url http://100.80.46.54:5000/video_feed --speech-device 1 --consent-confirmed --max-seconds 900
```

该模式不保存帧，后台只保留最新一帧；被覆盖的积压帧记录为
`FramePacket.dropped_before`。MJPEG 不含源端拍摄时间戳，因此当前时间语义是
laptop 收帧的 monotonic 时间。此入口仍使用 `SimulatedDealer`，不会让机器人
转动；真实转向 ACK 和稳定等待仍属于后续 Robotics 集成门。

若单条 MJPEG 连接结束但服务仍可达，相机层会进行最多五次有界重连。重连期间
runtime 只记录 `camera_read_status=missing`，不清空注册库、不接受新动作、也不
推进状态；恢复后记录 `camera_reconnected`。只有重连全部失败才记录
`disconnected` 并安全退出。

Setup 阶段：

- `1`–`4` 只选择注册目标，不改变游戏状态。
- `E` 为目标注册 `player_a`–`player_d`；每名玩家需要单独明确同意。
- `S` 创建一手四人牌局。Button 默认 `seat_a`，因此第一位行动者是 UTG/枪口位 `seat_d`。

默认模式是 `four_player_core`：`S` 前必须完整注册 `seat_a`–`seat_d`，缺少任何席位都会被硬拒绝。两人夹具必须在启动时显式增加 `--player-mode two_player_pilot`，且只接受两个顺时针相邻席位；两名玩家各完成一次身份—动作后即进入 Pilot 完成边界，不会继续到未注册席位。模式不再由注册人数自动推断。

四人模式动作窗口增加持续身份守卫：每帧必须仍能匹配同一 `player_id/focus_seat` 才允许新的手势或语音 evidence 进入；短暂丢脸有 1000 ms 默认 grace，但 grace 内也不接受新动作。出现另一名已注册玩家或多人入镜会立即关闭动作窗口，清空手势、Vosk 和融合缓存，返回身份核验。

四人模式中语音不是单独的权威动作。Vosk 候选必须在 3000 ms 新鲜度窗口内得到相同手势，或由现场 UI 按 `C` 显式确认；冲突仍为 `ambiguous`。两人 Pilot 可继续使用单语音候选。结构化日志现覆盖注册开始/完成/拒绝、转向请求/ACK、身份状态变化、手势状态、每条 final speech、融合决定和游戏版本转换。

运行阶段：

- runtime 自动产生并确认模拟 `rotate_to`；真实机器人没有连接。
- 只有当前 `acting_seat` 可按 `E` 补注册；`1`–`4` 不再能切换 focus。
- 身份 matched 后，玩家可使用五个手势或英文 `fold/check/call/bet/raise`；四人模式语音单候选需按 `C` 或给出相同手势确认。
- 窗口显示当前 phase、focus、合法动作和原始手势；JSON 输出记录融合、接受/拒绝原因和下一席位。
- `X` 仅在 Setup/身份核验阶段清空注册库，动作窗口开启时会拒绝清空；`Q/Esc` 退出并清空。

当前只闭合 Part A 的一个 betting round。若状态机进入 `dealing_board/settled`，runtime 停在 `round_complete`，等待 Part B 牌面和 Stage 3 发牌 ACK 后再继续下一 street；它不会伪造公共牌或发牌完成。

## Gate 尚未关闭

Laptop 冒烟只证明四个本地模型和摄像头/麦克风能够共存。还需要四名 held-out 参与者的顺序测试，报告每个动作的 precision/recall/F1、false accepted actions、语音/手势冲突、跨玩家/跨席泄漏、身份拒识、P95 端到端延迟，以及真实 `rotate_to` ACK/停稳后图像。500 ms、1.5 s 和各模型阈值均为 Pilot 参数。

四人真人执行步骤、固定 D→A→B→C 序列、九个安全/恢复 case、不可覆盖的 JSONL 证据和自动分析命令见[四人真人验收执行包](stage2a-four-player-live-acceptance.md)。该执行包当前仅为 `prepared_not_executed`。

设备/资产预检、匿名 session、九 Case 汇总、数据 manifest/split、离线安全 replay 与可选 landmark TCN 后训练骨架见[验收前基础设施](stage2a-prevalidation-infrastructure.md)。这些产物只表示工具已准备，不表示真人 Gate 或模型录取完成。
