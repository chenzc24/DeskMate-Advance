# 小车导航与游戏状态机接口说明

本文档说明 Poker Dealer 状态机与小车导航、发牌、人脸、玩家操作、筹码和
卡牌识别模块之间的接口细则、调用顺序及等待规则。

当前实现遵守以下单向依赖：

```text
摄像头/手机/小车输入
    -> 带版本号的 Observation 或 ACK
    -> 确定性游戏状态机
    -> 语义导航/发牌命令
    -> 树莓派或电机适配器
    -> ACK
```

识别模型只能提交观察结果，不能直接改变游戏阶段、玩家余额或控制电机。
只有游戏状态机能够决定当前玩家、合法操作以及是否进入下一阶段。

## 1. 主要入口

### 1.1 查询当前需要什么输入

以下三个入口返回 `RobotInterfaceRequirement`：

```python
RegistrationRuntime.robot_requirement()
SessionRuntime.robot_requirement()
HandRuntime.robot_requirement()
```

返回内容：

| 字段 | 类型 | 功能 |
| --- | --- | --- |
| `node` | `RobotWorkflowNode` | 当前对外等待节点 |
| `accepted_inputs` | `tuple[RobotInputKind, ...]` | 当前唯一允许提交的输入类别 |
| `accepted_python_types` | `tuple[type, ...]` | 输入类别对应的 Python 类型 |
| `hand_phase` | `HandPhase \| None` | 当前牌局阶段 |
| `state_version` | `int` | 当前状态版本 |
| `target_seat` | `Seat \| None` | 当前正在处理的玩家座位 |
| `target_slot` | `DealerTargetSlot \| None` | 小车需要前往的玩家或牌槽 |
| `vision_slots` | `tuple[VisionSlot, ...]` | 当前等待识别的指定牌槽 |
| `reason` | `str` | 状态机等待原因 |

调用方必须先查询 requirement，再提交 requirement 声明的输入。不能根据上一帧
缓存的节点猜测当前输入，因为合法动作提交后 `state_version` 会改变。

### 1.2 导航接口

`NavigationPort` 是状态机调用小车导航的边界：

```python
class NavigationPort(Protocol):
    device_id: str
    physical_motion: bool

    def open(self) -> None: ...
    def health(self) -> NavigationHealth: ...
    def execute(
        self,
        command: NavigationCommand,
        observed_at_ns: int | None = None,
    ) -> NavigationAck: ...
    def close(self) -> None: ...
```

树莓派、串口、HTTP 或其他传输方式只能在 adapter 内实现，不能进入游戏引擎。

### 1.3 发牌机构接口

`DealerPort` 负责发牌机构：

```python
class DealerPort(Protocol):
    device_id: str
    physical_motion: bool

    def open(self) -> None: ...
    def health(self) -> DealerHealth: ...
    def execute(
        self,
        command: DealerCommand,
        observed_at_ns: int | None = None,
    ) -> DealerAck: ...
    def confirm_navigation_target(self, acknowledgement: NavigationAck) -> None: ...
    def close(self) -> None: ...
```

`confirm_navigation_target()` 表示导航已经到达并对准目标。发牌 adapter 必须收到
匹配的导航确认后，才允许对该目标执行发牌动作。

## 2. 导航命令

状态机发送 `NavigationCommand`：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `command_id` | `str` | 命令唯一标识；用于去重和 ACK 关联 |
| `session_id` | `str` | 当前游戏 Session |
| `hand_id` | `str` | 当前牌局 |
| `expected_state_version` | `int` | 命令发出时的游戏状态版本 |
| `expected_pose_version` | `int` | 命令发出时的小车姿态版本 |
| `issued_at_ns` | `int` | 单调时钟命令时间 |
| `action` | `NavigationAction` | 语义动作 |
| `start_pose` | `RobotPoseNode` | 状态机认为的小车起始姿态 |
| `target_pose` | `RobotPoseNode` | 明确目标姿态；允许目标定位时为 `unknown` |
| `target_slot` | `DealerTargetSlot \| None` | 目标玩家或公共牌区域 |
| `timeout_ms` | `int` | 整条命令 ACK 超时 |
| `inter_motion_delay_ms` | `int` | 连续移动最小间隔，当前固定为 `2500` |

当前状态机主要发送：

```text
action = move_and_align_to_target
target_slot = seat_a / seat_b / seat_c / seat_d / board_*
```

`FOLLOW_LINE_*`、`TURN_*`、`RETURN_TO_LINE` 和 `STOP` 已定义为语义动作，但当前
游戏循环尚未把完整桌面路径拆成这些独立命令。真实 adapter 可以在
`move_and_align_to_target` 内部调用循线、转向、面部居中和回归动作。

## 3. 导航 ACK

小车完成或拒绝命令后返回 `NavigationAck`：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `command_id` | `str` | 必须等于原命令 |
| `session_id` | `str` | 必须等于原命令 |
| `hand_id` | `str` | 必须等于原命令 |
| `expected_state_version` | `int` | 必须等于原命令 |
| `action` | `NavigationAction` | 必须等于原命令 |
| `target_slot` | `DealerTargetSlot \| None` | 必须等于原命令目标 |
| `status` | `NavigationAckStatus` | `succeeded/failed/rejected/timed_out` |
| `observed_at_ns` | `int` | 单调时钟完成时间 |
| `actual_pose` | `RobotPoseNode` | 完成后的真实姿态 |
| `pose_version` | `int` | 新姿态版本，成功时必须递增 |
| `pose_confidence` | `float` | 姿态置信度，范围 `[0, 1]` |
| `line_locked` | `bool` | 是否已重新锁定白线 |
| `endpoint_confirmed` | `bool` | 是否确认位于 init/end 等端点 |
| `target_aligned` | `bool` | 是否已经对准目标 |
| `stable_frames` | `int` | 姿态或人脸居中连续稳定帧数 |
| `face_center_error_px` | `float \| None` | 玩家人脸与画面中心的像素误差 |
| `error_code` | `NavigationErrorCode \| None` | 失败原因枚举 |
| `reason` | `str \| None` | 失败的可读说明 |

成功 ACK 必须满足：

1. 所有关联字段与命令一致。
2. `status == succeeded`。
3. `actual_pose != unknown`。
4. `pose_version > expected_pose_version`。
5. 目标导航必须有 `target_slot` 且 `target_aligned == True`。
6. `error_code` 和 `reason` 必须为空。

失败 ACK 必须提供 `error_code` 和非空 `reason`。失败、超时、版本不一致、目标
不一致或未知姿态都会进入 `PAUSED_RECOVERY`，不会推进游戏。

## 4. 2.5 秒连续移动规则

### 4.1 状态机侧

对于 `physical_motion == True` 的导航 adapter：

```text
上一条导航成功 ACK 的 observed_at_ns
    + 2500 ms
    = 下一条物理导航最早允许启动的时间
```

在冷却期内：

- 不创建新的 `NavigationCommand`。
- 不调用 `NavigationPort.execute()`。
- 不启动新命令的 ACK 超时计时。
- 状态机仍停留在 `WAITING_TARGET_ACK`。
- 同步运行器每 50 ms 检查一次，不进行 2.5 秒整段阻塞。
- 手机界面和独立安全控制线程可以继续运行。

冷却开始时写入事件：

```text
navigation_inter_motion_delay_started
```

事件包含：

```json
{
  "command_id": "上一条成功命令",
  "inter_motion_delay_ms": 2500,
  "next_motion_not_before_ns": 123456789
}
```

`SimulatedNavigationAdapter` 的 `physical_motion == False`，因此模拟器、单元测试和
recorded replay 不进行真实等待。

### 4.2 树莓派侧

如果一条 `move_and_align_to_target` 内部包含多个电机动作，例如：

```text
循线到 end -> 停止 -> 左转 -> 人脸居中
```

或者：

```text
第一次右转 -> 第二次右转
```

树莓派 adapter 也必须读取 `inter_motion_delay_ms`，并在内部两个连续电机动作
之间等待至少 2500 ms。状态机侧只能保证两条独立导航命令之间的间隔，无法看到
adapter 内部没有上报的电机动作。

等待必须使用单调时钟，不能使用系统日期时间。急停、看门狗、连接断开和电流保护
必须能够打断等待。

## 5. 状态节点与输入类型

| 状态节点 | 接收类型 | 功能 |
| --- | --- | --- |
| `WAITING_REGISTRATION_CONTROL` | `ControlObservation` | 选择注册角色、开始采集、清空或开始 Session |
| `WAITING_FACE_ENROLLMENT` | `FaceEnrollmentObservation` 或 `ControlObservation` | 完成人脸采集或取消 |
| `WAITING_SESSION_CONTROL` | `ControlObservation` | 开始下一局、清桌、账本操作或结束 Session |
| `WAITING_TARGET_ACK` | `NavigationAck`；兼容节点也接受 `DealerAck` | 等待小车到达并对准玩家/牌槽 |
| `WAITING_VISUAL_SETTLE` | `bool` | 等待玩家视野稳定 |
| `WAITING_FACE_IDENTITY` | `FaceIdentityObservation` | 验证当前已选座位玩家身份 |
| `WAITING_PLAYER_ACTION` | `PlayerActionObservation` | 接收 check/bet/call/raise/fold 候选 |
| `WAITING_CHIP_OBSERVATION` | `ChipObservation` | 可选的 bet/raise 金额核验 |
| `WAITING_DISPENSE_ACK` | `DealerAck` | 等待一张牌实际发出 |
| `WAITING_BOARD_REVEAL` | `CardObservation` | 等待指定公共牌槽确认 |
| `WAITING_POST_BOARD_DELAY` | 无 | 公共牌全部识别后原地等待 1 秒 |
| `WAITING_SHOWDOWN_CARDS` | `CardObservation` | 等待当前玩家两张手牌确认 |
| `WAITING_CARD_OBSERVATION` | `CardObservation` | 通用指定牌槽确认 |
| `WAITING_OPERATOR_CONTROL` | `ControlObservation` | 故障后的人工恢复、重试或作废 |
| `GAME_INTERNAL` | 无 | 状态机内部确定性推进 |
| `COMPLETE` | 无 | 一局或 Session 已完成 |
| `VOIDED` | 无 | 本局已作废 |

## 6. 完整调用顺序

### 6.1 人脸注册

对 Button、Small Blind、Big Blind、Under the Gun 分别执行：

```text
WAITING_REGISTRATION_CONTROL
    -> ControlObservation(CONFIRM)
WAITING_FACE_ENROLLMENT
    -> FaceEnrollmentObservation(CONFIRMED)
WAITING_REGISTRATION_CONTROL
    -> 选择下一个角色
```

四人完成后：

```text
WAITING_REGISTRATION_CONTROL
    -> ControlObservation(START)
    -> 冻结 FrozenSessionRoster
```

注册接口只保存参与者编号、座位和样本数量，不在日志或磁盘中保存人脸图片及
embedding。传入 `navigation_port` 后，`RegistrationNavigationCoordinator`
会在每个录入窗口前等待相应导航 ACK，并在四人完成后回归 `I_E`。

### 6.2 底牌发牌

每一个发牌目标执行：

```text
WAITING_TARGET_ACK
    -> NavigationCommand
    -> 等待 NavigationAck(SUCCEEDED)
    -> DealerPort.confirm_navigation_target()
WAITING_DISPENSE_ACK
    -> DealerCommand(DISPENSE_ONE)
    -> 等待 DealerAck(SUCCEEDED)
    -> 对应底牌槽直接记为 PRESENT_FACE_DOWN
    -> 同一玩家第二张未发时继续 DISPENSE_ONE
    -> 两张完成后进入下一个玩家目标
```

当前流程采用单张、无 burn card 发牌，因此每张牌都需要独立终态 ACK。任何一张
命令未完成都不能推进发牌游标。按用户本轮决定，cocino_car 使用显式的
`arduino_command_ack_only` 完成依据；日志中的 deck、exit pulse、interlock 和
E-stop 仍保持 `null`，不会伪装成实体传感器验证。机构侧由用户确认每次调用能够
保证只出一张牌，因此状态机采用“一次 `dispense_one` 对应一个牌槽”的固定映射：
Flop 连续调用三次，Turn/River 各调用一次。底牌默认背面在位，不额外进入卡牌视觉
识别节点。状态机按
Button→Small Blind→Big Blind→UTG 的小车路线访问一次，每名玩家连续取得两次
独立的单张发牌 ACK。

### 6.3 Preflop 玩家操作

当前玩家的每次操作：

```text
WAITING_TARGET_ACK
    -> 等待 2.5 秒移动间隔
    -> NavigationCommand(move_and_align_to_target)
    -> NavigationAck(SUCCEEDED)
WAITING_VISUAL_SETTLE
    -> 视野稳定确认
WAITING_FACE_IDENTITY
    -> FaceIdentityObservation(MATCHED)
WAITING_PLAYER_ACTION
    -> PlayerActionObservation
```

如果操作为 bet/raise 且启用了筹码核验：

```text
WAITING_CHIP_OBSERVATION
    -> ChipObservation(CONFIRMED)
    -> 金额和动作一致后原子提交动作与账本
```

如果操作是 check/call/fold，或者筹码核验开关关闭，则不进入筹码等待节点。

动作提交后：

```text
更新数字账本和 state_version
    -> 状态机选择下一 acting_seat
    -> 回到 WAITING_TARGET_ACK
```

只有合法动作成功提交后才会更换当前玩家。

### 6.4 Flop

Preflop 下注闭合后，状态机进入公共牌发牌：

```text
WAITING_TARGET_ACK
    -> 启动 follow_line_to_end
    -> 持续巡线 1 秒
    -> 调用 stop 停在公共牌发牌位置
    -> NavigationAck(SUCCEEDED)
WAITING_DISPENSE_ACK
    -> 第一张 DealerCommand(DISPENSE_ONE)
    -> 第一张 DealerAck(SUCCEEDED)
    -> 第二张 DealerCommand(DISPENSE_ONE)
    -> 第二张 DealerAck(SUCCEEDED)
    -> 第三张 DealerCommand(DISPENSE_ONE)
    -> 第三张 DealerAck(SUCCEEDED)
WAITING_BOARD_REVEAL
    -> 打开同一个三牌视觉窗口
    -> 三个指定 VisionSlot 分别提交 CardObservation(CONFIRMED)
WAITING_POST_BOARD_DELAY
    -> 最后一张牌确认后原地等待 1 秒
    -> 等待结束后才允许重新启动巡线
```

Flop 只导航到公共牌区域一次，三次单张发牌 ACK 之间不重复导航，也不插入视觉
确认。第三次发牌 ACK 成功后，小车必须停在公共牌位置原地等待；等待期间不得回线、
转向玩家或生成新的导航命令。玩家把三张牌展示给摄像头，三个指定牌槽全部确认后，
状态机还会进入 `WAITING_POST_BOARD_DELAY` 原地等待 1 秒。等待完成后才允许离开
公共牌位置并进入新的下注轮：

```text
WAITING_TARGET_ACK
    -> WAITING_VISUAL_SETTLE
    -> WAITING_FACE_IDENTITY
    -> WAITING_PLAYER_ACTION
    -> 可选 WAITING_CHIP_OBSERVATION
```

Turn 和 River 使用同样门控顺序，但各只有一次单张发牌 ACK 和一个单牌视觉窗口。
发牌 ACK 后同样必须原地等待；相应公共牌识别成功后再等待 1 秒，才能继续移动。

下一次移动必须同时满足两个条件：

```text
公共牌所需 VisionSlot 已全部 CONFIRMED
并且
识别完成后的 1000 ms 原地等待已经结束
并且
距离上一条成功导航 ACK 已达到 2500 ms
```

因此实际放行时间不得早于“最后一张公共牌确认 + 1000 ms”，同时保留已有的
导航间隔门控。

### 6.5 Showdown

对每一位未弃牌玩家：

```text
WAITING_TARGET_ACK
    -> NavigationAck(SUCCEEDED)
WAITING_SHOWDOWN_CARDS
    -> 第一张手牌 CardObservation
    -> 第二张手牌 CardObservation
    -> 下一位存活玩家
```

所有手牌确认后，由确定性牌型算法计算赢家并结算数字底池。识别模型不能输出赢家。

## 7. Observation 约束

所有 Observation 至少应带有：

- 唯一 `observation_id`。
- 对应 `hand_id` 或 `session_id`。
- `expected_state_version` 或 `expected_roster_version`。
- 单调时钟 `observed_at_ns`。
- 当前目标玩家或指定牌槽。
- 置信度、稳定帧和模型/标定版本。

以下输入不会改变状态：

- 旧 `state_version`。
- 不是当前目标座位。
- 低置信度、unknown、遮挡或不稳定。
- 同时出现冲突动作。
- 重复命令或重复观察。
- 与当前合法动作集合冲突。

### 7.1 筹码观察

`ChipObservation` 的 `chip_counts` 必须能够精确求和为 `total_units`。当前数字账本
仍是余额权威，筹码识别只是 bet/raise 的可选证据。

### 7.2 卡牌观察

卡牌必须写入状态机指定的 `VisionSlot`。未知牌不能猜测为某张牌，同一局内出现
重复牌身份属于硬错误。

### 7.3 人脸观察

人脸只验证状态机已经选定的座位，不允许根据识别到的人脸切换 `acting_seat`。

## 8. 姿态节点

| 枚举 | 含义 |
| --- | --- |
| `i_e` | 位于 init，沿白线朝向 end |
| `i_w` | 位于 init，沿白线朝向 init/反向 |
| `i_button` | init 端对准 Button |
| `i_utg` | init 端对准 Under the Gun |
| `b_e` | 位于公共牌区域，朝向 end |
| `e_e` | 位于 end，沿原方向朝外 |
| `e_w` | 位于 end，沿白线朝向 init |
| `e_sb` | end 端对准 Small Blind |
| `e_bb` | end 端对准 Big Blind |
| `unknown` | 姿态未知；禁止继续导航 |

每次成功移动后 `pose_version` 必须增加。状态机不会接受从 `unknown` 开始的导航。

## 9. 错误与恢复

导航错误码包括：

- `line_lost`
- `endpoint_not_found`
- `board_marker_not_found`
- `target_not_found`
- `target_mismatch`
- `alignment_timeout`
- `pose_unknown`
- `interlock_open`
- `emergency_stop`
- `transport_lost`
- `protocol_error`

发生错误后的顺序：

```text
NavigationAck(FAILED/REJECTED/TIMED_OUT)
    -> PAUSED_RECOVERY
    -> WAITING_OPERATOR_CONTROL
    -> 操作员确认物理状态
    -> 重试、恢复或作废
```

未收到 ACK 时不能假定动作成功，也不能自动推进到识别或发牌节点。

## 10. 树莓派 adapter 实现清单

真实小车接入前至少完成：

1. 实现 `NavigationPort` 或与其一一对应的网络协议。
2. 对 `command_id` 做幂等处理；重复命令返回原 ACK，不能重复移动。
3. 校验 `session_id`、`hand_id`、状态版本、姿态版本和起始姿态。
4. 使用单调时钟记录时长。
5. 执行 `inter_motion_delay_ms == 2500`。
6. 对内部连续电机动作也执行 2500 ms 间隔。
7. 人脸居中后返回 `target_aligned`、稳定帧和像素误差。
8. 循线结束后返回 `line_locked` 和端点确认。
9. 任何失败都返回明确错误码，不返回伪成功。
10. 支持急停、看门狗、超时、堵转/卡住检测和人工恢复。
11. 先通过 simulator 和 recorded replay，再进行有人值守的低速空载测试。

## 11. 当前实现边界

已经实现：

- 机器人输入节点查询。
- 导航命令和 ACK 类型校验。
- 状态版本、目标和姿态版本关联。
- 模拟导航 adapter。
- 导航成功后才允许发牌或检测。
- 两条物理导航命令之间 2500 ms 冷却。
- 人脸、动作、筹码、卡牌和发牌 ACK 接口。
- 失败后停止推进并进入恢复状态。

仍待真实硬件闭环：

- 公共牌区标记停车 API。
- 可关联 `request_id` 的动作完成、规范姿态和姿态版本证据。
- 具有实体出牌传感器证据的真实 `DealerPort`。
- 桌面尺寸、端点、公共牌标记和相机安装标定。
- 有人值守的低速实机验证。

相关代码：

- `src/poker_dealer/domain/robot.py`
- `src/poker_dealer/runtime/robot_interfaces.py`
- `src/poker_dealer/runtime/registration.py`
- `src/poker_dealer/runtime/hand_runtime.py`
- `src/poker_dealer/runtime/hand_loop.py`
- `src/poker_dealer/runtime/sequential_part_a.py`
- `src/poker_dealer/runtime/sequential_part_b.py`
- `src/poker_dealer/robotics/navigation/port.py`
- `src/poker_dealer/robotics/navigation/adapters.py`
- `src/poker_dealer/robotics/navigation/cocino_car.py`
- `src/poker_dealer/robotics/navigation/table_route.py`
- `src/poker_dealer/robotics/navigation/timing.py`
- `src/poker_dealer/robotics/dealer/port.py`

## 12. cocino_car 正式 API v1.0 对接

`C:\Users\ASUS\Desktop\cocino_car` 仅作为只读 API 来源。本项目不会导入
该仓库代码，也不会直接操作 GPIO、串口或轮速。正式 HTTP 边界为：

```text
GET  /api/robotics/v1/capabilities
GET  /api/robotics/v1/status
POST /api/robotics/v1/actions
GET  /api/robotics/v1/requests/{request_id}
```

实现位置：

| 文件 | 作用 |
| --- | --- |
| `table_route.py` | I 形桌面的规范姿态、有向边和合法路径规划 |
| `cocino_car.py` | API v1.0 JSON 客户端、状态轮询和 `NavigationAck` 适配 |
| `robotics/dealer/cocino_car.py` | `dispense_one` 到 `DealerPort` 的安全门；证据不足时拒绝 |
| `runtime/registration_route.py` | Button→SB→BB→UTG 录入路线及结束归位 |
| `runtime/hand_runtime.py` | 根据当前 Button 把 Seat 解析为物理目标姿态 |
| `runtime/hand_loop.py` | 仅在成功导航 ACK 后进入身份/操作/牌面等待节点 |

### 12.1 状态机目标到小车 API 的分配

| 状态机目标 | 路径规划原语 | cocino_car action | 完成证据 |
| --- | --- | --- | --- |
| `init ↔ end` | 沿白线到端点 | `follow_line_to_end` | `route.state=END_REACHED` |
| 端点转向玩家 | 动态人脸对中 | `face_turn_start` + `face_turn_heartbeat` + `face_turn_stop` | 上位机连续人脸居中，并且 `FACE_CENTERED_STOP` |
| 玩家朝向回白线 | 白线重新居中 | `line_recenter_start` | `LINE_TURN_CENTERED` |
| SB→BB | 两次右转 90° | 两次 `preset_turn(RIGHT,90)` | 每次 `MANUAL_RED_ALIGNED` 或 `MANUAL_COMPLETE` |
| UTG→Button | 两次右转 90° | 两次 `preset_turn(RIGHT,90)` | 同上 |
| 任意故障停车 | 停止 | `stop` | 最佳努力；游戏进入恢复而不猜测 |
| init→公共牌区 | 沿线到公共牌标记 | **API v1.0 尚无对应 action** | 当前适配器返回 `BOARD_MARKER_NOT_FOUND` |

API 的 `follow_line_to_end` 名称没有方向参数。路径规划器只会在
`I_E→E_E` 或 `E_W→I_W` 这两种已知朝向调用它；方向由小车当前实体朝向
决定。

### 12.2 人脸对中输入

小车端不执行人脸检测。`CocinoCarNavigationAdapter` 接收：

```python
class FaceCenterProbe(Protocol):
    def observe_face_center(self) -> FaceCenterSample: ...
```

`FaceCenterSample` 字段为 `detected: bool`、`centered: bool`、
`stable_frames: int` 和 `center_error_px: float | None`。

`runtime/live_perception.py` 中的 `LiveFaceCenterProbe` 使用现有
`OpenCvFaceIdentityAdapter` 和玩家摄像头。只有一个人脸连续三帧进入画面
中心容差后才发送 `face_turn_stop`。未检测、多人、未稳定或超时都不能生成
成功导航 ACK。

### 12.3 人脸录入等待顺序

当 `LivePerceptionSession.acquire_roster(..., navigation_port=...)` 收到导航
接口时，每个玩家的录入顺序为：

```text
请求导航目标
→ 等待 NavigationAck(SUCCEEDED)
→ 等待单人脸
→ 等待操作员 CONFIRM
→ 采集并确认人脸
→ 切换下一个角色
```

四人依次为 Button、Small Blind、Big Blind、UTG。UTG 录入完成并收到
START 后，先等待小车从 `I_UTG` 回归 `I_E`，然后才返回冻结 roster。

### 12.4 牌局中的目标姿态

状态机按本手 `button` 动态解析固定 Seat：

- 行动询问和 Showdown：玩家正面姿态
  `I_BUTTON/E_SB/E_BB/I_UTG`。
- 底牌：沿线发牌姿态 `I_E/E_E/E_W/I_W`，顺序为
  Button→SB→BB→UTG，每名玩家连续发两张。
- Flop/Turn/River：`B_E`。Flop 连续三次发牌 ACK 后原地等待三张牌确认；
  Turn/River 发一张后同样原地等待确认。确认前不会继续前往 end。

### 12.5 2500 ms 连续运动间隔

有两层保护：

1. `CocinoCarNavigationAdapter` 在一个路径内的相邻原子移动之间等待
   `NavigationCommand.inter_motion_delay_ms`，默认 `2500`。
2. `HandRuntimeLoop` 在两个成功的物理 `NavigationCommand` 之间再次执行
   非阻塞冷却门。

延时从上一动作被判断完成后开始。模拟适配器不等待。

### 12.6 当前不能宣称实机闭环的地方

真实适配器已实现 API 调用与 fail-closed 逻辑，但默认
`robot_hardware` profile 仍保持禁用：

1. API v1.0 没有公共牌区标记停车动作。
2. API 状态没有 `request_id` 对应的最终运动完成记录，也没有规范姿态和
   `pose_version`；适配器只能在本进程内维护操作员确认的逻辑姿态。
3. `dispense_one` 只有 Arduino 命令接收证据，没有出牌口传感器证据，
   不能满足 `DealerAck(SUCCEEDED)` 的安全语义。
4. 小车内部人脸脉冲冷却仍约为 2 秒；上位机不能证明每个内部脉冲之间都
   达到 2.5 秒。

这些缺口不会被模拟成功或跳过；遇到缺口时状态机会暂停并等待人工恢复。
`CocinoCarDealerAdapter` 已把 `DISPENSE_ONE` 分配到该 API 边界，但在当前
`physical_card_exit_verified=false` 时会在发送电机动作前返回
`DealerAck(REJECTED, protocol_error)`。
