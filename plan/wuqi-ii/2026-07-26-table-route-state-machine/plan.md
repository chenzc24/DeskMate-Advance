# 四人桌面小车路线与游戏状态机计划

状态：`接口合同与可选状态机节点已实现；实机标定仍未开始`

日期：`2026-07-26`

负责人：`wuqi-ii`

关联计划：

- `plan/wuqi-ii/2026-07-25-game-flow-revision/plan.md`

## 1. 目标

本计划把四人德州扑克游戏状态机与小车在目标桌面上的实际路线合并为一个
可实现、可回放的任务规划方案。

小车在本阶段负责：

1. 沿白线在 `init` 和 `end` 两端之间定向巡航。
2. 在端点执行受约束的左转、右转和两次右转。
3. 通过人脸位于画面中心动态修正玩家朝向。
4. 依次完成人脸录入。
5. 给四名玩家发底牌，并在中心公共牌区域发 Flop、Turn 和 River。
6. 移动到状态机指定的当前行动玩家前，记录玩家行动。
7. 在需要时识别玩家下注区筹码。
8. 识别公共牌和 Showdown 中未弃牌玩家的底牌。
9. 订阅确定性游戏结果并宣布赢家与赢取数额。

小车不负责：

- 自动下大小盲实体筹码。
- 翻开公共牌；玩家负责展示或翻开。
- 移动、铲取、整理、退还或分配实体筹码。
- 自动收牌、洗牌或装牌。
- 通过视觉模型决定轮到谁、动作是否合法、牌型或赢家。

## 2. 场地拓扑

本计划依据用户提供的桌面照片建立以下拓扑，不提交临时照片本身：

```text
                    Button                         Small Blind
                       ↑                               ↑
                       │                               │
                       │                               │
            init  =====I=========== BOARD ============E=====  end
                       │                               │
                       │                               │
                       ↓                               ↓
                     UTG                           Big Blind
```

定义：

- `I`：白线与左侧红线的交点，即 `init`。
- `E`：白线与右侧红线的交点，即 `end`。
- `BOARD`：`init -> end` 方向白线附近的公共牌发牌/观察区域。
- `Button`：位于 `init` 端白线上方。
- `UTG`：位于 `init` 端白线下方。
- `Small Blind`：位于 `end` 端白线上方。
- `Big Blind`：位于 `end` 端白线下方。

“上方”和“下方”只作为桌面拓扑方向，不依赖相机图像的绝对上下方向。

## 3. 小车姿态模型

### 3.1 位置

```text
INIT
BOARD
END
```

### 3.2 朝向

```text
TO_END       沿白线从 init 指向 end
TO_INIT      沿白线从 end 指向 init
TO_BUTTON    init 端面向 Button
TO_UTG       init 端面向 UTG
TO_SB        end 端面向 Small Blind
TO_BB        end 端面向 Big Blind
```

### 3.3 规范姿态节点

路径规划器只允许在以下规范姿态之间切换：

| 节点 | 位置 | 朝向 |
| --- | --- | --- |
| `I_E` | `INIT` | `TO_END` |
| `I_W` | `INIT` | `TO_INIT` |
| `I_BUTTON` | `INIT` | `TO_BUTTON` |
| `I_UTG` | `INIT` | `TO_UTG` |
| `B_E` | `BOARD` | `TO_END` |
| `E_E` | `END` | `TO_END` |
| `E_W` | `END` | `TO_INIT` |
| `E_SB` | `END` | `TO_SB` |
| `E_BB` | `END` | `TO_BB` |

每条动作 ACK 必须报告动作后的规范姿态。姿态不确定时不能继续组合下一动作。

## 4. 唯一允许的运动原语

路径规划器不得直接输出左右轮速度、转动角度、GPIO 或串口字节，只能输出
以下语义动作：

### 4.1 沿线移动

```text
FOLLOW_LINE_TO_END
FOLLOW_LINE_TO_INIT
FOLLOW_LINE_TO_BOARD
FOLLOW_LINE_BOARD_TO_END
```

约束：

- `FOLLOW_LINE_TO_END` 只从 `I_E` 出发，到达 `E_E`。
- `FOLLOW_LINE_TO_INIT` 只从 `E_W` 出发，到达 `I_W`。
- 公共牌阶段允许从 `I_E` 到 `B_E`，确认公共牌后再从 `B_E` 到 `E_E`。
- 到端点必须由端点标记、里程/时间约束和线状态共同确认，不能仅凭固定延时。

### 4.2 面向玩家

```text
TURN_LEFT_TO_PLAYER
TURN_RIGHT_TO_PLAYER
```

单次转向以约 `90°` 为初始动作，以目标玩家人脸中心与画面中心的偏差进行
低速动态修正。完成条件不是“已经转了固定角度”，而是：

```text
目标人脸与预期座位一致
AND 人脸中心误差进入阈值
AND 连续稳定帧达到阈值
```

允许从白线姿态直接向左或向右约 `90°` 对准同端点玩家：

| 白线姿态 | 左转目标 | 右转目标 |
| --- | --- | --- |
| `I_E` | `I_BUTTON` | `I_UTG` |
| `I_W` | `I_UTG` | `I_BUTTON` |
| `E_E` | `E_SB` | `E_BB` |
| `E_W` | `E_BB` | `E_SB` |

这使规划器在同端点只有一名玩家需要服务时可以直接对准该玩家，但不能改变
游戏状态机给出的行动顺序。

### 4.3 同端点 180° 转向

仅允许以下有向转换，并且必须由两次右转组成：

```text
E_SB  --RIGHT_90 + RIGHT_90--> E_BB
I_UTG --RIGHT_90 + RIGHT_90--> I_BUTTON
```

不能生成反向的 `E_BB -> E_SB` 或 `I_BUTTON -> I_UTG` 快捷动作。若任务顺序
需要反向访问，规划器必须回归白线并按规范巡航环重新到达。

### 4.4 回归白线

`RETURN_TO_LINE` 只能在小车已经对准一个已知玩家时调用：

| 当前玩家姿态 | 固定回归方向 | 回归后姿态 |
| --- | --- | --- |
| `I_BUTTON` | 右转 | `I_E` |
| `I_UTG` | 左转 | `I_E` |
| `E_SB` | 左转 | `E_W` |
| `E_BB` | 右转 | `E_W` |

如果当前画面没有稳定绑定到预期玩家，禁止调用回归。应先重新定位该玩家或
进入人工恢复，避免从未知朝向盲转。

## 5. 路线图

```text
I_E --左转/人脸居中--> I_BUTTON --右转回归--> I_E
I_E --右转/人脸居中--> I_UTG --左转回归--> I_E

I_E --沿线到 end--> E_E
E_E --左转/人脸居中--> E_SB
E_E --右转/人脸居中--> E_BB
E_SB --两次右转/人脸居中--> E_BB
E_SB --左转回归--> E_W
E_BB --右转回归--> E_W

E_W --沿线到 init--> I_W
I_W --左转/人脸居中--> I_UTG
I_W --右转/人脸居中--> I_BUTTON
I_UTG --两次右转/人脸居中--> I_BUTTON
I_UTG --左转回归--> I_E
I_BUTTON --右转回归--> I_E

I_E --沿线到 BOARD--> B_E --沿线到 end--> E_E
```

## 6. 人脸录入专用流程

初始条件：

```text
pose = I_E
robot faces end
all four seats unregistered
```

顺序：

1. `I_E -> I_BUTTON`：左转并动态居中 Button 人脸。
2. 采集 Button 人脸，稳定完成后冻结 `Button -> participant_id`。
3. `I_BUTTON -> I_E`：Button 默认右转回归。
4. `I_E -> E_E`：沿白线前进到 end。
5. `E_E -> E_SB`：左转并动态居中 Small Blind 人脸。
6. 采集 Small Blind 人脸。
7. `E_SB -> E_BB`：连续两次右转并动态居中 Big Blind 人脸。
8. 采集 Big Blind 人脸。
9. `E_BB -> E_W`：Big Blind 默认右转回归。
10. `E_W -> I_W`：沿白线返回 init。
11. `I_W -> I_UTG`：左转并动态居中 UTG 人脸。
12. 采集 UTG 人脸。
13. `I_UTG -> I_E`：UTG 默认左转回归。
14. 四名玩家全部收录后冻结 SessionRoster。

输出不变量：

```text
pose = I_E
registered = {Button, SB, BB, UTG}
```

任一玩家采集失败时保持当前玩家姿态并重试，不提前移动到下一人。

## 7. 盲注和底牌发牌专用流程

### 7.1 自动盲注

- 状态机自动登记 Small Blind `10` 和 Big Blind `20`。
- 小车不识别盲注实体筹码。
- 盲注账本提交成功后才能进入发底牌流程。

### 7.2 底牌发牌

初始条件：

```text
pose = I_E
发牌口位于小车左侧
```

用户冻结的目标流程是“每个座位一次发两张”，不是当前代码中的两圈逐张发牌。
实现前必须显式迁移发牌步序和测试，不能沿用当前 `hole_deal_targets()`。

步骤：

1. 在 `I_E` 原地向左侧 Button 连续发两张牌。
2. 每张牌都必须收到独立的单张出牌 ACK。
3. `I_E -> E_E`，沿白线到 end。
4. 在 `E_E` 原地向左侧 Small Blind 连续发两张牌。
5. 每张牌都必须收到独立 ACK。
6. 从 `E_E` 右转约 `90°` 对准 Big Blind，但不识别人脸、不发牌。
7. 执行 Big Blind 默认右转回归，得到 `E_W`。
8. 在 `E_W` 原地从小车左侧向 Big Blind 连续发两张牌。
9. `E_W -> I_W`，沿白线到 init。
10. 在 `I_W` 原地从小车左侧向 UTG 连续发两张牌。
11. 发牌结束，保持 `I_W`，暂时不回归。

输出不变量：

```text
pose = I_W
每个座位恰好两个已确认发牌 ACK
不识别底牌牌面
```

发牌失败、双张、无牌、卡牌、超时或姿态不一致时进入恢复，禁止继续发下一张。

## 8. Preflop 专用行动路线

Preflop 的首次行动顺序：

```text
UTG -> Button -> Small Blind -> Big Blind
```

初始条件是底牌发完后的 `I_W`：

1. `I_W -> I_UTG`：左转动态居中 UTG。
2. 完成 UTG 身份核验、动作询问和必要的筹码识别。
3. 若牌局尚未结束，`I_UTG -> I_BUTTON`：连续两次右转。
4. 完成 Button 操作。
5. `I_BUTTON -> I_E`：Button 默认右转回归。
6. `I_E -> E_E`：前往 end。
7. `E_E -> E_SB`：左转询问 Small Blind。
8. `E_SB -> E_BB`：连续两次右转询问 Big Blind。
9. 若下注轮仍未闭合，`E_BB -> E_W -> I_W -> I_UTG`，从 UTG 继续下一圈。
10. 若下注轮闭合，将姿态规范化到 `I_E`，为 Flop 做准备。

只有确定性游戏状态机可以决定：

- 当前 `acting_seat`；
- 某玩家是否已经弃牌；
- 某玩家是否仍需对最新 bet/raise 响应；
- 下注轮是否闭合；
- 是否已经只剩一名未弃牌玩家。

路径规划器不得因为“已经经过该玩家”自行跳过状态机仍要求的行动。

## 9. 正常街道巡航路线

Flop、Turn、River 下注使用同一有向巡航环：

```text
I_E
  -> BOARD 发公共牌并识别
  -> E_E
  -> E_SB       询问 Small Blind
  -> E_BB       询问 Big Blind
  -> E_W
  -> I_W
  -> I_UTG      询问 UTG
  -> I_BUTTON   询问 Button
  -> I_E
```

该顺序等价于 Button 左侧第一名存活玩家开始的顺时针行动顺序。在全员存活时：

```text
Small Blind -> Big Blind -> UTG -> Button
```

如果下注发生 raise 且需要继续响应，重复上述巡航环，直到游戏引擎宣布本街
下注闭合。

## 10. Flop 流程

初始条件：

```text
pose = I_E
preflop betting closed
```

步骤：

1. 调用 `FOLLOW_LINE_TO_BOARD`，从 init 向 end 前进。
2. 到达公共牌发牌触发区后停车或进入受控低速发牌窗口。
3. 连续发三张牌，每张牌必须有独立 ACK。
4. 进入 `WAITING_FLOP_REVEAL`，等待玩家将三张公共牌展示给小车。
5. 卡牌识别分别确认 `FLOP_1/FLOP_2/FLOP_3`。
6. 三张牌全部稳定、无重复且与当前 hand/state version 对应后，才能继续前进。
7. `B_E -> E_E`，前进到 end。
8. 按正常巡航顺序从第一名仍需行动的存活玩家开始询问。
9. 下注轮闭合后把小车规范化到 `I_E`。

不能用固定等待时间代替玩家展示和卡牌确认。

## 11. Turn 和 River 流程

Turn 与 River 共用同一个参数化流程：

```text
RUN_BOARD_STREET(street, card_count=1)
```

步骤：

1. 从 `I_E` 向 BOARD 前进。
2. 在公共牌发牌区发一张牌并等待单张 ACK。
3. 分别进入 `WAITING_TURN_REVEAL` 或 `WAITING_RIVER_REVEAL`。
4. 等待玩家展示该公共牌。
5. 卡牌识别稳定确认且通过全局重复牌检查。
6. 继续前进至 `E_E`。
7. 按 `SB -> BB -> UTG -> Button` 的有向巡航环完成下注。
8. 若进入下一街，将姿态规范化到 `I_E`。

## 12. 弃牌后的路径优化

### 12.1 原则

路径优化的输入是游戏状态机给出的：

```text
live_seats
action_required_seats
acting_seat
betting_round_closed
```

优化只能：

- 跳过已弃牌或当前不需要行动玩家的“转向、身份识别、动作询问”任务；
- 在白线上直接经过该端点；
- 在下注轮闭合后提前进入规范化路线。

优化不能：

- 改变游戏状态机给出的行动顺序；
- 生成未定义的自由空间移动；
- 生成 `Button -> UTG` 或 `BB -> SB` 的反向 180°；
- 从未知朝向直接回归；
- 因为某人弃牌而跳过仍需访问的同端点玩家。

### 12.2 端点服务算法

为两个端点定义有向服务序列：

```text
END_SERVICE  = [Small Blind, Big Blind]
INIT_SERVICE = [UTG, Button]
```

当到达端点时：

1. 读取本端点在 `action_required_seats` 中的玩家。
2. 如果两人均需要行动，按完整有向服务序列执行。
3. 如果只有序列第一人需要行动，面向第一人、完成任务，然后按该玩家默认方向回归。
4. 如果只有序列第二人需要行动，从正常白线入口姿态向右转约 `90°` 直接
   面向该玩家，完成任务后按该玩家默认方向回归。
5. 如果两人均不需要行动，保持白线朝向并直接前往下一端点或执行规范化。

因此正常巡航允许：

```text
E_E --右转--> E_BB       仅 BB 需要服务
I_W --右转--> I_BUTTON   仅 Button 需要服务
```

但仍不允许从 `E_BB` 直接反向 180° 到 `E_SB`，也不允许从 `I_BUTTON`
直接反向 180° 到 `I_UTG`。

### 12.3 路线选择

规划器不是连续空间最短路，而是在有限有向图上寻找最小代价合法路径：

```text
route = shortest_path(
    graph=ALLOWED_POSE_GRAPH,
    start=current_pose,
    goal=next_required_service_pose,
    edge_allowed=primitive_is_frozen_and_safe,
    edge_cost=motion_time + turn_penalty + uncertainty_penalty,
)
```

推荐代价：

```text
沿线移动成本          = 预计移动时间
单次 90° 转向成本     = 转向时间 + 人脸居中时间
两次右转成本          = 两个转向成本
未经标定边            = 禁止，不是高成本
姿态不确定            = 无可用边，进入恢复
```

因为图很小，可以使用 Dijkstra；也可以预计算所有规范姿态之间的合法路径表。
不需要神经网络规划，也不需要自由空间 SLAM。

## 13. Showdown

触发条件：

```text
River betting closed
AND live player count >= 2
```

流程：

1. 根据当前规范姿态和所有未弃牌玩家构造 `SHOWDOWN_VISIT_QUEUE`。
2. 队列顺序必须兼容有向巡航环，默认从当前位置之后的首个存活玩家开始。
3. 到达玩家后先执行会话身份核验，确认是预期座位玩家。
4. 请求该玩家展示两张底牌。
5. 两张牌稳定识别、槽位绑定正确且无全局重复后，才弹出该玩家任务。
6. 前往队列中的下一名未弃牌玩家。
7. 所有人底牌确认后，确定性牌型算法计算结果。
8. 小车停止运动并宣布赢家、牌型和赢取数额。

Showdown 队列必须包含所有未弃牌玩家，不固定为两人。

如果 Showdown 前只剩一人未弃牌：

1. 立即取消所有未开始的移动和感知任务。
2. 不识别任何底牌。
3. 保持当前位置安全停车。
4. 由游戏引擎结算并播报赢家与赢取数额。

## 14. 顶层任务状态机

```text
BOOT
-> HOMING_TO_I_E
-> REGISTRATION_BUTTON
-> REGISTRATION_SB
-> REGISTRATION_BB
-> REGISTRATION_UTG
-> SESSION_READY
-> POSTING_BLINDS
-> HOLE_DEAL_BUTTON
-> HOLE_DEAL_SB
-> HOLE_DEAL_BB
-> HOLE_DEAL_UTG
-> PREFLOP_BETTING
-> NORMALIZE_TO_I_E
-> FLOP_DEAL_AND_REVEAL
-> FLOP_BETTING
-> NORMALIZE_TO_I_E
-> TURN_DEAL_AND_REVEAL
-> TURN_BETTING
-> NORMALIZE_TO_I_E
-> RIVER_DEAL_AND_REVEAL
-> RIVER_BETTING
-> SHOWDOWN_CAPTURE
-> SETTLEMENT
-> STOPPED_RESULT
```

任意活动阶段都允许：

```text
ANY_ACTIVE_STATE
-> PAUSED_RECOVERY
-> RESUME_SAME_TASK / VOID_HAND / END_SESSION
```

任意下注阶段都允许：

```text
BETTING
-> only_one_live_player
-> UNCONTESTED_SETTLEMENT
-> STOPPED_RESULT
```

## 15. 规划器输入与输出

### 15.1 输入

```text
GameContext:
    session_id
    hand_id
    state_version
    street
    acting_seat
    live_seats
    action_required_seats
    legal_actions
    betting_round_closed

RobotPose:
    position
    heading
    pose_version
    line_visible
    endpoint_marker
    confidence

TaskEvidence:
    navigation_ack
    rotation_ack
    face_identity_observation
    visual_settle_observation
    player_action_observation
    chip_observation
    dispense_ack
    card_observation
    operator_control
```

### 15.2 输出

```text
SemanticRobotCommand:
    command_id
    expected_game_state_version
    expected_pose_version
    action
    target_pose
    target_seat_or_slot
    timeout_ms
```

允许的 `action`：

```text
FOLLOW_LINE
TURN_LEFT_TO_PLAYER
TURN_RIGHT_TO_PLAYER
RETURN_TO_LINE
DISPENSE_ONE
WAIT_FOR_FACE
WAIT_FOR_ACTION
WAIT_FOR_CHIPS
WAIT_FOR_CARDS
STOP
```

完成 ACK 必须同时匹配：

```text
command_id
expected_game_state_version
expected_pose_version
target_pose
target_seat_or_slot
```

不匹配、过期或重复 ACK 不得推进任务。

## 16. 路径规划伪代码

```python
def plan_next_task(game, robot, task):
    if task.has_pending_command:
        return WAIT

    if not robot.pose_is_known:
        return PAUSE_RECOVERY("robot_pose_unknown")

    if game.only_one_live_player:
        return STOP_AND_SETTLE_UNCONTESTED

    if task.requires_evidence and not task.evidence_confirmed:
        return WAIT_FOR_REQUIRED_EVIDENCE

    if task.phase in BOARD_DEAL_PHASES:
        return plan_board_delivery_and_reveal(game, robot, task)

    if task.phase in BETTING_PHASES:
        seat = game.acting_seat
        if seat is None:
            return plan_normalize_to_init(robot)
        target = service_pose(seat)
        path = shortest_legal_pose_path(robot.pose, target)
        return path.first_command_or(wait_for_player_task(seat))

    if task.phase == SHOWDOWN_CAPTURE:
        seat = first_unconfirmed_live_seat_on_legal_route(game, robot)
        if seat is None:
            return COMPUTE_AND_ANNOUNCE_RESULT
        path = shortest_legal_pose_path(robot.pose, service_pose(seat))
        return path.first_command_or(wait_for_showdown_cards(seat))

    return task_specific_command(game, robot, task)
```

核心约束：

- 游戏引擎决定任务对象，路径规划器只决定如何合法到达。
- 每次只发出一个语义命令。
- 收到匹配 ACK 后才推进游标。
- 感知未知不是“没有玩家/没有牌”，应保持当前任务。
- 折叠优化发生在任务队列生成阶段，不发生在规则判断阶段。

## 17. 必需接口

现有可复用：

- `FrameSource`
- `RegistrationSource`
- `IdentitySource`
- `ActionSource`
- `CardSource`
- `VisualSettleSource`
- `ControlSource`
- `DealerCommand/DealerAck`

需要新增或扩展：

1. `NavigationCommand/NavigationAck`
   - 沿线方向、目标端点/BOARD、线锁定、端点确认和姿态版本。
2. `RobotPoseObservation`
   - 当前位置、朝向、置信度、线状态和端点标志。
3. `ChipObservation/ChipSource`
   - 只在 `bet/raise` 时提供当前下注区金额证据。
4. `DealerCommand/DealerAck` 目标绑定
   - `DISPENSE_ONE` 需要带目标座位/牌槽和预期姿态，不能只有
     `at_target=true`。
5. `BoardRevealObservation`
   - 可直接复用 `CardObservation` 聚合结果，但任务层必须有独立
     `WAITING_*_REVEAL` 门。

## 18. 异常、暂停与恢复

以下情况立即停车并进入 `PAUSED_RECOVERY`：

- 白线连续丢失超过阈值。
- 未确认到达端点或 BOARD。
- 实际朝向与预期规范姿态不一致。
- 人脸不是预期座位玩家或无法稳定居中。
- 在未知玩家朝向下请求回归。
- 导航、旋转或发牌 ACK 超时、失败、重复或目标不匹配。
- 发牌机构无牌、双张或卡牌。
- 公共牌/底牌未知、遮挡、重复或槽位冲突。
- 玩家动作或筹码金额长期不明确。
- 软件状态版本与机器人姿态版本不一致。
- 急停、安全互锁或通信断开。

恢复只能由操作员确认：

- 当前规范姿态；
- 物理牌槽与软件牌槽一致；
- 发牌机构安全；
- 是否重试当前原子任务、作废本局或结束 Session。

不得从恢复状态自动猜测最近的玩家、端点或已经发出的牌数。

## 19. 实现顺序

1. 冻结桌面尺寸、相机安装、白线宽度、端点和 BOARD 触发标志。
2. 建立规范姿态和有向动作图的数据结构。
3. 实现纯软件路径规划器和动作序列模拟器。
4. 为注册、底牌、Preflop、普通街道和 Showdown 编写黄金路线测试。
5. 实现折叠玩家任务裁剪，但保留有向动作约束。
6. 定义 `NavigationCommand/NavigationAck` 和 `RobotPoseObservation`。
7. 将循迹模块包装成导航 adapter，不让 OpenCV 直接推进游戏状态。
8. 接入玩家动态居中转向和回归动作。
9. 迁移“每座位连续发两张”的底牌发牌合同。
10. 接入 BOARD 发牌与手动展示确认。
11. 接入玩家动作、筹码和 Showdown 手牌观察。
12. 先 recorded replay，再空载低速路线测试，最后才允许装牌联调。

## 20. 验收用例

至少验证：

1. 从 `I_E` 完成四人注册并严格回到 `I_E`。
2. 每人连续发两张且恰好收到八个单张 ACK，结束于 `I_W`。
3. Preflop 首圈严格为 `UTG -> Button -> SB -> BB`。
4. Flop/Turn/River 严格从 `I_E` 发牌，确认后到 `E_E` 开始下注。
5. 普通街道全员路线严格为 `SB -> BB -> UTG -> Button`。
6. raise 后正确重复有向巡航环。
7. 任意一个、两个或三个玩家弃牌时，任务被裁剪但不出现禁止转向。
8. 本街提前闭合后能从任意合法姿态规范化到 `I_E`。
9. Showdown 访问所有未弃牌玩家，并等待每人两张牌确认后再移动。
10. Showdown 前只剩一人时原地停车结算。
11. 线丢失、人脸错误、姿态未知、发牌失败和卡牌重复都不能推进。
12. 日志回放生成完全相同的动作序列和最终姿态。

## 21. 文件边界、外部依赖与提交意图

本次拥有的跟踪路径：

- `plan/wuqi-ii/2026-07-26-table-route-state-machine/plan.md`
- `src/poker_dealer/domain/robot.py`
- `src/poker_dealer/domain/__init__.py`
- `src/poker_dealer/runtime/robot_interfaces.py`
- `src/poker_dealer/runtime/ports.py`
- `src/poker_dealer/runtime/registration.py`
- `src/poker_dealer/runtime/sequential_part_a.py`
- `src/poker_dealer/runtime/sequential_part_b.py`
- `src/poker_dealer/runtime/hand_runtime.py`
- `src/poker_dealer/runtime/hand_loop.py`
- `src/poker_dealer/runtime/session_runtime.py`
- `src/poker_dealer/runtime/__init__.py`
- `src/poker_dealer/robotics/navigation/`
- `src/poker_dealer/robotics/dealer/port.py`
- `src/poker_dealer/robotics/dealer/adapters.py`
- `tests/domain/test_robot_interfaces.py`
- `tests/runtime/test_robot_interface_nodes.py`
- `tests/runtime/test_robot_navigation_and_chip_gates.py`
- `tests/runtime/test_hand_loop_replay.py`

外部只读参考：

- `C:/Users/ASUS/AppData/Local/Temp/codex-clipboard-14a529cb-9610-4301-a4f1-15b8a7f35326.jpg`

后续预计修改但本次不修改：

- `docs/plans/POKER_DEALER_MASTER_PLAN.md`
- `docs/contracts/GAME_RULES.md`
- `docs/contracts/CORE_INTERFACES.md`
- `configs/game/core_v1.json`
- `configs/contracts/`
- `src/poker_dealer/domain/`
- `src/poker_dealer/game/`
- `src/poker_dealer/runtime/`
- `src/track_line/`
- 对应测试与模拟器

当前工作区中其他卡牌训练、模型、运行配置、数据处理和巡线修改均为只读，
不得纳入本计划提交。

物理运动状态：

- 本计划只定义路线和算法，不启动或授权实体小车运动。
- 实机阶段必须有操作员、空载、低速、急停、保护、清空路线和恢复说明。
- 协议/mock 测试必须先于真实电机动作。

验证要求：

- 检查 Markdown 结构、术语、姿态和动作图一致性。
- 后续机器可读配置必须通过 JSON/schema 解析。
- 路径规划器必须通过黄金路线、折叠优化、故障注入和确定性回放测试。
- 后续代码修改运行目标测试与实用完整测试。
- 始终运行 `git diff --check` 和 `git status --short --branch`。

提交意图：

- 本目标允许实现类型合同、可选状态机门、模拟导航 adapter 和针对性测试。
- 默认兼容路径必须继续跑完原有完整牌局。
- 用户已于 `2026-07-27` 明确要求把关联状态机整合结果直接提交并推送到
  `main`；提交时只纳入本目标与关联计划拥有的文件。

## 22. 已实现的小车输入节点与类型合同

实现原则：

- 原有 `HandEngine` 仍是唯一规则和 `acting_seat` 权威。
- 小车模型只产生带版本和目标绑定的观察或 ACK，不能直接切换游戏阶段。
- 默认关闭新筹码关口且保留旧 `DealerAck` 导航路径，因此原有整局回放仍可运行。
- 导航接口当前只连接无物理运动的模拟 adapter；本计划未授权真实电机动作。

| 节点 | 接受的 Python 类型 | 作用 |
| --- | --- | --- |
| `WAITING_REGISTRATION_CONTROL` | `ControlObservation` | 选择注册座位、开始采集、冻结名单 |
| `WAITING_FACE_ENROLLMENT` | `FaceEnrollmentObservation` 或 `ControlObservation` | 提交一次人脸采集结果，或取消采集 |
| `WAITING_SESSION_CONTROL` | `ControlObservation` | 局间清桌、补充账本、开始下一局或结束 Session |
| `WAITING_TARGET_ACK` | `NavigationAck`；兼容节点还接受 `DealerAck` | 确认小车已到达并对准状态机指定玩家/牌槽 |
| `WAITING_VISUAL_SETTLE` | `bool` | 确认目标视野已经稳定 |
| `WAITING_FACE_IDENTITY` | `FaceIdentityObservation` | 只核验当前已选座位玩家，不改变 `acting_seat` |
| `WAITING_PLAYER_ACTION` | `PlayerActionObservation` | 提交当前玩家的 check/bet/call/raise/fold 候选 |
| `WAITING_CHIP_OBSERVATION` | `ChipObservation` | 仅在可选关口启用；核验 bet/raise 对应金额 |
| `WAITING_DISPENSE_ACK` | `DealerAck` | 确认单张牌发出成功后推进发牌步骤 |
| `WAITING_BOARD_REVEAL` | `CardObservation` | 确认 Flop/Turn/River 的指定公共牌槽 |
| `WAITING_SHOWDOWN_CARDS` | `CardObservation` | 确认当前未弃牌玩家的指定底牌槽 |
| `WAITING_CARD_OBSERVATION` | `CardObservation` | 通用指定牌槽确认 |
| `WAITING_OPERATOR_CONTROL` | `ControlObservation` | 暂停恢复、冲突处理、重试或作废 |
| `GAME_INTERNAL` / `COMPLETE` / `VOIDED` | 无外部输入 | 状态机内部推进或终态 |

入口位置：

```text
RegistrationRuntime.robot_requirement()
SessionRuntime.robot_requirement()
HandRuntime.robot_requirement()
```

三个入口都返回 `RobotInterfaceRequirement`，其中包含：

```text
node
accepted_inputs
accepted_python_types
hand_phase
state_version
target_seat
target_slot
vision_slots
reason
```

具体写入入口：

```text
RegistrationRuntime.accept_control(ControlObservation)
RegistrationRuntime.accept_face_enrollment(FaceEnrollmentObservation)
HandRuntime.accept_navigation_ack(NavigationAck)
HandRuntime.accept_visual_settle()
HandRuntime.accept_identity(FaceIdentityObservation)
HandRuntime.accept_action(PlayerActionObservation)
HandRuntime.accept_attributed_action(AttributedActionCandidate)
HandRuntime.accept_chip_observation(ChipObservation)
HandRuntime.accept_rotation_ack(DealerAck)
HandRuntime.accept_dispense_ack(DealerAck)
HandRuntime.accept_card_observation(CardObservation)
DealerPort.confirm_navigation_target(NavigationAck)
SessionOperatorController.accept(ControlObservation)
```

完整字段合同位于：

- `src/poker_dealer/domain/robot.py`
- `src/poker_dealer/runtime/robot_interfaces.py`
- `src/poker_dealer/runtime/ports.py`
- `src/poker_dealer/robotics/navigation/port.py`

验证结果：

- 新接口及旧完整牌局相关目标测试：`60 passed`，包含导航接口整局回放。
- 实用完整测试：`460 passed, 9 failed`；9 项均来自本目标以外的既有工作区/环境，
  包括未安装 Vosk、当前 ONNX 与 OpenCV 版本不兼容、未安装包方式启动脚本，以及
  已修改但未归本目标所有的网络地址断言。
- 后续仍需在 recorded replay 之后做空载低速实机联调；当前没有完成物理验证。
