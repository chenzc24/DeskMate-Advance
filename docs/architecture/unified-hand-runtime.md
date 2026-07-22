# 统一手牌运行时

## 状态权威

`HandRuntime` 不拥有另一套高层状态。它只读取 `HandEngine.state.phase`，
并保证任意时刻至多激活一条轨道：

| `HandPhase` | 唯一活动轨道 | 允许输入 |
| --- | --- | --- |
| `DEALING_HOLE` | Part B | 转向 ACK、发牌 ACK、当前底牌槽背面观察 |
| `AWAITING_ACTION` | Part A | 画面稳定、身份/归属、当前玩家动作证据 |
| `DEALING_BOARD` | Part B | 转向 ACK、发牌 ACK、当前公共牌槽观察 |
| `SHOWDOWN` | Part B | 转向 ACK、未弃牌玩家两张底牌的确认观察 |
| `SETTLED/PAUSED_RECOVERY/VOIDED` | 无 | 仅允许对应的操作员流程 |

正式入口为 `HandRuntime.from_roster()`，它拒绝未冻结或不含四个唯一席位的
注册名单，然后通过 `HandEngine.setup_session()` 创建空桌状态，再用
`begin_hand()`扣除盲注并进入 `DEALING_HOLE`。`HandRuntime.new_hand()`保留给
模拟/测试装配；旧的 `HandEngine.start()`保留给 Stage 1 预发牌软件 oracle，
二者都不是完整产品注册入口。

## Part B 局部状态

Part B 只维护当前目标和命令相关性：

```text
WAITING_ROTATION_ACK
  -> WAITING_DISPENSE_ACK
  -> WAITING_VISUAL_CONFIRMATION
  -> 下一目标或 COMPLETE
```

Showdown 不发牌，因此成功转向后直接等待该玩家两个固定底牌槽。所有命令
必须使用唯一 `command_id`；重复的已接纳 ACK 幂等，未知/错目标/失败 ACK、
命令超时、视觉超时和牌面冲突都会让权威引擎进入 `PAUSED_RECOVERY`。

成功发牌 ACK 会先把目标槽持久化为 `delivery_pending`，之后才等待视觉。
进程重启时，`delivery_pending/face_up_unconfirmed` 从视觉步骤恢复，不会再次
发牌；若恢复快照仍有未决 command，则进入 `PAUSED_RECOVERY`，不猜测 MCU
是否已经执行。

Core v1 无烧牌。完整走到 River 时，Part B 只执行八次底牌和五次公共牌
发放，共十三次 `dispense_one`。

## Part A 约束

生产默认值要求画面稳定和 ActorBinding。动作窗口从身份核验成功后计时；
超出规则配置的动作时间会暂停权威手牌。转向 ACK 同样支持重复幂等和超时。

`ButtonBettingRuntime` 是直接测试规则/账本的 Laptop pilot，必须显式传入
`allow_direct_engine_pilot=True`。正式控制输入必须进入 `HandRuntime/Part A`，
不能绕过身份、当前席位、证据时序和合法性门禁。

## Showdown

正式路径调用 `settle_confirmed_showdown()`，只从引擎拥有的五个公共牌槽和
所有未弃牌玩家的两个底牌槽组装输入。外部模型、UI 和 runtime 不能直接
提供赢家或替代牌面。

## 当前集成边界

统一 runtime 已通过全模拟整手牌测试，包括八张底牌、四轮下注、五张公共
牌、四名玩家 Showdown 和最终结算。现有摄像头 Live Part A 与单槽 Live
Card Pilot 尚未装配到一个实体运行入口；在 Stage 3 安全、协议和机构 Gate
完成前，不得用本模块驱动无人看守的真实运动。

模拟证据见 [统一手牌 Runtime 模拟验证](../evaluation/unified-hand-runtime-simulator.md)。
