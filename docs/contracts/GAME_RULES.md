# 四人 Core v1 游戏规则合同

状态：四人位置/发牌/行动、状态机控制的玩家注意力、模型证据、牌槽生命周期、多人底池、showdown 和数字账本语义已冻结为 `v1.2`；Fixed-Limit 是待产品确认的候选，1/2、2/4、cap 4、stack 80 和 30 秒均为配置默认值。机器可读权威为 `configs/game/core_v1.json`。

## 玩家、Robot Dealer 与 Button

Robot 是实体荷官，不下注、不占玩家座位。四名玩家固定坐在顺时针排列的 `seat_a, seat_b, seat_c, seat_d`；其中一席持有 Dealer Button。Button 决定盲注、发牌和行动顺序，但不是第五名玩家。

以 Button=`seat_a` 为例：

```text
seat_a = Button
seat_b = Small Blind
seat_c = Big Blind
seat_d = UTG / pre-flop first actor
```

正常结算后 Button 顺时针移动一席；void/redeal 使用同一 Button。Button、SB、BB、UTG 和当前 acting seat 必须对四人清晰显示，具体用 Laptop、实体灯或机器人灯光仍开放。

## 底牌与行动顺序

牌从 Button 左侧第一名 active player 开始，按顺时针逐张发放，Button 每轮最后收到牌。Button=`seat_a` 时：

```text
seat_b -> seat_c -> seat_d -> seat_a
seat_b -> seat_c -> seat_d -> seat_a
```

Pre-flop 从 BB 左侧第一名 actionable player 开始；post-flop 从 Button 左侧第一名 actionable player 开始。寻找下一位时跳过 folded 或 all-in 玩家。[Poker TDA Rules](https://www.pokertda.com/view-poker-tda-rules/)

```text
Button A 时：
pre-flop  D -> A -> B -> C
post-flop B -> C -> D -> A
```

`src/poker_dealer/domain/game.py` 是顺时针角色和首位行动者的共享实现；UI、game 和 Robotics 不得各写一套顺序。

## 动作语义与可替换 UI

冻结的动作语义为 `fold/check/call/bet/raise`，每条请求携带 hand ID、action ID、expected state version、seat、source 和可选 `amount_units`。

- Fixed-Limit 候选中，金额由配置推导，`amount_units=null`。
- 若未来确认 No-Limit，bet/raise 必须带整数 `amount_units`。
- Laptop UI、实体按键、手势或语音都只是 adapter；game 必须再次检查 acting seat、版本和合法性。
- 旧版本、重复 action ID、非当前玩家或非法动作不改变任何账本/状态。

行为模型位于正式动作之前：game 输出唯一 `acting_seat`、合法动作和 state version，runtime 只激活机器人朝向的该席观察窗口；模型产生带时间窗口的 `PlayerActionObservation`，不能自行选择玩家或修改筹码。只有多帧/校准确认、seat/state 匹配且 game 复核合法后，动作与数字账本才原子提交；随后才切换到下一席。`ambiguous/occluded/out_of_roi/unknown`、非当前席活动和旧窗口都保持当前玩家。S0-21 允许显式同意后的本场人脸注册仅验证 `player_id ↔ seat_id`；人脸结果不能选择 acting seat、转移底牌/筹码或修改账本。

## Fixed-Limit 候选默认值

当前用于 walkthrough 和 Stage 1 预研的默认值为：SB=1、BB=2、pre-flop/flop bet=2、turn/river bet=4、每条 street 最多四个 full bets。它们不是产品冻结；Fixed-Limit 本身也要获得产品确认后，Stage 1 betting reducer 才能冻结。

## All-in、Main Pot 与 Side Pots

四人 Core 无论最终采用 Fixed-Limit 或 No-Limit，都必须支持 all-in、main pot 和多个 side pot。

例如最终投入为 A=10、B=20、C=40、D=40：

```text
main pot  = 10 * 4 = 40，eligible A/B/C/D
side pot1 = 10 * 3 = 30，eligible B/C/D
side pot2 = 20 * 2 = 40，eligible C/D
```

Folded player 已投入的资金留在 pot，但没有任何 pot eligibility。超过所有对手可匹配总额的 excess 必须退回。每个 pot 独立筛选 eligible players、比较最佳五张并处理 tie；所有 pot 与 stack 之和必须守恒，不能出现负余额。

Core 不识别实体筹码。数字账本是唯一权威；实体筹码若在桌面出现只能作为非权威道具。余额只允许由已验证动作或带 operator ID/reason 的人工调整事件改变，并与 state version 一起原子记录到 append-only hand log。

## 发牌与 Board Reveal

Hole cards face-down。每张物理牌使用 `rotate_to -> success ACK -> dispense_one -> success ACK`；只有 success ACK 才推进数量。

```text
Flop: burn_tray, board_flop_1, board_flop_2, board_flop_3
Turn: burn_tray, board_turn
River: burn_tray, board_river
```

Board card 最终必须正面朝上才能视觉确认；hole/burn 必须保持背面。选择性翻转滑道、独立 reveal board 或明确的人工翻牌 fallback 尚属 S0-13，不能在机构证明前声称已实现全自动揭示。

在任何玩家动作前发现 double-feed、错目标或数量错误，当前手 void，人工恢复完整牌副后同一 Button 重发。玩家已经实际行动后出现物理不确定性，进入 `PAUSED_RECOVERY`，软件不自动裁决。

## Showdown（S0-05 Frozen）

- Fold 到只剩一名玩家时直接结算，不要求读取 hole cards。
- 正常 showdown 时，每名 live player 把两张 hole cards 放入本席固定 ROIs；folded player 不参与识别或评估。
- 必须确认 5 张 board 和每名 live player 的 2 张 hole cards，且整手无重复身份。
- 每名玩家由确定性 evaluator 从 7 张选最佳 5 张。
- 每个 main/side pot 独立比较 eligible players；tie 时按该 pot 平分。
- Odd unit 给 Button 左侧第一名 eligible player。
- `unknown`、重复牌或缺少任一 required slot 均暂停，不能先结算某个 pot。

## 会话、Reset 与验收

默认 stack 80、动作超时 30 秒，仅为配置。每手结束后操作员取出桌面牌和 feeder 剩余牌、恢复完整牌副、洗牌、重新装牌并完成 ready/homing；精确 checklist 属 S0-14。

正式质量 Gate 固定为至少连续 20 手牌，不是产品功能或现场展示时长。20 手必须覆盖多人 fold、四人 showdown、all-in、多个 side pots、tie、位置轮转和恢复。现场 presentation 可更短，但不能替代证据。

18 个四人 walkthrough 位于 `configs/game/stage0_walkthroughs.json`；Stage 1 必须把它们转成可执行 replay，其中新增案例覆盖非当前席干扰、行为歧义、确认后切换注意力、ACK/牌面融合和审计式 rebuy。
