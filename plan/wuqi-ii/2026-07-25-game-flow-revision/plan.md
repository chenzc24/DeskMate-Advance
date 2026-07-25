# 四人德州扑克小车游戏流程修订计划

状态：`产品流程已由用户明确，等待规则合同、状态机与运行时实现迁移`

日期：`2026-07-25`

负责人：`wuqi-ii`

## 1. 目标与本次修改范围

本计划定义四人德州扑克小车的新目标流程。小车只负责：

1. 发出玩家底牌和公共牌。
2. 移动到当前行动玩家面前并询问、记录玩家操作。
3. 识别正面可见的扑克牌。
4. 在 `bet` 和 `raise` 时识别当前玩家投入的筹码金额。
5. 根据已经提交的游戏结果播报赢家和赢取数额。

小车本阶段明确不负责：

- 洗牌和装牌。
- 翻开公共牌；公共牌由玩家手动翻开。
- 移动、铲取、整理、分配或退还实体筹码。
- 自动收牌。
- 通过视觉模型直接判断动作是否合法、决定轮到谁或计算牌型。
- all-in、边池和由 all-in 引发的短注、重开下注权等规则。

确定性游戏状态机仍负责行动顺序、动作合法性、下注账本、街道推进、
牌型比较和底池结算。视觉模块只提供牌面、筹码金额和玩家行为的观察证据。

## 2. 本次冻结的产品规则

### 2.1 基本牌局

- 固定四个座位和四名玩家。
- 牌局开始前完成人物信息收录，并冻结本次 Session 的座位绑定。
- 每局显式确定庄家 Button；后续正常结束的牌局按顺时针移动 Button。
- 使用 52 张无鬼牌扑克牌。
- 不发 burn card。
- 小车向每名玩家发两张底牌。
- Flop 发三张公共牌，Turn 和 River 各发一张公共牌。
- 小车只负责把公共牌发到中心区，玩家负责把公共牌翻至正面。

### 2.2 盲注

- 每局开始时由数字账本自动登记大小盲，不要求小车识别盲注筹码。
- Big Blind 默认值为 `20`。
- Small Blind 暂按标准半盲规则设为 `10`。
- `10/20` 是当前产品默认值，后续仍应通过配置读取，不在算法中硬编码。
- 盲注提交后才允许进入底牌发牌阶段。

### 2.3 玩家动作

本版本只接受以下五种动作：

- `check`
- `bet`
- `call`
- `raise`
- `fold`

明确禁止：

- `all-in`
- 任何会使玩家余额低于零的动作
- 不足额 call
- 不足最小加注额的 raise

如果玩家余额不足以完成一个合法的 `call`、`bet` 或 `raise`，系统不能把
该动作降级成 all-in；应拒绝该候选并要求玩家重新选择合法动作。

### 2.4 筹码识别与账本

- `check`：不识别筹码，提交金额为 `0`。
- `fold`：不识别筹码，玩家退出本局后续行动。
- `call`：不识别筹码，由状态机计算需要补齐的金额。
- `bet`：小车识别当前玩家下注区域内本次新增的筹码金额。
- `raise`：小车识别当前玩家下注区域内本次新增的筹码金额。

`call` 金额必须按以下公式计算：

```text
call_amount =
    当前街道所有存活玩家中的最高累计投入
    - 当前玩家在本街道已经累计投入
```

不能直接使用“上一名玩家刚刚投入的金额”。上一名玩家可能 `check`、`fold`、
`call`，也可能并不是本轮最高下注的来源。

对于 `bet/raise`，筹码识别结果必须先成为待确认观察：

```text
ChipObservation
→ 多帧稳定与置信度检查
→ 金额合法性检查
→ 玩家或操作员确认
→ 原子更新数字账本和 state_version
```

筹码识别未知、抖动、遮挡、金额不合法或与当前下注上下文冲突时，保持当前
行动玩家和当前下注状态，不得猜测金额或推进到下一玩家。

由于小车不移动实体筹码，数字账本记录的是已确认动作带来的逻辑投入。实体
筹码是否确实保持在玩家下注区，需要由现场规则或人工监督保证。

## 3. 完整单局流程

### 3.1 Session 人物信息收录

1. 操作员启动 Session，并指定初始 Button。
2. 按固定座位依次收录四名玩家的人物信息。
3. 每名玩家必须绑定唯一的 `participant_id` 和固定 `seat_id`。
4. 人脸等身份信息只用于核验状态机已经选中的座位，不得决定行动顺序。
5. 四个座位均登记完成后冻结 `SessionRoster`。
6. 未完成收录时不得开始牌局。

### 3.2 开始新牌局和登记盲注

1. 状态机检查上一局已经结束且桌面已由人工确认可以开始。
2. 为新牌局生成唯一 `hand_id`。
3. 根据 Button 计算 Small Blind、Big Blind 和 UTG。
4. 从 Small Blind 数字余额扣除 `10`。
5. 从 Big Blind 数字余额扣除 `20`。
6. 两笔盲注和新的 `state_version` 原子提交。
7. 任何玩家余额不足以支付规定盲注时，不得开始本局；应进入人工处理，
   调整余额或结束 Session，不能自动形成 all-in。

### 3.3 发放玩家底牌

1. 从 Button 左侧第一名玩家开始顺时针发牌。
2. 一圈每名玩家各发两张。
3. 每次发牌必须遵循：

```text
状态机指定目标座位
→ 小车到达并对准目标
→ 发出一张牌
→ 发牌机构返回单张成功 ACK
→ 状态机登记该底牌槽位已占用
→ 进入下一目标
```

5. 底牌保持背面朝上，本阶段不识别底牌牌面。
6. 发牌失败、无牌、双张、卡牌、超时或目标位置不一致时进入
   `PAUSED_RECOVERY`，不得继续发下一张。

### 3.4 Preflop 下注

1. Preflop 首位行动者为 Big Blind 左侧第一名仍可行动的玩家，即四人局
   默认的 UTG。
2. 状态机指定唯一 `acting_seat`。
3. 小车移动到该玩家面前并询问操作。
4. 行为感知输出动作候选，经过时序确认和合法性检查后才能提交。
5. 如果动作是 `bet` 或 `raise`，小车额外识别该玩家下注区域的筹码金额。
6. 如果动作是 `call`，状态机按当前最高累计投入计算补齐金额。
7. 动作、下注账本和新 `state_version` 原子提交后，才能切换到下一位玩家。
8. 跳过已经 `fold` 的玩家。
9. 本轮持续循环，直到所有仍可行动玩家满足以下任一条件：

   - 已 `fold`；
   - 已对当前最高下注完成响应；
   - 本街道累计投入已经等于当前最高下注。

10. 如果下注被 `raise`，在该 raise 之后尚未响应的存活玩家必须继续行动。
11. 若只剩一名玩家未弃牌，立即进入无人竞争结算，不再发公共牌。

### 3.5 Flop

1. Preflop 下注闭合后，状态机进入 `DEALING_FLOP`。
2. 小车移动到桌面中心发牌区。
3. 小车连续发出三张公共牌，每张都必须收到独立的单张成功 ACK。
4. 小车不执行翻牌；由玩家将三张牌翻至正面。
5. 卡牌识别模块必须稳定识别三个公共牌槽位。
6. 未知牌、遮挡、重复牌或槽位冲突时保持 Flop 发牌阶段并请求重新观察。
7. 三张公共牌全部确认后进入 Flop 下注。
8. Flop 首位行动者为 Button 左侧第一名未弃牌且可行动的玩家。
9. 按与 Preflop 相同的询问、筹码识别、合法性检查和闭合算法完成本轮。

### 3.6 Turn

1. Flop 下注闭合后，小车回到桌面中心发牌区。
2. 小车发出一张 Turn 公共牌并等待成功 ACK。
3. 玩家手动将 Turn 翻至正面。
4. 卡牌识别稳定确认 Turn，重复牌或未知牌不得推进。
5. 从 Button 左侧第一名未弃牌且可行动的玩家开始 Turn 下注。
6. 使用与 Flop 相同的下注闭合规则。

### 3.7 River

1. Turn 下注闭合后，小车回到桌面中心发牌区。
2. 小车发出一张 River 公共牌并等待成功 ACK。
3. 玩家手动将 River 翻至正面。
4. 卡牌识别稳定确认 River，重复牌或未知牌不得推进。
5. 从 Button 左侧第一名未弃牌且可行动的玩家开始 River 下注。
6. 使用与 Flop 相同的下注闭合规则。

### 3.8 提前结束

在任意下注阶段，如果只剩一名玩家没有弃牌：

1. 立即停止后续询问和发牌。
2. 不要求该玩家展示底牌。
3. 状态机把当前完整底池授予该玩家。
4. 播报赢家和赢取数额。
5. 将牌局置为 `SETTLED`。

### 3.9 Showdown

River 下注闭合且仍有至少两名玩家没有弃牌时进入 Showdown：

1. 状态机冻结全部公共牌、存活玩家和最终底池金额。
2. 小车按固定且可重放的座位顺序访问每一名未弃牌玩家。
3. 玩家向小车摄像头展示两张底牌正面。
4. 卡牌识别模块稳定确认两张底牌，并执行全局重复牌检查。
5. 所有未弃牌玩家的底牌确认后，由确定性牌型算法比较最佳五张牌。
6. 状态机计算赢家和应赢取数额。
7. 播报结果：

```text
赢家座位/玩家
获胜牌型
赢取数额
```

8. 数字账本结算后进入 `SETTLED`。

“小车到两位玩家面前记录手牌”只适用于 Showdown 恰好剩两人的情况。
如果有三名或四名玩家未弃牌，小车必须访问所有未弃牌玩家，否则无法正确
判断赢家。

## 4. 建议状态机

```text
SESSION_REGISTRATION
→ READY_FOR_HAND
→ POSTING_BLINDS
→ DEALING_HOLE
→ PREFLOP_BETTING
→ DEALING_FLOP
→ WAITING_FLOP_REVEAL
→ FLOP_BETTING
→ DEALING_TURN
→ WAITING_TURN_REVEAL
→ TURN_BETTING
→ DEALING_RIVER
→ WAITING_RIVER_REVEAL
→ RIVER_BETTING
→ SHOWDOWN_CAPTURE
→ SETTLEMENT
→ SETTLED
```

所有下注阶段均允许：

```text
BETTING
→ 仅剩一名未弃牌玩家
→ UNCONTESTED_SETTLEMENT
→ SETTLED
```

任何需要人工恢复的异常均允许：

```text
任意活动阶段
→ PAUSED_RECOVERY
→ 恢复原阶段 / 作废本局
```

### 4.1 单个玩家行动子状态

```text
SELECT_ACTING_SEAT
→ MOVE_TO_PLAYER
→ WAIT_MOVE_ACK
→ WAIT_VISUAL_SETTLE
→ OPTIONAL_IDENTITY_VERIFY
→ ASK_FOR_ACTION
→ CONFIRM_ACTION
→ [BET/RAISE] OBSERVE_CHIPS
→ VALIDATE_ACTION_AND_AMOUNT
→ COMMIT_ACTION_AND_LEDGER
→ SELECT_NEXT_ACTOR 或 CLOSE_BETTING_ROUND
```

### 4.2 公共牌发牌子状态

```text
MOVE_TO_BOARD
→ WAIT_MOVE_ACK
→ DISPENSE_ONE
→ WAIT_DISPENSE_ACK
→ 重复至本街所需张数
→ WAIT_PLAYER_REVEAL
→ OBSERVE_VISIBLE_CARD
→ CONFIRM_ALL_REQUIRED_SLOTS
→ START_BETTING
```

## 5. 下注算法约束

状态机至少维护以下字段：

```text
street_highest_commit
player_street_commit[seat]
player_hand_commit[seat]
minimum_raise_increment
acting_seat
acted_since_last_full_raise
folded_seats
pot_total
```

### 5.1 Check

仅在：

```text
player_street_commit[seat] == street_highest_commit
```

时合法。

### 5.2 Call

```text
required = street_highest_commit - player_street_commit[seat]
```

- `required > 0` 时允许 call。
- 玩家余额必须大于等于 `required`。
- 直接从数字账本扣除 `required`。
- 不触发筹码视觉识别。

### 5.3 Bet

- 仅在当前街道尚无玩家主动下注时合法。
- 识别结果是玩家本次新增投入金额。
- 金额必须不低于配置中的最小 bet。
- 玩家余额必须足够支付完整金额。
- 金额确认后更新 `street_highest_commit` 和最小加注增量。

### 5.4 Raise

- 仅在当前街道已经存在主动下注时合法。
- 必须明确区分：

```text
本次新增投入金额
玩家 raise 后的本街道累计投入
相对旧最高下注的加注增量
```

- 加注增量不得低于当前 `minimum_raise_increment`。
- 玩家余额必须足够完成完整 raise。
- 合法 raise 后，其他尚未弃牌玩家需要重新获得响应机会。

具体采用“筹码视觉输出本次新增金额”还是“筹码视觉输出玩家下注区总金额”，
必须在实现前冻结。推荐识别玩家下注区的本街道总金额，然后减去账本中该玩家
已登记金额，以降低连续增加筹码时的漏计风险。

### 5.5 下注轮闭合

下注轮只有在以下条件同时满足时关闭：

1. 至少有两名玩家仍未弃牌，或已经触发提前结束。
2. 每名仍可行动玩家在最近一次有效 bet/raise 后都已经获得行动机会。
3. 每名仍可行动玩家的本街道累计投入都等于 `street_highest_commit`。
4. 不存在待确认动作、待确认筹码观察或未完成的小车命令。

## 6. 与当前 Core v1 的差异

本计划是新的目标流程，但当前代码和 `configs/game/core_v1.json` 尚未完成
对应迁移。实现前必须显式修改并重跑下游测试。

| 项目 | 当前 Core v1 | 本计划目标 |
| --- | --- | --- |
| 大盲默认值 | `2` | `20` |
| 小盲默认值 | `1` | 暂定 `10` |
| 下注结构 | Fixed-Limit 固定金额 | `bet/raise` 读取筹码金额 |
| all-in | 允许 | 禁止 |
| 边池 | 必须支持 | 本版本正常流程不产生 |
| 筹码识别 | 非权威核对 | `bet/raise` 金额确认输入 |
| 公共牌翻开 | 尚未冻结 | 玩家手动翻开 |
| Showdown 人数 | 所有存活玩家 | 保持所有未弃牌玩家 |
| 筹码搬运 | 不负责 | 不负责 |

这不是单纯调整参数。`bet/raise` 携带金额会改变动作 schema、规则校验、
账本算法、手机 UI、播报、回放日志和测试。不得只修改盲注配置后继续沿用
Fixed-Limit 引擎并声称流程已经完成。

## 7. 异常和恢复要求

- 人物信息未完成：禁止开始 Session。
- 发牌 ACK 失败、双张、卡牌或超时：暂停并人工恢复。
- 玩家操作不明确：保持当前 `acting_seat`，重新询问。
- 非当前玩家的动作：记录为拒绝证据，不改变状态。
- 筹码未识别或金额抖动：保持当前动作待确认，不修改账本。
- 非法 check/call/bet/raise：拒绝并向当前玩家播报合法选项。
- 玩家余额不足：拒绝需要 all-in 才能完成的动作，要求重新选择。
- 公共牌尚未由玩家翻开：保持 `WAITING_*_REVEAL`。
- 公共牌未知或重复：暂停，不进入下注阶段。
- Showdown 手牌未知或重复：暂停，不结算赢家。
- 软件与物理桌面状态不一致：进入 `PAUSED_RECOVERY`，由操作员核对后
  选择恢复或作废本局。

## 8. 实现分解

1. 更新产品权威文档、规则配置和 schema。
2. 把 Fixed-Limit 金额为空的动作合同迁移为金额型 `bet/raise` 合同。
3. 在引擎中禁止 all-in 和不足额动作。
4. 实现基于累计投入的 call、raise 和下注轮闭合算法。
5. 增加 `WAITING_*_REVEAL`，明确玩家翻牌后的视觉确认门。
6. 增加筹码观察的稳定、确认、合法性与账本提交接口。
7. 调整小车任务协调器：
   玩家座位、中心发牌区和 Showdown 手牌采集点均需明确 ACK。
8. 调整手机 UI、播报、日志和恢复界面。
9. 更新 simulator 和 recorded replay。
10. 增加规则、异常、筹码抖动、重复牌和多玩家 Showdown 测试。

## 9. 验收场景

至少覆盖：

1. Preflop 无 raise，四人 call/check 后正常进入 Flop。
2. Preflop 发生一次和多次 raise，行动权正确回绕。
3. 上一名玩家 fold，但下一名 call 仍追平本轮最高累计下注。
4. Flop、Turn、River 等待玩家翻牌，未识别前不推进。
5. 任意街只剩一名玩家时立即结算。
6. 两人、三人和四人 Showdown 均访问全部未弃牌玩家。
7. 筹码识别未知、金额抖动和非法最小加注不会修改账本。
8. 余额不足时拒绝动作，不生成隐式 all-in。
9. 重复牌触发暂停且不宣布赢家。
10. 发牌 ACK 超时或目标错误时不推进牌槽。
11. 整局资金守恒且玩家余额永不为负。
12. 日志回放得到相同的行动顺序、底池、赢家和赢取数额。

## 10. 文件边界、外部依赖与提交意图

本计划当前拥有的跟踪路径：

- `plan/wuqi-ii/2026-07-25-game-flow-revision/plan.md`

后续实现预计涉及但本次不修改：

- `docs/plans/POKER_DEALER_MASTER_PLAN.md`
- `docs/contracts/GAME_RULES.md`
- `docs/contracts/CORE_INTERFACES.md`
- `configs/game/core_v1.json`
- `configs/contracts/core_rules.schema.json`
- `src/poker_dealer/domain/`
- `src/poker_dealer/game/`
- `src/poker_dealer/runtime/`
- 对应的 `tests/`

外部依赖：

- 发牌机构的单张出牌和目标位置 ACK。
- 小车导航与玩家/中心服务点定位。
- 玩家操作感知。
- 扑克牌识别。
- 筹码定位、面额识别和多帧金额稳定。
- 手机交互与结果播报。

当前工作区中与本计划无关的修改、模型、数据处理脚本和未跟踪文件全部只读，
不得纳入本计划提交。

物理运动状态：

- 本计划只定义流程，不启动或授权任何实体小车运动。
- 后续实机测试必须在操作员在场、低速、可急停、区域清空和机构防护有效的
  条件下进行。

验证要求：

- Markdown 链接、术语和状态名称检查。
- 后续规则配置修改时解析 JSON/schema。
- 后续代码修改时运行目标测试、完整实用测试、模拟整局和 recorded replay。
- 始终运行 `git diff --check` 和 `git status --short --branch`。

提交意图：

- 本次仅新增计划文档。
- 未经用户后续明确要求，不提交、不推送、不创建分支或 PR。
