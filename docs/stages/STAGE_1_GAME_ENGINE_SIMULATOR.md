# Stage 1：四人确定性游戏引擎与全软件模拟器

状态（2026-07-22）：`ENGINEERING COMPLETE / TESTED / FIXED-LIMIT CORE V1 CONFIRMED`

实现、可执行场景与测试证据已通过；详细结果见 [Stage 1 Gate Test](../evaluation/stage-1-gate-test.md)。2026-07-22 产品决定已经关闭 S0-07：Core v1 采用配置驱动的 Fixed-Limit，数值继续只是可配置默认值。

## 目标与当前边界

在没有摄像头和机器人的条件下完成四人牌局，并提供规则 oracle。Core v1 使用已经确认的 Fixed-Limit reducer；位置、发牌、牌型、pot builder、状态/日志和 simulator 仍保持配置驱动与设备无关。

## 入口条件与阶段边界

- 入口：Stage 00A 的 rules v1.3、domain/schema 和 18 个 walkthrough 已可用。
- 可立即开始：事件/快照、状态 reducer、当前席 focus、数字账本、pot builder、evaluator、simulator 和 replay。
- 已关闭条件：S0-07 已由产品确认；Fixed-Limit adapter 是 Core v1 正式 betting reducer，押注数值由配置提供。
- 禁止：导入 OpenCV/模型/serial，以真实相机或机器人作为单元测试前提，或让 UI/evidence 直接写账本。

## 交付物

- 四席、Button/SB/BB/UTG、street、动作和版本化 hand state。
- 顺时针 active/actionable seat traversal，跳过 folded/all-in，并为四个 Button 位置生成角色与行动顺序。
- 候选 betting reducer：读取版本化 betting config，而不是把 1/2、2/4、cap 4 写死。
- 数字账本、all-in contribution layers、main/side pot builder、pot eligibility、unmatched excess return 和资金守恒。
- 5–7 张牌 evaluator；对每个 pot 独立比较 eligible live players，返回牌型、最佳五张和 comparison key。
- `SimulatedDealer`：9 个目标，可脚本化 success、timeout、jam、double-feed、disconnect、重复/乱序 ACK 和 board reveal failure；无烧牌目标。
- `SimulatedCardPerception`：13 slots，可注入 confirmed/unknown/empty/face_down/face_up_unconfirmed/occluded、重复牌、错槽和延迟。
- `SimulatedActionPerception`：为当前席产生 `no_action/action_start/candidate/ambiguous/occluded/out_of_roi/unknown`，并可注入非当前席动作、旧 state window、取消动作和重复候选。
- Adapter-neutral action input；evidence 先走 `PlayerActionObservation`，通过确认/复核后才生成 `PlayerAction`；至少一个 Laptop UI 走同一最终 schema。
- CLI/replay：四名玩家可完成整手牌，显示 Button/SB/BB/UTG/current actor、pots、合法动作和 hand log。

## 工作包与实现顺序

| 工作包 | 核心工作 | 可验收产物 | 依赖/并行 |
| --- | --- | --- | --- |
| S1.0 Contract harness | 加载/版本检查、18 walkthrough runner、schema fixtures | 一个命令运行全部合同 replay | 首先完成 |
| S1.1 State + event log | hand/street phase、state version、append-only events、snapshot/recovery | 纯 reducer、事件日志、重启恢复测试 | 与 S1.2/1.3 的纯函数部分并行 |
| S1.2 Action focus | acting seat、action ROI context、evidence temporal adapter、合法性/过期/重复拒绝 | 非当前席和 stale evidence 不推进；提交后才切 focus | 依赖 S1.1 版本语义 |
| S1.3 Ledger + pots | stack、street/hand contribution、all-in layers、main/side pots、unmatched return、audited rebuy | 守恒/幂等属性测试和独立 pot vectors | 可与 S1.4 并行 |
| S1.4 Evaluator | 5/7 选 5、comparison key、tie/odd unit、逐 pot eligibility | 权威牌型向量和 settlement tests | 可与 S1.3 并行 |
| S1.5 Betting adapter | 配置驱动 legal actions、round closure、Core v1 Fixed-Limit | 正式 reducer；无硬编码数值 | S0-07 已关闭 |
| S1.6 Three simulators | action/card/dealer evidence、延迟/故障/乱序/重复脚本 | 可复现 recorded-like replay 和 fault matrix | 依赖 1.1 接口稳定 |
| S1.7 Runtime shell | CLI、状态/账本显示、暂停恢复、bounded queue/log | 无相机/无机器人完成整手牌 | 汇合 1.1–1.6 |

优先级不是“先把所有牌型写完”，而是尽早完成 S1.0–S1.2 的纵向薄片：当前席 evidence 被接受后，动作、账本、state version 和下一席 focus 一次提交；unknown/非法证据保持原状态。随后再把全部下注和结算语义填入同一 reducer。

## 必测情形

- 每个 Button 位置：Deal 从 Button 左侧开始，Button 每轮最后；pre-flop 从 BB 左侧开始，post-flop 从 Button 左侧开始。
- 中间 seat fold/all-in 后的顺时针跳过；无 actionable player 时自动 run out board，而不是等待输入。
- 多人 fold 到一人、四人 check-through、candidate raise cap、非法 action/旧 state version/重复 action ID；非当前席、ambiguous、occluded 或取消动作均不得推进或切换 attention。
- 至少三层投入、folded contributor、unmatched excess、多个 side pots、不同 pot winners、pot 内 tie/odd unit。
- High card 到 straight flush、wheel、board plays、kicker；每位玩家最多用 7 张选 5 张。
- 9 target no-burn deal sequence、13-slot lifecycle/unique card set、未 ACK 不推进、face-up 未确认/unknown/duplicate/未清桌不结算；完整 River 发牌为 13 次 dispense。
- Timeout、jam、reveal failure、重复/乱序 ACK、进程/MCU 重启、void/redeal same Button。

## Gate 1

- 四个 Button 的位置/发牌/行动黄金测试 100% 通过。
- Property tests 不产生负 stack/pot、资金丢失、重复牌、错误 actor、错误 eligibility 或 Button 漂移。
- 随机合法四人序列至少 10,000 手，无崩溃、死锁或不可达状态。
- 18 个 Stage 0 walkthrough 转成可执行 replay，不能替换为更简单案例。
- Game tests 不导入 OpenCV、模型框架、serial 或真实设备。
- S0-07 已确认，Fixed-Limit betting reducer 总门关闭；模型、硬件与安全 Gate 仍独立开放。

### 当前 Gate 判定

- 软件工程子门：通过。状态机、focus、账本/边池、牌型、牌槽门控、三类 simulator、恢复、18 场景、CLI 和 10,000 手随机测试均有可执行证据。
- Fixed-Limit Core v1 技术子门：通过。配置默认值、合法动作、cap、all-in 和 round closure 已测试。
- Fixed-Limit product release 总门：S0-07 已由产品决定关闭；结构确认为 Core v1，数值继续由配置提供默认值。

### Gate 1 交付包

- 源码与冻结配置、18 个 executable replay、黄金/属性/随机测试摘要。
- 每个状态迁移的 before/after snapshot、触发 event 和拒绝原因；append-only log 可从冷启动恢复到一致状态。
- action focus 报告：当前席确认、非当前席干扰、ambiguous/occluded、stale/duplicate evidence、提交后切换。
- ledger 报告：总资金守恒、0 个负余额、side-pot eligibility、rebuy audit 和重复 settlement 幂等。
- simulator 场景目录和一条无设备完整四人牌局命令。

### 失败回退

- 状态或账本不守恒：停止 UI/模型接入，回到纯 reducer/event vectors。
- attention 错切：保持单席显式 simulator 输入，修复 state/version 原子边界后再恢复行为 evidence。
- S0-07 已定：Fixed-Limit adapter 已收敛为 Core v1；其他模型、硬件和安全 Gate 仍需独立关闭。
