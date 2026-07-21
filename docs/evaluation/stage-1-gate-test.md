# Stage 1 Gate Test

测试日期：2026-07-21

结论：`ENGINEERING GATE PASS / FIXED-LIMIT CANDIDATE PASS / S0-07 PRODUCT RELEASE GATE OPEN`

Stage 1 的无相机、无机器人软件 oracle 已完成并通过测试。四人规则状态机、当前席行为 evidence、数字账本、main/side pot、牌型与逐 pot 结算、13 牌槽门控、三类 simulator、哈希链日志、冷启动恢复、18 个冻结场景、无设备整手 CLI 和 10,000 手牌随机 Gate 均可执行。

S0-07 仍是外部产品决定：当前 Fixed-Limit reducer 是完整、可替换、配置驱动且已测试的 candidate，不能标记为产品 release。该开放项不代表 Stage 1 代码未完成。

## 独立命令与结果

| 命令 | 结果 | 证明内容 |
| --- | --- | --- |
| `python -m pytest -q tests/game` | `56 passed` | Stage 1 规则、focus、账本、牌型、场景、恢复、CLI、故障矩阵、随机 Gate |
| `python -m pytest -q tests` | `93 passed` | 全仓库无回归 |
| `python scripts/game/replay_stage1.py` | `18/18 passed` | 原 Stage 0 walkthrough 全部按原 ID 和预期执行 |
| `python scripts/game/random_stage1.py --hands 10000 --seed 20260721` | `10000 hands / 84644 actions / 422 showdowns / 9578 folds` | 固定 seed 下无崩溃、死锁、超步、负余额或资金丢失 |
| `python scripts/game/demo_stage1.py` | `settled / 13 confirmed cards / 21 transitions / 26 events` | 无摄像头、无机器人完成一整手并输出状态、合法动作、pot、奖项与日志尾 hash |
| Stage 1 import-boundary scan | `clean` | `game`、测试和脚本不导入 CV、模型框架、serial、设备或 runtime |

## Gate 证据矩阵

| Gate 要求 | 判定 | 可执行证据 |
| --- | --- | --- |
| 四个 Button 位置、盲注、发牌与行动顺序 | `PASS` | `tests/domain/test_game_contract.py` 与 `tests/game/test_engine.py` |
| hand/street reducer 与合法 round closure | `PASS` | check/call/bet/raise/fold、cap、street 转换、自动 runout 和终局测试 |
| 当前席 evidence promotion 与 focus 切换 | `PASS` | 非当前席、ambiguous、低置信/时长/帧数、stale/duplicate 被拒绝；提交后才切换 |
| state version、账本和 append-only log 原子性 | `PASS` | before/after snapshot、拒绝不改状态、SHA-256 hash chain、JSONL 冷启动恢复及篡改检测 |
| main/side pot、eligibility、unmatched return | `PASS` | 三层投入、folded contributor、不同 side-pot eligible seats、资金守恒 |
| 5–7 选 5 与逐 pot showdown | `PASS` | 九类牌型、wheel、kicker、board plays、tie、odd unit 和不同 pot winner |
| 13-slot lifecycle 与可恢复 snapshot | `PASS` | schema 1.1 验证；unknown/occluded 不推进；face-up 未确认不推进；重复牌冻结；恢复一致 |
| Action/Card/Dealer 三类 simulator | `PASS` | 当前席/旧版本/取消类 evidence，牌面 unknown/duplicate，dealer success/fault/idempotent ACK |
| 18 个 Stage 0 walkthrough | `PASS` | `run_walkthroughs` 要求 handler 集合与配置 ID 精确相等，18/18 预期匹配 |
| 至少 10,000 随机合法四人手牌 | `PASS` | 10,000/10,000 settled；每手 200 次状态迁移上限、日志校验、资金与非负余额断言 |
| audited rebuy | `PASS` | setup-only、operator ID/reason、state version +1、重复 ID 幂等、恢复后幂等 |
| 无设备完整 CLI | `PASS` | 13 张可见牌确认后摊牌结算，输出 Button/actor/legal actions/pot/awards/event hash |
| Fixed-Limit betting adapter 技术实现 | `PASS — CANDIDATE` | 从 `core_v1.json` 加载默认值；amount rejection、cap、all-in 与 round closure 已测试 |
| S0-07 产品 release | `OPEN BY DESIGN` | 仍需人确认是否正式采用 Fixed-Limit；不得用测试通过替代产品签字 |

## 工作包完成度

| 工作包 | 状态 | 主要产物 |
| --- | --- | --- |
| S1.0 Contract harness | 完成 | 18 场景执行器、schema 快照验证 |
| S1.1 State + event log | 完成 | reducer、版本化状态、hash-chain JSONL、冷恢复/篡改检测 |
| S1.2 Action focus | 完成 | `PlayerActionObservation -> ActionRequest -> atomic commit -> next focus` |
| S1.3 Ledger + pots | 完成 | 非负 ledger、多层 pot、eligibility、unmatched return、audited adjustment |
| S1.4 Evaluator | 完成 | 5/7 选 5、comparison key、best five、逐 pot 分奖 |
| S1.5 Betting adapter | candidate 完成 | 配置驱动 Fixed-Limit；产品 release 等 S0-07 |
| S1.6 Three simulators | 完成 | action/card/dealer 正常与故障路径 |
| S1.7 Runtime shell | 完成 | walkthrough、随机 Gate 和整手可视 JSON CLI |

## 复现入口

- `scripts/game/demo_stage1.py`：一手完整四人 check-through 与摊牌。
- `scripts/game/replay_stage1.py`：18 个冻结 walkthrough，可用 `--scenario` 选择单例。
- `scripts/game/random_stage1.py`：固定 seed 随机合法手牌 Gate，可配置手数。

所有命令仅使用软件 simulator，不访问相机、串口、机器人或物理运动。
