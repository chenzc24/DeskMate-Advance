# Stage 4：模型—规则—机构纵向集成

## 目标

将经过各自 Gate 的 game、perception 和 dealer 接入单一 runtime，证明系统只在正确证据链下推进，并能从可预期故障中安全暂停和恢复。

## 入口条件

- Gate 1 已通过，S0-07 已确认 Fixed-Limit Core v1；集成仍必须读取配置默认值，不能把 1/2、2/4、cap 4 或 stack 80 写死。
- Gate 2A/2B 各有固定 model/export/config/calibration/manifest hashes 和 recorded replay。
- Gate 3 有固定 firmware/protocol/calibration、保护、操作员 runbook 和安全签字。
- runtime 启动时能拒绝 schema/model/firmware/calibration 不兼容组合；共享合同变化必须回 Stage 00 做迁移。

## 集成顺序

1. `game + simulated perception + simulated dealer`：回归 Stage 1。
2. `game + recorded action/card perception replay + simulated dealer`：验证当前席注意力、真实视觉时序和 unknown。
3. `game + simulated perception + real dealer`：操作员在场，验证机构 ACK/超时，不跑真人牌局。
4. `game + live perception + simulated dealer`：桌面实景但不运动。
5. 全实物低速 dry run：无玩家手、脚本化动作。
6. 四名玩家的受控完整手牌；先一手一复盘，再做连续牌局和 side-pot 场景。

任何组合失败时退回能通过的上一层，不允许同时修改模型、规则和 MCU 来“试到能跑”。

### 六级集成 Gate

| 级别 | 固定真实组件 | 本级必须证明 | 失败回退 |
| --- | --- | --- | --- |
| I4.1 全模拟 | 无 | 18 replay、账本/状态/attention/ACK 语义 | Stage 1 |
| I4.2 录制感知 | action/card recordings | 时间戳、拒识、focus/slot lifecycle 与状态对齐 | Stage 2A 或 2B replay |
| I4.3 真实机构 | dealer only | command/ACK/timeout/recovery，game 不猜物理状态 | Stage 3 bench |
| I4.4 实时感知 | camera/models only | 四席 focus、牌槽、长运行、无运动安全性 | Stage 2A/2B live evaluation |
| I4.5 全实物 dry run | camera/models/dealer，无玩家手 | 感知与运动时序、遮挡、watchdog 和恢复 | 最近失败子系统 |
| I4.6 受控四人 | 全部，操作员在场 | 一手一复盘后连续/side-pot/故障牌局 | I4.5 或对应模型/机构 Gate |

每级冻结输入版本并输出独立报告；未通过不能开始下一级。I4.3 与 I4.4 可在各自 Gate 后并行，但 I4.5 必须汇合。

## Runtime 必须实现

- 单一 hand/state version 和 append-only log；原始 action evidence、正式 action、card observation、command、ACK 与账本事件均可关联。
- runtime 只能按 game 给出的 acting seat 激活固定 action ROI；行为 candidate 必须经时序/校准、seat/state 和 legal-action 复核，提交后才能切换 focus。
- 每个物理步骤的 two-phase 行为：记录 intent/command，等终态 ACK，再提交 game transition。
- 幂等：重复 ACK/动作/重连不能重复发牌、重复扣筹码或重复结算。
- bounded queue、stale timestamp、camera/MCU health、watchdog 和 shutdown。
- `PAUSED_RECOVERY` UI 展示物理/软件期望状态、明确原因和允许的恢复操作。
- 启动自检：配置/schema、模型 hash、相机校准、MCU 版本、home、传感器、E-stop 状态、离线资产。
- 结束自检：马达安全、hand log 写入完整、无待确认命令。

## 故障注入矩阵

至少覆盖：相机缺帧/断连、四方向牌 ROI/action ROI 校准漂移、牌面低置信/错序/重复、当前玩家动作取消/遮挡/歧义、非当前席同时动作、跨席泄漏、旧 evidence window、重复/非法正式动作、side-pot eligibility 冲突、dealer timeout/failed/double-feed/jam、board 未正确翻面、ACK 丢失/重复/未知 ID、进程重启、MCU 重启、E-stop、玩家在发牌中移动牌。

对每个故障记录：注入点、预期状态、是否运动、停止时间、日志字段、人工恢复步骤、恢复后是否继续或作废。只有预期与实际一致才通过。

## Gate 4

- I4.1–I4.6 全部通过；每个真实硬件/相机故障可由 simulator/replay 重现。
- 任何未 ACK 的物理动作、unknown/duplicate 牌、非当前席/未确认行为 evidence 或非法 action 都无法推进 state version、切换 attention 或改变账本。
- 进程重启后能从 log 恢复到安全暂停，不能自动猜测物理桌面。
- 连续 10 手牌 dry run 无非法状态；所有人为注入故障进入正确恢复路径。
- 性能、内存、时延和日志规模有目标 Laptop 的连续运行报告。
- Robotics owner 和 operator 签字后才能进入真人连续验收。

## Gate 4 交付包与变更纪律

- compatibility matrix：Git/config/schema、action/card model、ROI/calibration、firmware/protocol/CAD/BOM hashes。
- 六级报告、故障注入矩阵、P50/P95/P99 时延、吞吐/内存、bounded log/queue 和恢复结果。
- 至少一份由 append-only log 生成的独立 replay/checker 结果，证明 ledger 与 state 可重建。
- 集成中若只改阈值，仍需生成新 config hash 并重跑受影响的 I4 级别；若改 schema/动作语义/槽位，回 Stage 00；若换模型/机构候选，回对应 Gate 2/3。
- 不允许在 demo-only 分支关闭 unknown、跳过 ACK、放宽动作确认或直接修正赢家/余额。
