# 四人 Stage 0 Gate 审计

审计日期：2026-07-21

结论：`FOUR-PLAYER PERCEPTION/STATE CONTRACT MIGRATED / PRODUCT + MODEL + HARDWARE EVIDENCE OPEN / STAGE 0 NOT CLOSED`

## 要求—证据矩阵

| 要求 | 当前证据 | 结论 |
| --- | --- | --- |
| 四人产品范围 | Master、Stage 0、AGENTS、rules v1.2 | 已迁移 |
| Button/SB/BB/UTG 分离 | 四座顺时针 domain functions + 每个 Button 测试 | 软件通过 |
| 8 张 hole deal order | 每个 Button 两轮顺序测试，Button 每轮最后 | 软件通过 |
| Pre/post-flop 与 skip | UTG/post-flop first、folded/all-in skip tests | 软件通过 |
| 多人 main/side pot 语义 | Rules + WT-05/06；Stage 1 builder 尚未实现 | 合同通过、执行开放 |
| S0-05 | 13 vision slots 与所有 live players showdown | 冻结 |
| S0-06 | action schema 含五种语义、replaceable source、可选 amount | 按用户决定调整 |
| S0-07 | Fixed-Limit candidate + configurable defaults | 产品确认开放 |
| S0-12 | 20 hands 只在 acceptance contract | 冻结 QA 策略 |
| S0-16 行为 evidence | `PlayerActionObservation` schema/domain；手势/阈值未定 | 接口通过、模型开放 |
| S0-17 attention | acting seat 唯一权威、提交后切换、非当前席不推进 | 冻结 |
| S0-18 确认/拒识 | 多帧、校准、版本/合法性、unknown 保持 | 合同通过、阈值开放 |
| S0-19 table scene | 13-slot lifecycle、ACK 与视觉职责分离 | 合同通过、实测开放 |
| S0-20 ledger | 数字账本唯一权威、原子提交、append-only audit | 冻结 |
| 10 mechanical / 13 vision IDs | JSON、schema 与 Python enums 精确匹配 | 逻辑通过 |
| 四人 walkthrough | 18 个规则/故障/side-pot/轮转/行为/场景/账本案例 | 合同通过 |
| 相机、feeder、reveal、safety | 无实体证据 | 开放 |
| Cross-language MCU mock | 无 Robotics parser evidence | 开放 |

## 明确未完成

- Fixed-Limit 尚未获得产品确认，因此 Stage 1 可以实现位置、pot builder 和 evaluator，但不能冻结最终 betting reducer。
- S0-13 尚未决定 Robot 如何让 board face-up；“全自动公共牌揭示”不能写成已实现能力。
- 四座桌面毫米几何、相机视场、四个牌面方向、运动禁手区、传感器与急停均未实测。
- 18 个 walkthrough 是合同案例；Stage 1 必须转成可执行 replay。
- 行为模型尚无目标参与者/session 数据；具体手势、action ROI、阈值、冷却和显式确认策略均未录取。
- 牌槽 lifecycle 已定义，但 occupancy/face orientation、未清桌和 ACK/视觉融合尚无 target-table replay。

## 关闭 Gate 所需证据

确认 S0-07；用真实牌完成 feeder/reveal 和 10 target paper/prototype tests；提交 13-slot 与四个 action ROI 的 target-camera sample，包含 held-out participant/session、no-action、邻席干扰、遮挡/取消动作与完整牌槽 lifecycle；冻结行为交互/阈值、安全/传感器和 MCU framing；完成四玩家完整纸模、role indication、reset 及 schema 互解析。此前允许纯软件工作和受控小样，不允许模型录取、CAD release 或物理集成。

## 下游阶段就绪度

| 下游 | 当前允许 | 当前禁止/阻塞 |
| --- | --- | --- |
| Stage 1 | contract harness、state/event、attention、ledger/pots、evaluator、三类 simulator | betting reducer release 等 S0-07 |
| Stage 2A | 采集工具、纸模、少量 feasibility pilot | 模型录取等 S0-02/16/18 与 participant/session plan |
| Stage 2B | ROI/图像质量工具、少量 feasibility pilot | 模型录取等 S0-02/04/11/13/19 与 deck/session plan |
| Stage 3 | 有操作员/保护的 feeder/reveal/protocol bench | CAD/firmware release、牌局驱动和无人运动 |
| Stage 4/5 | simulator 接口设计 | 真实纵向集成和验收，等待 Gate 1/2A/2B/3 |

因此当前关键路径不是等待整个 Stage 00 后再工作：Stage 1 软件闭环可以启动，同时推进 00B 的产品/相机/机构证据；但不得提前把 Stage 2/3 候选标成 release。
# S0-21 增量审计

| 要求 | 当前证据 | 结论 |
| --- | --- | --- |
| 本场、显式同意注册 | 配置强制 consent；UI 无 `--consent-confirmed` 时禁止注册 | 软件边界通过，现场同意流程未验收 |
| embedding 仅内存 | observation schema 无 embedding；图库不提供序列化并在退出时清空 | 单测通过，进程/崩溃场景仍需审计 |
| 不接管 acting seat | 输入为状态机 `focus_seat/state_version`；mismatch/unknown 不写游戏 | 合同与隔离测试通过 |
| 玩家身份质量 | YuNet + SFace Laptop Pilot，无 held-out participant/session 指标 | 开放，不能晋升 candidate |
| 活体与防重放 | 未实现 | 开放，不能用于安全认证 |

S0-21 是对先前“完全不使用生物身份”的显式产品修订，但 S0-17 的席位权威保持冻结。它不使 Stage 0 或 Stage 2A Gate 自动关闭。
