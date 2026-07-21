# Stage 2：玩家行为与目标桌面牌面感知

当前状态（2026-07-21）：Stage 2A 已有 Laptop 手势、英文七词语音和保守冲突融合；真人失败证据已将 `Closed_Fist -> bet` 改为 `Victory -> bet`。四席 Laptop ROI Pilot 现可检测最多四只手、标注 A/B/C/D，并只让当前 focus seat 进入动作确认；象限布局不是目标桌面几何。真人多席矩阵、最终交互语法、participant/session 数据与目标相机 Gate 仍开放。详见 [Laptop 手势 Pilot](../evaluation/stage2a-laptop-gesture-pilot.md)、[多模态动作 Pilot](../evaluation/stage2a-multimodal-action-pilot.md)和[四席 Laptop Pilot](../evaluation/stage2a-multiseat-laptop-pilot.md)。Stage 2B 尚未启动。

## 目标

在状态机指定的固定玩家 action ROI 中确认当前玩家行为，并在固定桌垫/牌槽中高精度确认占用、朝向和正面牌身份；证据不足时稳定拒识。该阶段负责“当前关注区域看见什么”，不负责“轮到谁”“动作是否合法”“现在应该发什么”或“谁赢”。

Stage 2 分成可以并行、必须分别过 Gate 的 2A/2B。二者共享目标相机、帧时间戳、标定版本、manifest/provenance 和离线部署约束，但使用不同标签、split、模型和指标；不得把同一总 accuracy 当成两个 Gate 的证据。

当前若由两人实施，默认一人端到端拥有 2A 手势/语音/融合/动作 UI，另一人端到端拥有 2B fixed-ROI/牌面分类/牌槽 lifecycle；不把手势和语音拆给两人。独占路径、共享只读区、Laptop 一周节奏和 handoff 见 [Stage 2 双人分工](../plans/STAGE2_TWO_PERSON_WORKSPLIT.md)。

## 入口条件

| 子轨 | 必须先满足 | 未满足时允许做什么 |
| --- | --- | --- |
| 2A 玩家行为 | S0-02/16/18：动作语法、反馈/确认方式、四个 action ROI、目标相机 pilot、participant/session split | 只能做纸模、录制工具和少量不录取模型的 feasibility pilot |
| 2B 牌面场景 | S0-02/04/05/11/13/19：桌垫/牌槽、showdown/reveal、牌副、相机 pilot、deck/session split | 只能做 ROI/图像质量基线和不录取模型的 feasibility pilot |

共同前置：Stage 1 提供可脚本化 focus/slot lifecycle replay，模型输出 schema 已冻结；任何原始身份媒体留在 ignored data paths，许可/同意状态进入 manifest。

## A. 玩家行为感知

### A1. 输入、输出与注意力门控

- 输入必须带 `hand_id`、`expected_state_version`、状态机给出的 `focus_seat` 和对应固定 action ROI；不用人脸识别确定玩家。
- 非当前席活动可以保留为负样本/干扰评估，但不得进入正式动作候选路径。
- 模型/时序层只输出 `PlayerActionObservation`：`no_action|action_start|candidate|ambiguous|occluded|out_of_roi|unknown`，其中只有 candidate 可带五种语义动作。
- runtime 负责多帧聚合和置信校准；game 再检查当前 seat/state/legal actions。正式动作原子提交后才能切换到下一席 ROI。

### A2. 候选架构与选择 Gate

先冻结具体手势语法并采集 target-camera pilot，再选模型。默认比较顺序：

1. 明确动作区域和持续时长的规则/模板基线，用于暴露交互定义歧义。
2. 手/上肢 landmark 序列 + compact TCN，适合人为设计、几何可分的动作，数据量和部署成本较低。
3. 小型 RGB 时序模型，仅在 landmark 对手持牌、遮挡或视角变化的失败证据表明其不足时比较。
4. 语音、按钮或 UI 是同一动作语义的可替换 evidence adapter，不与视觉模型共同成为账本权威；语音/手势一致、单通道或冲突都必须先落回 `PlayerActionObservation` 再由 game 裁决。

模型选择不得只看离线 accuracy。关键是错误接受率、拒识覆盖、当前/邻席隔离和确认延迟；如果动作语法本身难以区分，先改变交互设计而不是堆模型。

### A3. 行为数据与评估

- 原始单元是包含完整当前席和邻席上下文的连续片段；manifest 记录 participant、完整 session、seat、camera、lighting、gesture grammar version、action/no-action、取消/遮挡/邻席活动和 hash。
- split 先按 participant 与完整 session，再生成窗口；同一动作及相邻窗口不得跨 split。
- 正样本覆盖五种动作、四个席位、不同体型/衣着/速度/左右手；负样本必须覆盖发呆、拿牌、摆筹码、摸脸、喝水、离席、动作开始后取消、两人同时运动和非当前席完整动作。
- 报告每动作 precision/recall/F1、混淆矩阵、校准、unknown/rejection coverage、false accepted actions per hour/per hand、跨席泄漏率、取消动作误确认率及 P50/P95 确认延迟。
- Gate 要求在 held-out participant/session 上达到项目确认阈值，长 no-action 牌局 replay 不出现未拦截的账本变更；具体数值在 pilot 后写入配置，不在 Stage 00 猜定。

### A4. 执行工作包

| 工作包 | 产物 | 完成条件 |
| --- | --- | --- |
| S2A.0 Interaction pilot | 动作说明、可用性记录、失败手势和反馈方式 | 人能稳定表达且彼此可区分；否则返回 S0-16 |
| S2A.1 Capture/labels | source manifest、participant/session split、标注指南/复核集 | 无相邻窗口泄漏，no-action/邻席/取消/遮挡齐全 |
| S2A.2 Baselines | 规则/模板、landmark quality、简单时序基线 | 明确错误来自交互、landmark 或分类 |
| S2A.3 Candidate training | compact TCN；必要时 RGB temporal 对照 | 相同 split/seed/输入合同下比较，run 可追溯 |
| S2A.4 Confirmation | calibration、多帧 entry/exit、cooldown、stale/冲突规则 | 输出稳定 `PlayerActionObservation`，不直接发正式 action |
| S2A.5 System replay | 长 no-action、四席轮转、邻席同时动作、取消和遮挡 | false acceptance/跨席泄漏/延迟报告完成 |
| S2A.6 Export/admission | Laptop 离线 export、model manifest candidate/release 记录 | reference/export 一致，所有资产有 hash |

S2A.2/2A.3 可以与 S2B 训练并行；S2A.4 必须消费 Stage 1 的真实 focus/state-version contract，不能在孤立 clip classifier 上宣布通过。

## B. 牌面感知

### B1. 候选架构与选择逻辑

### 定位层

首选固定 ROI + 桌垫基准点/单应性校准 + 牌边缘/角点质量检查。它可解释、数据需求低、不会把手或筹码当牌。只有当 target-camera replay 证明牌的落点漂移超出 ROI、角点恢复率无法达标时，才启动轻量 detector。

### 识别层

首选 ImageNet 预训练 `MobileNetV3-Small` backbone，使用两个分类头：13 类 rank、4 类 suit；再由组合得到 52 张牌。优势是共享视觉特征、样本效率高、错误可诊断，并能对 rank/suit 分别校准。比较候选包括：

1. 无训练的 template/ORB 基线，用来测量任务下限和数据问题。
2. MobileNetV3-Small 双头模型，默认主候选。
3. EfficientNet-Lite0 或同量级 backbone，仅在相同输入/数据/导出条件下比较。
4. 52 类单头模型作为消融，不作为默认架构。
5. detector + classifier 只在定位 Gate 触发后进入。

不在本文锁定特定框架版本或网络权重 URL；选择时必须核对当前官方实现、权重许可、Laptop 导出支持并登记 SHA-256。旧 DeskMate 模型和 Baseline 权重禁止复用。

### B2. 数据计划

- 原始单元是完整 target-camera 帧，不是裁剪图；manifest 记录牌副 ID、完整 session、相机/镜头、桌垫、灯光、槽位、牌面、方向、遮挡、同意/许可和文件 hash。
- split 先按实体牌副和完整 session，再生成 ROI/crop/增强；同一源帧所有牌槽和所有增强留在同一 split。
- 至少留出一副未参与训练的实体牌和多个未参与训练的 capture session 作为最终 test。
- 正样本覆盖 52 张牌、13 个 ROI、四席牌面方向、允许角度和亮度；负样本覆盖空槽、牌背、手、筹码、阴影、反光、模糊、半张牌、两张重叠、错误牌副和校准失败。
- Derived view 记录 parent ID、homography/ROI config hash、crop/normalization/augment hash、extractor version；它是可删除 cache。
- 优先采集少量完整矩阵做 pilot；先看混淆和域偏移，再扩大采集，避免堆积同质相邻帧。

### B3. 训练与评估阶段

1. 建立 ROI/角点可恢复率与图像质量报告。
2. 跑 template/ORB，发现字体/花色/反光问题。
3. 冻结 pilot manifest 与 deck/session split。
4. 相同 split、输入尺寸、seed 集比较候选；记录 run provenance。
5. 对 rank/suit 分别做 calibration，定义确认/unknown 阈值。
6. 加多帧稳定器：一致身份连续确认、变化/遮挡回 unknown、冷却与 stale 超时。
7. 导出离线模型，比较原框架与部署输出，登记 manifest。
8. 用长时间完整牌局 replay 测量误确认、重复牌、延迟和内存，而非只测裁剪 accuracy。

### B3.1 执行工作包

| 工作包 | 产物 | 完成条件 |
| --- | --- | --- |
| S2B.0 Optical pilot | 每槽 glyph pixels、反光/遮挡/落点和 homography 报告 | 相机/牌槽候选可用；否则回 S0-02/04/11 |
| S2B.1 Scene states | empty/face-down/face-up unreadable/occluded 基线 | lifecycle 状态不会被强行分类为某张牌 |
| S2B.2 Dataset | source manifest、deck/session split、标签复核集 | 52 张、13 slots、四方向和完整负样本覆盖 |
| S2B.3 Identity candidates | template/ORB、MobileNetV3 双头、必要消融 | 同 split/seed/export 条件比较 |
| S2B.4 Temporal/fusion | calibration、多帧稳定、ACK/预期 slot 融合、duplicate check | 阶段错误/错槽/未知不推进 |
| S2B.5 Full-hand replay | flop/turn/river/showdown/reset 和故障片段 | 按 slot/deck/session 报告系统指标 |
| S2B.6 Export/admission | Laptop 离线 export、model manifest candidate/release 记录 | reference/export 一致，资产和阈值有 hash |

### B4. 指标与 Gate 2

- 定位：每槽 ROI/角点成功率、空槽误定位、遮挡/反光失败率。
- 分类：rank/suit/52-card 的 per-class precision、recall、F1、confusion matrix、ECE/可靠性、unknown coverage。
- 系统：确认 precision 建议 ≥ 99.5%；低置信可降低 recall 但不能猜牌；0 个未被拦截的 duplicate identity；稳定确认 P95 目标 ≤ 1 s。
- 泛化：分别报告 held-out physical deck、session、光照和槽位；不能用相邻帧随机 split 支撑泛化结论。
- 四人布局：分别报告 A/B/C/D 四个牌面方向和 13 个 slots；不能用 board-only 指标代表多人 showdown。
- 部署：目标 Laptop 离线加载，连续 60 分钟 bounded memory，无运行时下载，输出与 reference export 在容差内一致。
- 失败证据明确：哪些情况要用户重翻牌/移开手，哪些情况需要重新校准，哪些情况触发 detector 训练。

牌槽还必须报告 `empty/face_down/face_up_unconfirmed/confirmed/occluded/unknown` 在阶段驱动 replay 中的混淆，以及 ACK 已到但卡牌缺失/错槽、未清桌和 reveal 失败的拦截率。

Gate 2A/2B 分别录取行为模型和牌面模型；任一模型未达到阈值时，先改变手势/ROI/牌槽/光照等交互约束，再根据失败证据决定继续训练。两个模型均不得绕过状态机或扩张为无约束通用场景理解。

## Gate 2A/2B 交付包与回退

- 两条子轨分别交付 source/split/view manifest hashes、resolved config、run provenance、每类指标、混淆、校准、长 replay、失败样例摘要、export/hash 和 manifest 状态。
- Gate 2A 必须同时证明非当前席 evidence 不会被提升；Gate 2B 必须同时证明 unknown/duplicate/错槽/未清桌不会被登记为有效局面。
- 达不到指标但交互/光学证据明显不可分时，回 Stage 00 修改手势、反馈、相机、桌垫或光照并版本迁移；不得只扩大模型。
- 仅 landmark 失败时才升级 RGB 时序候选；仅固定 ROI 失败时才升级 detector。升级后使用相同 held-out 分组重新比较。
- 2A 或 2B 单独通过不允许进入全实物 Stage 4，但可以与 Stage 1 simulator 形成对应 recorded replay 子 Gate。
# Stage 2A 补充：旋转视角下的玩家身份核验

原四象限 Laptop ROI 仅保留为输入路由测试夹具，不代表机器人真实视角。真实系统按状态机的顺时针行动顺序旋转到一个席位；可选的人脸模块随后核验当前画面是否为该席位本场注册的 `player_id`。实现、边界和待测指标见 [本场人脸身份核验 Pilot](../evaluation/stage2a-session-face-identity-pilot.md)。它属于 Part A，但不是动作识别器，也不是席位选择器。

顺序式 Part A runtime 现已把模拟转向 ACK、身份门、手势/英文语音融合、游戏合法性和下一 `acting_seat` 串成一个 betting-round 闭环。它到 `dealing_board/settled` 即停止，不能代替 Part B 或真实发牌 ACK。详见 [顺序式纵向闭环](../evaluation/stage2a-sequential-vertical-loop.md)。
