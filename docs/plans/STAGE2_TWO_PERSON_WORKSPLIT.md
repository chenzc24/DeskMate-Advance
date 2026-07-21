# Stage 2 双人并行分工与交接计划

状态：`执行参考 / 不扩大 Core / Gate 2A 与 Gate 2B 独立`

本计划把当前 Laptop 可先行的感知工作拆成两条端到端轨道。A 负责玩家如何表达动作并安全进入状态机；B 负责桌面牌槽里出现了什么牌以及该观察是否可信。两人复用已经冻结的 Stage 0/1 合同，但不共同编辑对方实现，也不以单次现场演示代替模型录取证据。

## 1. 当前起点与共同边界

- Stage 1 软件 oracle 已完成，可提供 acting-seat、legal-actions、state-version、账本、牌槽 lifecycle 和 recorded replay 真值。
- Stage 2A 已有 Laptop 手势、英文七词语音和四席固定 ROI 开发基线。`Closed_Fist -> bet` 已因真人失败证据替换为 `Victory -> bet`；action/fusion/multiseat 目标测试当前为 29 passed。四席象限是 Laptop fixture，不是目标桌面几何。
- Stage 2A 尚未完成真人七词/五动作矩阵、四席串扰、held-out participant/session 指标或目标相机 Gate；MediaPipe/Vosk 都仍是 `development`。
- Stage 2B 尚未启动。Laptop 可以先做单牌 fixed-ROI、牌角归一化、分类和拒识预研，但不能替代最终俯视相机、牌桌和四方向证据。
- 两条轨道都只产生 observation evidence。游戏规则、轮到谁、动作是否合法、牌局推进、赢家和账本变化始终由确定性 game engine 决定。
- 不保存身份媒体到 Git；原始视频、图片和音频只能进入 ignored data paths。当前 Pilot 默认不保存摄像头帧或麦克风 PCM。
- 本阶段不连接机器人，不授权实体运动。

## 2. A 轨：玩家动作与多模态输入

### 2.1 责任范围

A 负责 `手势/语音 -> PlayerActionObservation -> 融合/拒绝 -> Stage 1 合法性复核` 的完整闭环：

1. 维护五种动作的开发期手势语法：
   - `Thumb_Down -> fold`
   - `Open_Palm -> check`
   - `Thumb_Up -> call`
   - `Victory -> bet`
   - `Pointing_Up -> raise`
2. 维护英文封闭词表 `fold/check/call/bet/raise/cancel/confirm`；语音可以独立形成高置信 evidence，但不能证明说话座位。
3. 完成手势/语音 agreement、单通道、冲突、低置信、取消、cooldown、stale state 和非当前席拒绝。
4. 维护四席验收 UI：最多四只手按固定 ROI 标注 A/B/C/D，显示当前 focus seat、原始手势、置信度、拒绝原因和待提交 candidate；后续再并入语音与 game 状态面板。
5. 完成 Laptop 真人矩阵：左右手、距离、角度、光照、不同参与者、普通拿牌/摸脸/喝水等负样本、背景谈话、邻席干扰、遮挡和取消。
6. 报告 per-action precision/recall/F1、confusion、unknown/rejection、false accepted actions per hour/hand、跨席泄漏和 P95 确认延迟。
7. 根据失败证据决定保留 canned gesture、训练 landmark classifier/compact TCN，或在 landmark 明确失败后比较小型 RGB 时序模型。

### 2.2 A 独占路径

```text
configs/perception/actions_*.json
src/poker_dealer/perception/actions/
scripts/perception/*action*
scripts/perception/*speech*
tests/perception/actions/
docs/evaluation/stage2a-*
data/manifests/action-*
```

### 2.3 A 的 handoff Gate

- 五动作与七词测试协议、grammar version 和 participant/session split 可复现。
- 手势和语音均能独立输出冻结 schema；冲突/低置信/取消不改变 game state。
- 非当前席、旧 state-version 和重复 observation 不改变 focus 或 ledger。
- 完成 Laptop 指标和失败样例摘要；没有真人矩阵时不得晋升模型状态。
- 交付 resolved config、模型/数据 hash、运行入口、指标、失败模式和回退方案。

## 3. B 轨：牌桌场景与牌面识别

### 3.1 责任范围

B 负责 `fixed slot/ROI -> occupancy/orientation/identity evidence -> CardObservation -> Stage 1 slot lifecycle` 的完整闭环：

1. 建立牌面采集和 manifest 工具，按实体牌副设计与完整 session 分组后再切分，保持原始字节不变。
2. 先用 Laptop 摄像头和固定单牌 ROI 建立图像质量下限，不等待最终机器人或俯视相机。
3. 实现桌垫基准点/单应性、牌边缘/角点、透视矫正、方向归一化和 rank/suit 角标裁剪。
4. 比较 template/ORB 下限与 MobileNetV3-Small rank/suit 双头分类候选；只有 fixed-ROI 定位失败证据充分时才引入 detector。
5. 覆盖空槽、牌背、手、阴影、反光、模糊、遮挡、错误方向和未知牌副等负样本；低置信不得猜牌。
6. 实现同一手重复 card identity、错槽、未清桌和 observation/state-version 冲突拦截。
7. 对接 13 个逻辑牌槽的 phase-aware lifecycle，并使用 Stage 1 recorded replay 验证 unknown/occluded/duplicate 不推进牌局。
8. 报告 per-rank/per-suit precision/recall/F1、confusion、unknown/rejection、per-slot stability、duplicate detection 和 P95 延迟。

### 3.2 B 独占路径

```text
configs/perception/cards_*.json
configs/calibration/card-*
src/poker_dealer/perception/cards/
src/poker_dealer/perception/table_scene/
scripts/perception/*card*
tests/perception/cards/
docs/evaluation/stage2b-*
data/manifests/card-*
```

### 3.3 B 的 handoff Gate

- source/split/view manifests 带 SHA-256，且按 physical deck design 和完整 capture session 隔离。
- 单牌 fixed-ROI baseline、质量拒绝、rank/suit 分类和 duplicate 拦截可独立运行。
- 13-slot replay 能证明 unknown、遮挡、重复、错槽和未清桌不会被登记为有效局面。
- Laptop 结果明确标记为 feasibility；没有目标相机四方向与 held-out deck/session 时不得晋升模型状态。
- 交付 resolved config、数据/模型 hash、运行入口、指标、失败模式和 target-camera 迁移清单。

## 4. 共享只读区与集成责任

并行期间双方默认只读：

```text
configs/contracts/
configs/game/
configs/table/logical_layout_v1.json
src/poker_dealer/domain/
src/poker_dealer/game/
docs/plans/POKER_DEALER_MASTER_PLAN.md
docs/stages/
models/manifest.yaml
pyproject.toml
```

- 不允许 A 直接导入 B 的实现，也不允许 B 直接导入 A 的实现；二者只通过冻结 observation schema 和 Stage 1 simulator/replay 协作。
- 指定 A 暂任本轮共享集成维护人。双方若需要新增依赖、manifest 条目或公共 runtime 变更，先在各自 handoff 中声明，由共享维护人在子轨测试通过后一次性更新。
- 共享 schema、seat/slot ID、game reducer 或状态迁移的任何变化都不是 Stage 2 局部实现，应回到 Stage 0 合同迁移并重跑所有消费者。
- `models/manifest.yaml` 中可以先登记 `development` 资产；只有各自 Gate 证据齐全后才能进入 `candidate/release`。

## 5. 一周 Laptop 并行节奏

| 时间 | A：动作感知 | B：牌面感知 |
| --- | --- | --- |
| Day 1 | 固化 Pilot grammar/词表版本与验收 UI 输入输出 | 固化采集、manifest、deck/session split 规范 |
| Day 2 | 手势+语音统一 UI，完成冲突/取消/拒识 replay | Laptop 单牌 fixed ROI、牌角和透视归一化 |
| Day 3 | 真人五动作矩阵和 no-action 负样本 | template/ORB 与 MobileNet 双头 baseline |
| Day 4 | 真人七词、距离/口音/噪声/背景谈话 | empty/back/glare/blur/occlusion/unknown |
| Day 5 | 当前席窗口、邻席干扰和 Stage 1 action replay | duplicate、错槽、未清桌和 slot lifecycle replay |
| Day 6–7 | 指标、模型选择和 A handoff bundle | 指标、模型选择和 B handoff bundle |

时间是执行顺序，不是跳过 Gate 的日期承诺。缺少参与者、实体牌副或目标相机时，应交付明确的开放项和失败证据，不得用未测假设补齐结论。

## 6. 汇合条件

两条轨道分别通过 handoff 后，才进入统一 runtime：

1. 固定同一版本的 schema、Stage 1 engine、action/card configs、模型资产和 calibration。
2. 先运行 game + action replay + card replay + simulated dealer，不连接机器人。
3. 验证 observation、正式 action、slot transition、ledger event、dealer command/ACK 都能按 hand/state version 关联。
4. 任意 unknown、冲突、重复牌、非当前席动作、非法动作或 stale evidence 均不得推进状态。
5. Gate 2A、2B 单独完成不授权实体发牌；真实运动仍等待 Stage 3 安全 Gate。
# Part A 补充交付：session face identity

Part A 独占 `src/poker_dealer/perception/identity/`、`configs/perception/face_identity_session.json`、身份 observation schema、身份 Pilot UI 与对应测试。它接收状态机只读的 `focus_seat/state_version`，只输出 `player_id/unknown/seat_mismatch` 证据；不得写入 Part B 牌面管线、游戏引擎、账本或机器人控制。注册图像与 embedding 不进入 handoff bundle，handoff 只包含源码、配置、模型哈希、无生物特征的评估统计与操作说明。
