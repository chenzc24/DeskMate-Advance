# Part B（Interaction）文件结构与 AI 工作边界

本文只帮助负责 Part B 的同事或 AI 快速找到工作区，不替代需求和总体计划。发生冲突时，按以下顺序执行：

1. `DeskMate_Advance_Proposal (1).pptx`；
2. `docs/plans/ADVANCE_PROJECT_MASTER_PLAN.md`；
3. `docs/architecture/perception-architecture.md`；
4. `configs/perception/candidates.json`；
5. 本文。

Part B 负责从 Hand/Phone 感知到语义事件候选的完整纵向闭环，不负责总 runtime、统一事件转换、机器人动作或 Part A 的人体工学实现。

## 1. 开始工作前

负责 Part B 的 AI 应先：

1. 完整阅读仓库根目录 `AGENTS.md`；
2. 查看 `git status --short --branch`，保留其他人的修改；
3. 为本次实现建立一个有界目标计划：`plan/<git-user>/<YYYY-MM-DD-target>/plan.md`；
4. 只把 Part B 独占路径列为 owned paths；共享路径保持只读；
5. 不从 Baseline 仓库复制代码、配置、标签、权重或接口假设。

## 2. Part B 独占目录

```text
src/deskmate_advance/
├─ perception/interaction/       # Hand/Phone 模型适配与项目 observation
├─ features/interaction/         # 手部序列、Phone-Hand 空间关系等纯特征
└─ temporal/interaction/         # 手势 TCN、时序确认、手机使用状态逻辑

configs/interaction/             # Part B 唯一配置入口
scripts/interaction/             # 评测、数据构建、训练、回放命令
tests/interaction/               # 与上述三个源码目录对应的测试
data/manifests/interaction-*.jsonl
docs/evaluation/interaction-*.md
```

只在当前目标确实需要时创建文件，不要一次性生成空目录、占位模块或第二套同义文档。

### `perception/interaction/`

只负责模型边界：

- 从共享 `FramePacket` 读取帧、颜色信息和单调时间戳；
- 调用冻结版本的 Hand Landmarker 和 Phone detector；
- 将 MediaPipe result 转换成 Part B 自有 observation；
- 标记 `valid`、`stale`、missing、模型版本和失败原因；
- 不在这里做手势分类、持续时间判断或机器人控制。

建议按实际需要逐步建立：

```text
observations.py       # Part B observation 数据记录
hand_landmarker.py    # Hand adapter；框架对象不得向外泄漏
phone_detector.py     # Phone adapter；只输出 phone evidence
```

### `features/interaction/`

只放确定性、可回放的纯特征转换：

- Hand `21 × 3` landmarks 的手腕中心化、尺度归一化和 handedness 处理；
- 时间间隔、missing mask、landmark gap 和窗口构建；
- phone box 与 hand landmarks 的距离、重叠、相对位置；
- 不读取摄像头、不持有 MediaPipe 对象、不发事件。

建议按需要建立：

```text
hand_sequence.py      # 单帧 landmarks → 带时间戳序列特征
phone_hand.py         # Phone-Hand 空间特征
windows.py            # 有界时间窗和缺失掩码
```

### `temporal/interaction/`

负责 Part B 的时间逻辑和学习模型：

- 轻量手势 TCN 的网络、加载和推理包装；
- `wave/swipe/circle/no_gesture/unknown` 的平滑、多帧确认和 cooldown；
- Phone + Hand + 时间窗口的手机使用融合；
- 可选 head direction 只能通过共享、版本化 observation 输入；
- 输出语义事件候选，不输出电机速度、舵机角度、Arduino 指令或最终动作优先级。

建议按需要建立：

```text
gesture_tcn.py        # 网络定义及项目自有输入/输出
gesture_postprocess.py
phone_use_state.py    # 手机使用进入、退出、持续时间和 unknown
```

## 3. 非源码目录的职责

### `configs/interaction/`

所有可调参数必须从这里解析，不能散落在脚本和模型代码中。可按功能拆为：

```text
hand_landmarker.json
phone_detector.json
gesture_tcn.json
phone_fusion.json
```

模型路径最终通过共享 manifest 解析；Part B 并行开发期间不得直接修改 `models/manifest.yaml`。配置要明确时间单位、置信度阈值、窗口、采样频率、missing/stale 规则和模型版本。

### `scripts/interaction/`

脚本只是薄入口，核心逻辑必须可从 `src/` 导入和测试。典型任务包括：

```text
benchmark_pretrained.py       # 同一录像比较 Hand/Phone 候选
build_gesture_sequences.py    # 根据已冻结 split 生成可复现 derived view
train_gesture_tcn.py
evaluate_gesture_tcn.py
replay_interaction.py         # 不接真实机器人
```

### `tests/interaction/`

至少覆盖：

- framework result 到 observation 的边界转换；
- timestamp、missing、stale 和乱序输入；
- 手部归一化与窗口确定性；
- `no_gesture`、普通手部动作和长负样本；
- phone 在手中、桌面闲置、遮挡、短暂出现和低置信度；
- head direction 缺失时手机判断仍可独立降级；
- 平滑、进入/退出、duration、cooldown 和重复候选。

### 数据和评估

- 原始视频放在忽略的 `data/raw/`，不能进入 Git；
- 先按参与者和完整 recording session 划分 train/selection/test，再生成窗口；
- tracked manifest 使用 `data/manifests/interaction-*.jsonl`；
- derived sequences 放在忽略的 `data/work/<dataset-id>/`；
- 训练输出放在忽略的 `runs/<run-id>/`；
- 紧凑、可复核的结果写到 `docs/evaluation/interaction-*.md`。

## 4. Part B 可读但不可修改的共享路径

并行阶段以下路径属于共享边界：

```text
src/deskmate_advance/domain/
src/deskmate_advance/perception/camera/
src/deskmate_advance/events/
src/deskmate_advance/runtime/
src/deskmate_advance/integration/
configs/perception/candidates.json
configs/integration/
models/manifest.yaml
pyproject.toml
docs/plans/ADVANCE_PROJECT_MASTER_PLAN.md
docs/architecture/perception-architecture.md
```

可以导入或读取已冻结契约，但不能直接修改。如果 Part B 发现共享契约缺字段，应在本轨道计划或 handoff 记录中写清：请求字段、类型、单位、默认/拒绝规则、生产者和消费者，由共享边界负责人统一处理。

Part B 不得直接导入 `perception/ergonomics/`、`features/ergonomics/` 或 `temporal/ergonomics/`。Part A 的 head direction 必须表现为可缺失的共享 observation；缺失、过期或低置信时，Part B 返回降级结果或 `unknown`，不能等待 Part A 才运行。

## 5. Part B 内部数据流

```text
FramePacket
  ├─> Hand adapter  ─> HandObservation  ─> hand features ─> gesture TCN/state
  └─> Phone adapter ─> PhoneObservation ─> phone-hand features ─> phone-use state

optional versioned HeadDirectionObservation ────────────────────────┘

gesture/phone semantic event candidates ─> 最终集成人负责的 UnifiedEvent/runtime
```

低置信、缺失、过期或互相冲突的证据必须变成 `unknown` 或 no event，不能当作确定负样本。手机检测框只表示“手机存在证据”；只有 Hand/Phone/时间融合满足条件，才能生成 `phone_usage_detected` 候选。

## 6. 推荐执行顺序

1. 用共享 fixture 验证 `FramePacket → Part B observation`，不训练；
2. 在同一录制证据上评测 Hand 和 Phone development assets；
3. 冻结 Hand extractor、预处理和 observation 字段；
4. 冻结 participant/session split 后采集并构建手势序列；
5. 训练、评估手势 TCN，同时保留 `no_gesture/unknown` 和长负样本；
6. 先做透明的 Phone-Hand 规则融合；只有失败证据充分时才微调 detector 或训练融合模型；
7. 冻结 Part B handoff bundle，停止修改共享边界，交给单一集成人。

## 7. Handoff bundle 最低内容

- 固定配置、配置 SHA-256 和所有单位；
- Hand/Phone extractor 版本、资产 SHA-256、离线加载说明和许可证状态；
- 手势 checkpoint 建议条目及 hash，但不直接编辑共享 manifest；
- observation 与事件候选 JSONL fixtures；
- 可重复的录像回放命令和预期摘要；
- per-class precision/recall/F1、confusion matrix、unknown/calibration；
- 长负样本误触发率、检测延迟、P50/P95、CPU、内存、valid/missing/stale rate；
- 明确的 fallback、已知失败条件和仍未冻结参数。

完成 handoff 不表示可以控制机器人。模型回放和事件 fixture 先由最终集成人通过契约测试，再进入 controller dry-run；任何物理运动仍受 controller 的 watchdog、障碍保护、速度/距离限制和 manual stop 管理。
