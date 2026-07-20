# DeskMate Advance 总体实施计划

状态：总体参考 / 当前权威计划
需求来源：`DeskMate_Advance_Proposal (1).pptx`
适用范围：感知、模型训练、时序判断、统一事件、部署及控制端对接

## 1. 先回答：第一步是不是预训练模型选择？

**不是严格意义上的第一步，但它是第一个技术实施阶段。**

正确顺序是：

1. **Stage 0：功能、事件和验收冻结。**先明确系统需要识别什么、什么情况才算成立、模型要向谁输出什么。
2. **Stage 1：预训练模型选择与实机输入评测。**根据 Stage 0 的功能和目标硬件选择 Pose、Face、Hand、Phone 等组件。
3. 再建立特征流水线、采集训练数据和训练自定义时序模型。

预训练模型选择不能跳过 Stage 0，原因是：

- 同一“提醒功能”可能需要组合多个观测量，而不是单个模型类别；
- 手势训练数据依赖固定版本的手部关键点提取器，提取器未冻结就采集数据会造成训练/部署不一致；
- 模型输出必须对齐统一事件和控制端消费方式，而不是直接输出电机动作；
- 目标运行环境可能是 laptop 或 Raspberry Pi 5，选型必须同时满足功能、准确性和实时性。

## 2. 总体目标与成功定义

DeskMate Advance 的核心目标不是展示若干孤立模型，而是实现一条可重复运行的闭环：

```text
摄像头 / laptop 麦克风
        ↓
预训练感知模型与程序化信号
        ↓
统一、带时间戳的特征
        ↓
规则或自训练时序模型
        ↓
平滑、确认、持续时间、迟滞与冷却
        ↓
UnifiedEvent
        ↓
控制器安全决策
        ↓
移动、舵机、显示与电脑声音
```

项目成功需要同时满足：

- 核心功能有明确的输入、事件和验收方式；
- 每个预训练组件都在目标摄像头/设备上经过证据驱动的选择；
- 至少一个自训练时序模型完成数据、训练、评估、部署闭环；
- 帧级输出不会直接触发机器人动作；
- 模型和控制器可以通过回放器、事件模拟器独立测试；
- 最终演示不依赖网络下载，并保留安全停止与失败降级。

## 3. 范围与优先级

### P0：必须形成演示闭环

- 摄像头与 laptop audio 输入可用；
- Pose / Face / Hand / Phone 预训练感知组件完成选型和封装；
- 长时间静坐或静止提醒；
- 手机持续使用提醒；
- 动态手势识别，用于计时器控制或提醒确认；
- 统一事件、时间门控、冷却和日志；
- laptop 运行路径；
- 与控制器的事件级联调；
- 录制回放、事件模拟和备份演示。

### P1：核心稳定后完成

- 不良姿态与屏幕距离提醒；
- 头部方向、眨眼率等可解释衍生功能；
- Raspberry Pi 5 运行评估或部署；
- 动态手势跨用户优化；
- 手机小目标/遮挡失败时的检测器微调。

### P2：仅在 P0/P1 冻结后考虑

- 用户身份识别和个性化问候；
- off-task 多模态分类；
- YAMNet 音频类别识别；
- 用户级姿态、距离和使用习惯校准；
- 两轮底盘、机械臂等机械扩展。

## 4. 功能—模型—事件总表

| 功能 | 优先级 | 主要输入 | 首选感知/方法 | 时序判断 | 输出事件 | 是否需要自训练 |
| --- | --- | --- | --- | --- | --- | --- |
| 长时间静坐 | P0 | Pose 关键点、运动量、计时器 | MediaPipe Pose Landmarker + 位移/方差 | 静止状态持续时间、进入/退出阈值 | `static_too_long` | 否；规则失败才训练 |
| 不良姿态 | P1 | 肩、颈、躯干关键点 | Pose Landmarker + 角度特征 | 平滑、持续时间、迟滞 | `bad_posture` | 条件启动 |
| 屏幕过近 | P1 | Face 关键点、头部尺度/姿态 | MediaPipe Face Landmarker + 个体校准 | 连续超阈值确认 | `screen_too_close` | 否 |
| 低眨眼率 | P1 | 眼部关键点/blendshape、计时窗口 | Face Landmarker + 眼部状态统计 | 长窗口计数与有效帧门控 | `low_blink_rate` | 否 |
| 手机持续使用 | P0 | 手机框、置信度、焦点计时器、头部方向 | 轻量目标检测器；PPT 候选 EfficientDet-Lite0 | 多帧确认、持续时间、冷却 | `phone_usage_detected` | 仅检测失败时微调 |
| 动态手势 | P0 | `T × 21 × 3` 手部关键点序列 | MediaPipe Hand Landmarker + 小型 TCN | 滑窗、投票、unknown、冷却 | `gesture_detected` | **是，主训练任务** |
| 环境亮度 | P1 | 视频帧亮度统计 | OpenCV/数值统计 | 时间平均与上下阈值 | `environment_too_bright` / `environment_too_dark` | 否 |
| 环境音量 | P0/P1 | laptop 音频窗口 | RMS/相对 SPL | 窗口平均与持续时间 | `noise_too_high` | 否 |
| 音频类别 | P2 | Log-Mel/音频窗口 | 预训练 YAMNet | 多窗口确认 | 扩展音频事件 | 否或轻量适配 |
| 用户身份 | P2 | 对齐人脸 | YuNet + SFace embedding | 相似度、质量和多帧确认 | `user_recognized/unknown_user` | 否 |
| off-task 行为 | P2 | 手机、头向、静止、计时器、历史事件 | 规则融合；数据充分后 MLP/TCN/GRU | 多模态时间窗口 | `off_task_behaviour` | 条件启动 |
| 焦点计时器 | P0 | 程序状态、手势事件 | 程序逻辑 | 状态机 | 控制/状态事件 | 否 |

原则：**先用预训练感知和可解释时序逻辑完成闭环，再用真实失败证据决定是否训练第二个模型。**

## 5. Stage 0：功能、场景、事件与验收冻结

### 当前冻结状态（2026-07-20）

Gate 0 状态：**部分冻结，允许 Stage 1 候选调研和 benchmark 准备，但尚未正式通过。**

| 项目 | 状态 | 当前决定或待决事项 | 对后续阶段的影响 |
| --- | --- | --- | --- |
| 动态手势类别 | 部分冻结 | 第一版仍以 `wave`、`swipe`、`circle`、`no_gesture` 为候选；到具体控制指令的映射稍后配置 | 不阻塞 Hand 候选评测；正式数据采集前必须冻结动作标签边界 |
| 手机事件语义 | 已冻结 | `phone_usage_detected` 表示有证据表明用户正在使用手机，不等同于手机仅出现在画面中 | Phone 检测需与持续时间、Hand 及可选头向证据组合 |
| 静坐/提醒时间 | 已冻结为接口 | 触发时间、清除时间和 cooldown 均通过配置暴露；演示值与正常使用值可不同 | 不阻塞模型选择；Stage 3 再用证据校准默认值 |
| 普通语义事件关系 | 已冻结 | 静坐、手机和手势事件可并行 active，模型侧不设相互优先级 | 每类事件使用独立生命周期和 cooldown；物理动作冲突仍由控制器仲裁 |
| 正式模型运行端 | 已冻结 | laptop 为 P0 正式运行端 | Stage 1/6 的性能门以 laptop 为主 |
| 麦克风 | 已冻结 | laptop 麦克风可用 | RMS/相对 SPL 纳入 P0；YAMNet 仍为 P2 |
| 摄像头输入 | **未冻结** | 等待 robotics 团队确认最终设备、传输、分辨率、帧率、色彩空间和时间戳 | 可以使用 laptop 内置摄像头做探索，但不得据此冻结最终模型结果 |
| acknowledgement 与 cleared | **未冻结** | 需要确认用户确认、检测条件清除和控制端回执之间的事件生命周期 | 在冻结 UnifiedEvent v1 前必须解决 |
| 故障安全状态 | **未冻结** | 摄像头、模型、serial 或控制器失败后的安全状态待单独讨论 | Gate 0/7 安全验收的阻塞项 |
| 验收指标 | **未冻结** | 误触发率、漏检率、响应延迟和重复场景通过线待讨论 | 候选可先采集原始指标，但不能宣布最终入选 |

边界说明：现有 laptop 摄像头适配工作只作为独立探索输入。即使最终模式概念上与另一个项目相似，Advance 仍需重新冻结自己的输入契约，不导入 Baseline 代码或接口。

### 目标

把产品描述转成可测试的模型责任，避免模型选型、数据采集和控制端各自理解不同。

### 进入条件

- 需求 PPT 可读；
- 核心团队确认目标硬件和演示时间；
- 模型负责人、控制负责人和硬件负责人可确认边界。

### 任务

1. 冻结三个 P0 端到端场景：
   - 长时间静坐 → 提醒 → 用户确认 → 冷却；
   - 焦点会话中持续使用手机 → 重聚焦提醒 → 用户确认；
   - 手势控制计时器或确认提醒 → 控制器反馈。
2. 为每个场景定义：
   - 触发前置状态；
   - 可观测输入；
   - 最短持续时间；
   - 允许的响应延迟；
   - 低置信度/缺失输入行为；
   - 冲突事件优先级；
   - 用户确认与冷却条件。
3. 冻结事件词汇、字段、单位和 schema 版本。
4. 明确模型只输出语义事件，控制器负责机器人动作与安全。
5. 冻结目标运行路径：laptop 为 P0，Raspberry Pi 5 为 P1 或并行性能验证。
6. 将功能分为 P0/P1/P2，禁止无决策门扩张范围。

### 产物

- 功能—输入—事件—验收矩阵；
- UnifiedEvent schema v1；
- 三个 P0 场景脚本；
- 责任边界与风险清单。

建议位置：

```text
docs/contracts/event-schema.md
docs/scenarios/core-scenarios.md
configs/events.yaml
```

### 退出门 Gate 0

- 控制端确认能够消费事件而不依赖模型内部对象；
- 每个 P0 场景均有可测的成功与失败条件；
- 没有模型类别直接等同于电机指令；
- 目标设备、摄像头、麦克风和演示环境已明确。

### 验证重点

- 逐条走查三个 P0 场景，确认触发、清除、超时和冷却均有定义；
- 用两组示例事件验证模型侧和控制侧对字段、单位和语义的理解一致；
- 对照 PPT 功能表检查遗漏，并记录明确延期到 P1/P2 的功能。

### 失败处理

若团队无法统一功能含义，缩减 P0 场景，不进入模型选型和大规模采集。

## 6. Stage 1：预训练模型选择与目标环境评测

执行状态：**ACTIVE，自 2026-07-20 开始。**当前只开放候选确认、环境兼容性检查、统一指标设计和非最终摄像头探索；模型资产下载与最终选型等待相应输入和验收条件具备后执行。

### 目标

选择稳定、轻量、可离线运行的感知组件，并冻结版本、输入输出和资产哈希，为后续特征和训练数据提供稳定基础。

### 进入条件

- Gate 0 通过；
- 获得目标摄像头短视频、典型桌面距离和 laptop 音频样本；
- 明确 laptop/Pi 5 的优先级和可接受帧率。

### 组件初选

| 模块 | PPT 首选候选 | 需要验证的输出 | 选择重点 |
| --- | --- | --- | --- |
| Pose | MediaPipe Pose Landmarker | 身体关键点、有效性 | 坐姿可见率、遮挡、肩颈稳定性、CPU 延迟 |
| Face/Eye | MediaPipe Face Landmarker | 人脸/眼部关键点、blendshape、头部信息 | 近远距离、侧脸、眼镜、低光、有效帧率 |
| Hand | MediaPipe Hand Landmarker/cvzone | 21 点、左右手、置信度 | 手离开/重入、快速运动、遮挡、关键点抖动 |
| Phone | 轻量 Object Detector；PPT 候选 EfficientDet-Lite0 | 类别、框、置信度 | 小目标、局部遮挡、桌面背景、输入分辨率与延迟 |
| Identity（可选） | YuNet + SFace | 人脸框、embedding、相似度 | 质量门控、误认、隐私、离线资产 |
| Audio | RMS/相对 SPL；可选 YAMNet | 音量或类别概率 | 采样设备、背景噪声、窗口长度、CPU 占用 |

### 评测任务

1. 建立同一套 benchmark harness，所有候选读取同一批录制证据。
2. 每个组件记录：
   - 版本、模型资产来源和 SHA-256；
   - 输入尺寸、预处理和线程设置；
   - 初始化时间；
   - 平均、P50/P95 推理延迟；
   - 完整循环 FPS、CPU、内存；
   - 有效输出率、漏检/失效场景；
   - 低光、距离、角度、遮挡和快速运动表现。
3. 使用目标摄像头而不是公开视频做最终选择。
4. 为每个模块保留一个明确 fallback，但不要同时维护多条主路径。
5. 固定 Hand Landmarker 版本后，才开始正式动态手势训练数据提取。

### 选择规则

按以下顺序决策：

1. P0 功能是否能被支持；
2. 目标环境有效输出率和稳定性；
3. P95 延迟和完整循环 FPS；
4. 离线可部署性、依赖复杂度和模型体积；
5. 输出是否足够可解释并能转换为项目域对象；
6. 许可证和隐私约束。

不以单张图片的主观效果或公开 benchmark 排名直接决定选型。

### 产物

```text
docs/evaluation/pretrained-selection.md
configs/perception/*.yaml
models/manifest.yaml
scripts/evaluation/benchmark_perception.py
```

### 退出门 Gate 1

- Pose、Face、Hand、Phone 至少各有一个已冻结主候选；
- 模型版本、资产哈希、预处理和输出字段可复现；
- laptop 完整感知循环达到可用实时性；
- Hand 提取器足够稳定，可以开始正式采集；
- 失败模块有明确降级或替代策略。

### 验证重点

- 在同一批目标设备录制上重复运行 benchmark，并核对结果波动；
- 人工抽查关键点、检测框和有效性标记是否与画面一致；
- 验证断网冷启动、资产哈希和配置解析；
- 对主候选与 fallback 使用同一指标表，禁止凭主观画面选择。

### 失败处理

- 先调整摄像头位置、ROI、输入分辨率和采样频率；
- 仍失败时再替换单个模块；
- Phone 只有在真实场景召回不足且具备足够框标注数据时进入微调；
- 不因一个 P1 模块失败阻塞 P0 闭环。

## 7. Stage 2：统一输入、预处理与特征流水线

### 目标

把不同框架输出转换为项目自有、带时间戳、可记录和可回放的域对象。

### 进入条件

- Gate 1 通过；
- 主候选版本和预处理配置已冻结。

### 任务

1. 建立统一 Frame/Audio packet：
   - 单调时间戳；
   - 来源、尺寸、色彩空间和采样参数；
   - freshness 与丢帧信息。
2. 预处理：BGR→RGB、ROI、缩放、归一化、音频单声道固定采样率。
3. 将框架对象转换为项目域记录：
   - `PoseObservation`；
   - `FaceObservation`；
   - `HandObservation`；
   - `DetectionObservation`；
   - `AudioObservation`。
4. 计算衍生特征：
   - 肩/颈/躯干角度；
   - 身体和手部位移、速度、方差；
   - 头部方向、距离等级、眼部状态；
   - 手机框面积、位置和置信度；
   - RMS/相对 SPL、亮度统计。
5. 保留有效性、置信度、缺失掩码和时间戳。
6. 建立录制、关键点导出和离线回放入口。
7. 队列、滑窗和日志全部有长度上限。

### 产物

```text
src/deskmate_advance/domain/
src/deskmate_advance/perception/
src/deskmate_advance/features/
scripts/runtime/record_session.py
scripts/runtime/replay_session.py
tests/perception/
tests/features/
```

### 退出门 Gate 2

- 同一录制输入重复处理得到一致特征；
- 缺失手/脸/身体不会导致崩溃或残留旧状态；
- 时间戳可用于真实持续时间计算；
- 预训练框架对象不会越过感知模块边界；
- 可在不连接机器人时完整回放并生成特征日志。

### 验证重点

- 对固定录制做重复回放，比较输出数量、时间戳和特征摘要；
- 注入丢帧、空 observation、顺序错乱和输入断开；
- 检查窗口/队列上限以及长运行内存；
- 对归一化、坐标系、单位和左右手标记编写针对性测试。

### 失败处理

若同步或时间戳不稳定，先修输入层；不得在下游用固定 FPS 假设掩盖问题。

## 8. Stage 3：可解释功能与时序规则闭环

### 目标

在训练自定义模型前，先用冻结的感知组件和可解释逻辑完成核心非训练功能，为数据采集和控制联调提供稳定事件。

### 进入条件

- Gate 2 通过；
- 核心场景录制与回放可用。

### 任务

1. 长时间静坐：运动方差 + 有效 Pose + 计时器。
2. 不良姿态：角度特征 + 个体/场景阈值 + 持续时间。
3. 屏幕距离：人脸尺度/姿态 + 初始校准 + 连续确认。
4. 眨眼率：有效眼部帧计数 + 长窗口统计。
5. 手机使用：检测置信度 + 框稳定性 + 焦点状态 + 持续时间。
6. 环境亮度/音量：窗口统计 + 上下阈值。
7. 对每个功能实现：
   - 平滑；
   - 进入阈值与退出阈值；
   - 最短持续时间；
   - unknown/invalid；
   - reminder cooldown；
   - 结构化 evidence。
8. 在录制数据上建立参数扫描，而不是在演示现场手工猜阈值。

### 产物

```text
src/deskmate_advance/temporal/
configs/temporal/*.yaml
docs/evaluation/core-events.md
tests/temporal/
```

### 退出门 Gate 3

- `static_too_long` 和 `phone_usage_detected` 能在录制场景稳定复现；
- 普通动作、短暂手机出现和短时关键点丢失不应触发持续事件；
- 所有事件均含有效 duration 和 evidence；
- 阈值可配置、可记录、可通过回放复现。

### 验证重点

- 使用正场景、短暂干扰、长负样本和缺失输入分别测试；
- 对平滑、进入/退出阈值、持续时间和 cooldown 做参数扫描；
- 报告事件级 precision/recall、误触发率和检测延迟，而非帧级准确率；
- 确认相同配置和录制能复现相同状态转换。

### 失败处理与训练触发

- 若姿态角度 + 时间逻辑跨用户持续失败，允许启动连续姿态模型；
- 若 Phone 在目标距离/遮挡下召回不足，允许启动检测器微调；
- 训练触发必须附带失败证据、数据可得性和预计联调成本。

## 9. Stage 4：数据采集与动态手势模型训练

### 目标

完成主自训练任务：基于归一化手部关键点序列的动态手势识别，并形成可部署事件。

### 进入条件

- Hand Landmarker 版本与归一化已冻结；
- 回放/特征导出可用；
- 手势语义和控制端用途已冻结；
- Gate 2 通过，最好 Gate 3 的基础事件也已可用。

### 标签范围

第一版最低标签：

- `no_gesture`；
- `wave`；
- `swipe`；
- `circle`。

只有控制语义明确且数据充足时，才拆分 `swipe_left/right` 或更多方向类别。

### 数据计划

- 5–8 名参与者；
- 每人每类 20–30 条有效序列；
- 覆盖左右手、速度、距离、角度、背景和轻度遮挡；
- `no_gesture` 至少与全部正样本同量级；
- 额外采集连续长视频评估误触发；
- 按参与者和完整会话划分，不允许相邻窗口跨 split；
- 原始视频不进入 Git，manifest 记录样本来源、参与者/会话、标签和哈希。

### 特征与增强

- 输入：`T × 21 × 3`；
- 手腕为原点、手掌尺度归一化；
- 保留左右手、时间戳、置信度和缺失掩码；
- 可加入速度、方向和轨迹派生特征；
- 增强：轻微坐标噪声、时间拉伸、随机丢帧、允许时的镜像；
- 不把不同标签的过渡段错误裁成纯正样本。

### 模型与实验顺序

1. 先训练小型 1D-TCN；
2. 先修数据、标签和窗口问题，再调网络宽度；
3. 仅在 TCN 对速度变化、长上下文或跨用户表现不足时训练 GRU 对照；
4. 用同一 held-out split 比较 macro-F1、每类 recall、误触发率、延迟和模型成本；
5. 校准置信度并保留 unknown/rejection；
6. 冻结 checkpoint、配置、标签映射、归一化和模型哈希。

### 建议初始验收线

这些是工程起点，可在试采后经团队确认调整：

- 跨用户 macro-F1 ≥ 0.85；
- 每个核心手势 recall ≥ 0.80；
- 连续 5 分钟真实负样本误触发不超过 1 次；
- 动作完成到稳定事件延迟 ≤ 1 秒；
- 模型自身 CPU 推理 ≤ 20 ms/窗口；
- 短时缺失关键点时不产生错误确认事件。

### 产物

```text
data/manifests/gesture-<dataset-id>.csv
configs/training/gesture_tcn.yaml
scripts/data/build_gesture_sequences.py
scripts/training/train_gesture.py
scripts/evaluation/evaluate_gesture.py
docs/evaluation/gesture-<run-id>.md
models/manifest.yaml
```

### 退出门 Gate 4

- 数据 manifest、split、配置和 checkpoint 全部可追溯；
- 跨用户和长负样本测试通过；
- 选定模型可由回放器实时运行；
- 标签映射和归一化在训练/部署间一致；
- 候选模型已进入 manifest，未选择权重不作为发布依赖。

### 验证重点

- 审计 participant/session 分组、重复样本和相邻窗口，排除数据泄漏；
- 固定 split 比较至少两个随机种子或记录无法重复训练的原因；
- 检查混淆矩阵、每类指标、unknown、长负样本和失败视频；
- 将训练导出与在线推理对同一序列的 logits/类别进行一致性比较；
- 在目标 CPU 上测量模型延迟与完整感知循环开销。

### 失败处理

按顺序处理：标签审计 → 补负样本/失败场景 → 窗口与归一化 → 增强 → 模型结构。不得通过扩大模型掩盖数据泄漏或错误标签。

## 10. Stage 5：统一事件、融合与行为门控

### 目标

把规则功能和学习模型统一成稳定、可解释、可版本化的事件流。

### 进入条件

- Gate 3 核心事件可用；
- Gate 4 动态手势候选可用；
- 控制端已确认 schema v1。

### UnifiedEvent 最低字段

```json
{
  "schema_version": "1.0",
  "event": "gesture_detected",
  "model_level": "advanced",
  "confidence": 0.91,
  "timestamp_ms": 123456,
  "duration_s": 0.8,
  "evidence": {
    "gesture": "wave",
    "window_frames": 24
  },
  "suggested_action": "acknowledge"
}
```

### 任务

1. 统一规则事件与模型事件格式。
2. 实现事件状态：candidate、confirmed、active、cleared、cooldown。
3. 处理优先级和冲突：安全事件优先、重复提醒合并、低置信度拒绝。
4. 所有 duration 使用时间戳计算。
5. evidence 仅保留最小可解释信息，不附带大对象或隐私帧。
6. `suggested_action` 只是建议，不越过控制器安全规则。
7. 日志包含 schema、模型、配置和 trace ID。
8. 建立事件模拟器并覆盖全部合法/非法事件。

### 产物

```text
src/deskmate_advance/events/
src/deskmate_advance/runtime/event_engine.py
scripts/runtime/simulate_events.py
tests/events/
docs/contracts/event-schema.md
```

### 退出门 Gate 5

- 模型回放和事件模拟都能驱动同一消费者；
- stale、unknown、冲突和 cooldown 行为有测试；
- schema 变更会被版本或测试发现；
- 模型层不存在电机速度、舵机角度或 Arduino 直接调用。

### 验证重点

- 对每种事件执行 candidate→confirmed→active→cleared→cooldown 状态测试；
- 注入缺字段、错误类型、未知 schema、重复和乱序事件；
- 比较回放器与事件模拟器输出是否被同一消费者一致处理；
- 检查日志能通过 trace ID 关联原始 observation、模型版本和配置。

### 失败处理

- schema 不一致时停止控制端联调，先冻结字段、单位和版本；
- 状态冲突时优先回到 `unknown`/no-event，不猜测用户意图；
- 若 `suggested_action` 被控制端当成强制动作，删除该耦合后再继续。

## 11. Stage 6：部署、性能与离线资产

### 目标

建立可重复的 laptop 主运行版本，并验证 Raspberry Pi 5 的可行性，不让设备差异改变模型语义。

### 进入条件

- Gate 5 通过；
- 候选模型和预训练资产已登记；
- 目标设备可用。

### 任务

1. 锁定 Python/运行时依赖和模型资产版本。
2. 统一 laptop/Pi 配置入口，设备差异只存在于 adapter/config。
3. benchmark：
   - 启动时间；
   - 完整循环 FPS；
   - CPU、内存；
   - 每模块 P50/P95；
   - 丢帧、stale 和有效输出率；
   - 长时间运行内存增长。
4. 需要时进行量化/导出，但必须重新验证指标和事件一致性。
5. 启动前验证所有模型文件哈希，不允许静默下载。
6. 提供一条命令启动与一条命令运行离线 smoke test。

### 建议验收线

- laptop 完整感知循环 ≥ 10 FPS；
- P95 事件处理不会造成输入队列持续增长；
- 连续运行演示时长的两倍无内存持续增长；
- 断开网络后仍可加载全部 P0 资产；
- Raspberry Pi 5 若未达标，明确保留 laptop 为正式运行端而非临场切换。

### 产物

```text
requirements*.txt / pyproject.toml
configs/runtime/laptop.yaml
configs/runtime/pi5.yaml
scripts/runtime/run_deskmate.py
scripts/evaluation/benchmark_runtime.py
docs/evaluation/runtime-targets.md
```

### 退出门 Gate 6

- laptop 运行路径稳定、离线、可复现；
- 目标资产有 manifest 和哈希；
- 设备 benchmark 有真实证据；
- Pi 5 路径只有通过性能门才进入演示依赖。

### 验证重点

- 在断网、冷启动和空缓存条件下启动完整运行时；
- 使用同一录制比较 laptop/Pi 的事件类别、时序和置信度；
- 运行至少两倍演示时长，记录内存、队列、FPS 和 P95 latency；
- 人为移除或篡改模型资产，确认启动时能明确失败而非静默替换。

### 失败处理

- Pi 5 不达标时固定 laptop 为正式运行端，不在演示前继续迁移；
- 性能不足时依次调整采样频率、ROI、输入尺寸和模块调度，再考虑量化；
- 量化或导出改变事件结果时回退到已验证格式。

## 12. Stage 7：控制端对接与端到端联调

### 目标

通过统一事件连接模型与控制器，同时确保双方可以独立开发、测试和降级。

### 进入条件

- Gate 5 事件契约通过；
- Gate 6 至少 laptop 路径稳定；
- 控制器支持模拟事件或安全 dry-run。

### 对接边界

| 模型/事件侧负责 | 控制器/硬件侧负责 |
| --- | --- |
| 语义事件、置信度、持续时间、evidence | 动作优先级、速度、距离和舵机参数 |
| unknown、stale、冷却和清除事件 | 障碍、边缘、watchdog、manual stop |
| schema 版本、日志和 trace ID | USB serial、Arduino 指令与执行反馈 |
| 回放器、事件模拟器 | 动作模拟/dry-run 与硬件安全测试 |

### 任务

1. 先用事件模拟器联调控制器，不运行模型。
2. 再用模型回放器输出事件，不连接真实电机。
3. 接入控制器 dry-run，核对事件映射与状态清除。
4. 接入显示/声音等低风险反馈。
5. 最后在操作员、低速、空旷桌面、障碍保护和急停条件下测试运动。
6. 覆盖三条 P0 场景与异常：
   - 摄像头断开；
   - 模型无输出/低置信度；
   - serial 断开；
   - 重复事件；
   - 用户无响应；
   - 安全传感器阻止动作。
7. 每次联调记录模型版本、配置、事件日志和控制端版本。

### 产物

```text
src/deskmate_advance/integration/
configs/integration/*.yaml
tests/integration/
docs/evaluation/end-to-end.md
```

### 退出门 Gate 7

- 模型和控制器可分别替换为模拟端；
- 三个 P0 场景从感知到反馈可重复运行；
- 安全传感器和 manual stop 始终能覆盖模型建议；
- 断开任一输入/通信不会导致持续危险动作。

### 验证重点

- 先运行 mock/protocol 测试，再进行 display/audio，最后才进行低速运动；
- 对每个 P0 场景记录事件、控制选择、Arduino 指令和执行反馈；
- 注入摄像头、模型、serial 和传感器故障；
- 重复测试 manual stop、watchdog、障碍和边缘保护的最高优先级。

### 失败处理

- 任一安全覆盖失败立即停止物理运动，只保留模拟联调；
- 模型与控制日志无法关联时先修 trace/版本记录，不继续调行为；
- 端到端不稳定时冻结新功能，只修事件、状态和通信阻塞问题。

## 13. Stage 8：系统验收、冻结与演示

### 目标

冻结一套可复现、可离线、可备份演示的发布候选，停止非必要扩展。

### 进入条件

- Gate 7 通过；
- P0 场景、模型、配置和硬件版本固定。

### 验收层次

1. **离线模型验收**
   - per-class precision/recall/F1；
   - confusion matrix；
   - calibration/unknown；
   - 跨用户结果；
   - 长负样本误触发率。
2. **运行时验收**
   - FPS、CPU、内存、P95 latency；
   - 丢帧、stale、关键点缺失；
   - 离线资产加载；
   - 长时间稳定性。
3. **事件验收**
   - 进入、确认、清除、冷却；
   - 重复、冲突、unknown；
   - schema 兼容与日志追踪。
4. **端到端场景验收**
   - 每条核心场景至少重复 5 次；
   - 建议至少 4/5 次得到正确语义事件和安全响应；
   - 失败案例必须记录，不通过临场修改隐藏。
5. **安全验收**
   - manual stop；
   - 障碍/边缘保护；
   - watchdog；
   - 低速和距离限制；
   - 通信/模型失败后的停止或安全状态。

### 冻结任务

- 冻结 Git commit、配置、模型 manifest 和所有资产哈希；
- 生成 model card、data card、运行说明和接口说明；
- 删除运行时对网络下载的依赖；
- 录制完整备份演示视频；
- 准备事件模拟模式，保证部分硬件失败时仍能解释系统闭环；
- Gate 7 后不再增加类别、模型或接口字段。

### 产物

```text
docs/evaluation/release-candidate.md
docs/model-cards/
docs/data-cards/
models/manifest.yaml
configs/release.yaml
README.md
```

### 退出门 Gate 8

- 一条命令可在断网环境启动核心演示；
- 代码、配置、模型和数据证据可追溯；
- 备份视频和事件模拟模式可用；
- 演示版本不再接受非阻塞变更。

### 验证重点

- 从干净环境按 README 执行安装、资产校验、smoke test 和启动；
- 核对 release config、Git commit、manifest 和模型哈希；
- 按最终顺序完整演练现场演示与模拟/视频备份；
- 让非开发成员按文档运行一次，记录未写明的依赖和操作。

### 失败处理

- 无法离线冷启动时不得冻结发布；
- 核心场景未达到重复验收线时回退到最后一个通过 Gate 7 的版本；
- 演示前只接受阻塞问题修复，任何扩展进入后续版本清单。

## 14. 条件训练分支

### A. 连续姿态模型

触发条件：规则法在跨用户、桌椅高度或摄像头角度变化下持续失败，且失败不能通过校准解决。

计划：

- 输入归一化身体关键点序列；
- 先尝试小型 MLP，再考虑 1D-TCN/GRU；
- 必须包含 good/bad/transition/unknown；
- 按参与者/会话划分；
- 与规则法在同一证据上比较；
- 若改进不足以抵消数据和部署成本，不进入发布。

### B. Phone 检测器微调

触发条件：冻结的预训练检测器在目标摄像头的小目标、局部遮挡或角度下召回不足。

计划：

- 先验证分辨率、ROI 和摄像头位置；
- 收集目标设备图像和真实负样本；
- 标注框并按会话划分；
- 对比微调前后召回、误报、P95 延迟和完整循环 FPS；
- 数据不足时不启动微调。

### C. Off-task 多模态模型

触发条件：P0/P1 全部冻结、标签定义一致且拥有足够连续会话数据。

计划：

- 输入手机置信度、头向、静止时间、焦点状态和历史事件；
- 先做透明规则融合或简单 MLP；
- 避免使用容易泄漏身份/场景的原始背景；
- 明确 off-task 的人工标注协议和一致性；
- 不作为核心演示依赖。

## 15. 十天交付映射

PPT 给出的总窗口为 10 天。建议映射如下：

| 日程 | 主阶段 | 当日必须得到的结果 |
| --- | --- | --- |
| Day 1 | Stage 0 | 功能、事件、验收和 P0 场景冻结 |
| Day 2 | Stage 1 | 预训练组件 shortlist、benchmark harness 和首批目标视频结果 |
| Day 3 | Stage 1–2 | 主候选冻结；统一 observation 和回放入口可运行 |
| Day 4 | Stage 2–3 | 核心特征、`static_too_long`、phone 候选事件可回放 |
| Day 5 | Stage 3–4 | 手势数据集冻结；第一版 TCN 训练报告 |
| Day 6 | Stage 4–5 | 候选手势模型和 UnifiedEvent engine 冻结 |
| Day 7 | Stage 6–7 | laptop 部署、事件模拟器和控制端 dry-run |
| Day 8 | Stage 7 | 三个 P0 场景端到端联调，完成阈值调整 |
| Day 9 | Stage 8 | 系统测试、失败记录、备份视频和 release candidate |
| Day 10 | Stage 8 | 只修阻塞问题，冻结并演示 |

如果执行从 2026-07-19 才开始，应在首日合并完成 Day 1 和 Day 2 的 shortlist，不应压缩最终系统测试与备份时间。

## 16. 关键决策门与停止规则

| 决策门 | 核心问题 | 不通过时 |
| --- | --- | --- |
| Gate 0 | 功能、事件、验收是否一致？ | 缩减 P0，不开始大规模技术工作 |
| Gate 1 | 预训练组件在目标环境是否稳定实时？ | 调输入/替换单模块，不扩展功能 |
| Gate 2 | 特征是否有时间戳、可回放、可复现？ | 修输入层，不训练 |
| Gate 3 | 可解释规则是否足够完成核心事件？ | 用失败证据决定条件训练 |
| Gate 4 | 手势模型跨用户、低误触发且可部署？ | 先修数据和窗口，不急于加大模型 |
| Gate 5 | 统一事件是否稳定、解耦且版本化？ | 停止硬件联调，先修契约 |
| Gate 6 | laptop/目标设备是否满足实时与离线？ | 固定 laptop，不临时冒险切设备 |
| Gate 7 | 端到端场景和安全是否可重复？ | 禁止扩展，只修阻塞问题 |
| Gate 8 | 发布证据、备份和离线资产是否齐全？ | 不声明完成 |

任何阶段出现以下情况必须停止扩张：

- P0 闭环尚未运行；
- 数据或标签不可追溯；
- 训练/部署预处理不一致；
- 模型输出绕过事件层或安全控制；
- 目标设备实时性没有真实测试；
- 演示仍依赖网络下载；
- 新模型没有明确消费者和验收条件。

## 17. 推荐工程目录

```text
advanced_project/
├─ configs/
│  ├─ perception/
│  ├─ temporal/
│  ├─ training/
│  ├─ runtime/
│  └─ integration/
├─ data/
│  └─ manifests/              # tracked; raw/work ignored
├─ docs/
│  ├─ contracts/
│  ├─ data-cards/
│  ├─ evaluation/
│  ├─ model-cards/
│  ├─ plans/
│  └─ scenarios/
├─ models/
│  └─ manifest.yaml           # weights ignored
├─ plan/
├─ scripts/
│  ├─ data/
│  ├─ evaluation/
│  ├─ runtime/
│  └─ training/
├─ src/deskmate_advance/
│  ├─ domain/
│  ├─ events/
│  ├─ features/
│  ├─ integration/
│  ├─ perception/
│  ├─ runtime/
│  └─ temporal/
└─ tests/
```

此目录结构是 Advance 项目独立定义，不继承 Baseline 代码包或接口。

## 18. 最终交付清单

### 需求与契约

- [ ] 功能—输入—事件—验收矩阵
- [ ] UnifiedEvent schema 和版本规则
- [ ] 三个 P0 场景脚本
- [ ] 模型/控制/硬件责任边界

### 预训练感知

- [ ] Pose/Face/Hand/Phone 选型报告
- [ ] 冻结版本、配置、资产哈希
- [ ] 目标摄像头有效率与性能证据
- [ ] 每个模块的 fallback 决策

### 数据与训练

- [ ] 数据说明、manifest 和分组 split
- [ ] 动态手势 TCN 配置、checkpoint 和评估
- [ ] 长负样本误触发测试
- [ ] model card、data card 和 manifest 条目

### 运行与事件

- [ ] 统一 observation 和特征流水线
- [ ] 回放器和事件模拟器
- [ ] 平滑、持续时间、unknown、迟滞和 cooldown
- [ ] laptop 离线启动和性能报告

### 集成与演示

- [ ] 控制器 dry-run 和 schema 测试
- [ ] 三个 P0 场景重复验收
- [ ] manual stop、watchdog 和异常测试
- [ ] 冻结 release config
- [ ] 备份演示视频

## 19. 执行原则

先冻结功能与事件，再选择预训练模型；先冻结关键点提取器，再采集训练数据；先用一个动态手势模型跑通完整闭环，再根据真实失败证据决定是否训练姿态、Phone 或 off-task 模型。
