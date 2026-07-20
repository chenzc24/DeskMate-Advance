# Part A：人体工学与环境实施计划

状态：**A1 执行中；目标摄像头验收未开始**
范围：Pose、Face、亮度、音量及其衍生的人体工学事件
上位依据：`DeskMate_Advance_Proposal (1).pptx` 和 `ADVANCE_PROJECT_MASTER_PLAN.md`

## 1. 目标与边界

Part A 独立交付以下模型到事件候选闭环：

- `static_too_long`；
- `bad_posture`；
- `screen_too_close`；
- head-direction evidence；
- `low_blink_rate`；
- `environment_too_bright` / `environment_too_dark`；
- `noise_too_high`。

Part A 只消费共享 `FramePacket` 和后续冻结的音频窗口，不修改 camera、domain、runtime、events、integration 或 `models/manifest.yaml`。Part A 输出 observation 和事件候选，不输出电机命令，也不决定跨功能优先级。

## 2. 模型与训练决策

| 功能 | 首选方法 | fallback | 训练策略 |
| --- | --- | --- | --- |
| 人体关键点 | MediaPipe Pose Landmarker Full | Pose Lite | 提取器保持冻结；目标摄像头 P95 超预算或有效率无收益时切 Lite |
| 面部关键点 | MediaPipe Face Landmarker | 缺失时 unknown | 提取器保持冻结；按收益决定是否关闭 blendshapes/matrix |
| 静坐 | Pose 运动量 + timestamp 状态机 | unknown | 不训练 |
| 姿态 | 关节角/相对位置 + 校准 | unknown | 跨用户和桌椅条件下规则失败后，才训练小型 MLP/TCN |
| 屏幕距离 | Face 几何 + 工位/用户标定 | unknown | 不训练，不声称跨摄像头绝对距离 |
| 头向 | Face matrix/landmarks + 校准 | unknown | 不训练 |
| 眨眼率 | eye landmarks/blendshapes + 有效观测窗口 | unknown | 不训练 |
| 亮度 | RGB luminance statistics | unknown | 不训练 |
| 音量 | RMS / dBFS 或校准后的相对 SPL | unknown | 不训练；YAMNet 仅为 P2 声源类别扩展 |

RTX 4070 不构成 Part A 必要依赖。A1–A3 以 CPU 为基线；只有条件性 Pose 时序模型启动时才使用 GPU，仍无需租算力。

## 3. 数据流与稳定边界

```text
FramePacket
  -> Pose/Face adapter 或 luminance calculator
  -> Part A owned observation
  -> normalized features + missing mask + true timestamps
  -> per-function temporal rule/state machine
  -> Part A event candidate JSONL fixture
  -> final integrator -> UnifiedEvent

Audio window
  -> RMS/dBFS calculator
  -> Part A audio observation
  -> temporal rule/state machine
  -> Part A event candidate JSONL fixture
```

MediaPipe result、OpenCV图像、NumPy特征数组和设备句柄不得越过 adapter。Observation 必须保存 source/frame/timestamp、模型版本、valid/missing/error、原因、置信证据和实际推理时间。

## 4. 分阶段计划

### A0：共享边界就绪

输入：只读 `FramePacket`、显式本地资产路径。
任务：冻结 Part A observation 字段、配置格式和 fixture 版本。
Gate：无需修改共享路径即可对合成输入进行独立测试。

### A1：预训练组件与非模型信号

任务：

1. 验证 Pose Full/Lite 和 Face 本地资产可离线初始化；
2. 实现 VIDEO timestamp 模式的 Pose/Face adapter；
3. 实现亮度统计和 RMS/dBFS 纯计算；
4. 验证 missing/error 不会被解释成正常负样本；
5. 建立 recorded-input benchmark 入口，但当前不录制隐私媒体。

Gate：单元测试通过；资产 hash 验证通过；合成帧 smoke 可运行；目标摄像头选择保持 pending。

### A2：目标摄像头证据与特征

前置：robotics 冻结 camera device/transport/resolution/FPS/color/timestamp。
任务：采集按 participant/session 管理的短录像和长负样本，比较 Pose Full/Lite，验证 Face 在距离、侧脸、眼镜、低光和遮挡下的有效率；实现归一化角度、运动量、面部尺度、头向、眨眼和有效时间特征。

Gate：目标录像 manifest 可追溯；训练/选择/测试按参与者和 session 隔离；Full/Lite 有证据驱动的主候选/fallback结论。

### A3：可解释事件闭环

任务：为每个功能实现独立状态机，包括进入、退出、duration、hysteresis、cooldown、unknown 和 stale；持续时间全部来自 timestamp；用长负样本报告误触发率和检测延迟。

Gate：所有 Part A 功能均可通过回放独立产生事件候选，且并行 active 时互不覆盖。

### A4：条件训练

仅当 `bad_posture` 规则在冻结测试集上留下系统性跨用户失败时启动。模型输入为归一化 Pose 序列、时间间隔和缺失掩码，顺序比较 MLP、TCN；不训练原始视频模型，不微调 MediaPipe。

Gate：候选必须在相同 held-out participant/session 上优于规则基线，并同时报告 per-class precision/recall/F1、混淆矩阵、unknown 行为、误触发率和延迟；否则保留规则方案。

### A5：单人集成交付

交付固定配置、资产 hash、observation/event fixture、回放命令、测试、评估摘要、fallback 和未冻结项。集成人只消费稳定表面，不导入 Part A 内部 MediaPipe 对象。

## 5. 当前未冻结项

- 最终 robotics 摄像头及传输契约；
- Full/Lite 的目标环境选择；
- Face blendshapes/matrix 的最终启用组合；
- 屏幕距离和头向的标定流程；
- 静坐、姿态、距离、眨眼、亮度、音量的正式阈值；
- 误触发、漏检、P95延迟和有效率验收线；
- 音频是否只报告 dBFS，还是由硬件校准后报告相对 SPL；
- acknowledgement 与 condition-cleared 的最终事件生命周期。

未冻结项不会阻止 A1/A2 采集原始指标，但会阻止 release 选择和 Gate A3/A5 的最终通过。

## 6. Part A 独占路径

```text
src/deskmate_advance/perception/ergonomics/
src/deskmate_advance/features/ergonomics/
src/deskmate_advance/temporal/ergonomics/
configs/ergonomics/
scripts/ergonomics/
tests/ergonomics/
docs/evaluation/ergonomics-*.md
data/manifests/ergonomics-*.jsonl
```

任何共享路径变更均推迟到单人集成阶段；Part A 不得直接导入 Part B 实现。
