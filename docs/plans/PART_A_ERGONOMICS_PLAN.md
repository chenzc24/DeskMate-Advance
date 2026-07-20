# Part A：人体工学与环境实施计划

状态：**A1 基础完成；A2 工具完成、目标摄像头证据待采集；A3 状态机、标量回放、候选契约与合成评估闭环完成，真实标注/长负样本 Gate 待完成**
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

Part A 现有单目路径只消费共享 `FramePacket` 和后续冻结的音频窗口，不修改 camera、domain、runtime、events、integration 或 `models/manifest.yaml`。移动双目实现需要由后续单人集成阶段提供版本化的同步 pair domain record；在该共享契约冻结前，Part A 不得把两个普通 `FramePacket` 自行拼接成隐式接口。Part A 输出 observation 和事件候选，不输出电机命令，也不决定跨功能优先级。

## 2. 模型与训练决策

| 功能 | 首选方法 | fallback | 训练策略 |
| --- | --- | --- | --- |
| 人体关键点 | MediaPipe Pose Landmarker Full | Pose Lite | 提取器保持冻结；目标摄像头 P95 超预算或有效率无收益时切 Lite |
| 面部关键点 | MediaPipe Face Landmarker | 缺失时 unknown | 提取器保持冻结；按收益决定是否关闭 blendshapes/matrix |
| 静坐 | Pose 运动量 + timestamp 状态机 | unknown | 不训练 |
| 姿态 | 关节角/相对位置 + 校准 | unknown | 跨用户和桌椅条件下规则失败后，才训练小型 MLP/TCN |
| 屏幕距离 | 同步标定双目 + Face 对应点 + 屏幕平面拟合 | 仅在已知固定 dock 位姿时允许单目相对距离；其余情况 unknown | 不训练；绝对距离能力必须通过独立物理量测验收 |
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

前置：robotics 冻结双目 camera device/transport/resolution/FPS/color/timestamp、刚性基线、安装位姿与同步能力。
任务：先验证预期移动包络内人脸与屏幕参考点的双目共同视野，再采集按 participant/session 管理的同步短录像和长负样本；比较 Pose Full/Lite，验证 Face 在距离、侧脸、眼镜、低光和遮挡下的有效率；实现归一化角度、运动量、双目人脸参考点、屏幕平面、头向、眨眼和有效时间特征。

Gate：目标录像 manifest 可追溯；训练/选择/测试按参与者和 session 隔离；Full/Lite 有证据驱动的主候选/fallback结论。

当前状态：feature schema、缺失掩码、真实时间运动量、Face原始几何/旋转/眨眼证据，以及同录像比较 Full/Lite/Face 的 benchmark 已完成。正式 Gate 仍等待 robotics 摄像头契约和目标录像；合成或 laptop 探索性结果不得用于关闭该 Gate。

laptop 探索性实时入口已完成：`scripts/ergonomics/live_part_a.py` 将共享 camera adapter、Pose/Face、Part A features 和2 Hz亮度统计串联，采用单线程最新帧和独立模型频率，不录制媒体。该入口用于功能观察和性能探测，不改变目标摄像头 Gate。

### A3：可解释事件闭环

任务：为每个功能实现独立状态机，包括进入、退出、duration、hysteresis、cooldown、unknown 和 stale；持续时间全部来自 timestamp；用长负样本报告误触发率和检测延迟。

Gate：所有 Part A 功能均可通过回放独立产生事件候选，且并行 active 时互不覆盖。

当前状态：八个功能已接入独立三值状态机，具有基于 timestamp 的进入/退出、数值迟滞、cooldown、stale/unknown 处理；摄像头 miss/drop 或超过 500 ms 的证据间隔会中断确认且暂停 active duration；中性校准使用有界标量和 robust median；眨眼率按有效眼部观测时间计算；麦克风为显式启用的 latest-only RMS/dBFS。实时 UI 已显示校准进度与并行状态。

离线闭环已增加 `deskmate.ergonomics.scalar-replay/1.0`：先验证外部 artifact SHA、精确 config/model/feature provenance、拟匿名 source、连续 sequence、单调 timestamp、latest-only age/stale 和不可放大的资源上限，再将不含图像、landmark 或音频采样的标量输入八路规则。`part-a-event-candidate/1.0` 仅输出独立 lane 的 `start/update/unknown/available/clear`；clear 保留最终 timestamp duration，unknown 不清除 active episode，available 只表示 idle unknown 恢复，输出不含控制建议。当前 semantic confidence 没有 held-out 校准证据，因此显式为 `null/uncalibrated`，v1 对未带版本化校准来源的 `calibrated` fail closed，不能用规则命中伪造概率。

合成回放契约已用八个单-lane 测试证明无跨路接线，再用 all-lane 测试证明并行 active 不覆盖，并用 gap 测试证明 unknown 不 clear、episode 不变且 duration 精确暂停；JSONL 逐字节可复现。producer 输出在提交前由严格 candidate consumer 按 SHA、schema、固定上下文、单调时间与逐 lane 生命周期重新解析，并与本次 replay 的 source、context fingerprint、data status 和 input hash 交叉核对。连续评估器可以流式报告 known/unknown duration、active duration、并行数、cooldown、censoring、标注负区间 FPH 和标注正例 latency；整体正式效果资格要求八个 lane 各自同时具备有效 negative FPH 分母与 positive latency 标注。合成结果强制标记 `contract_only`，无时间标注的数据只能报告 screening alert rate。开发阈值、真实目标摄像头标注正例、长负样本误触发率和检测延迟仍未冻结，因此 Gate A3 继续保持 open。

### A4：条件训练

仅当 `bad_posture` 规则在冻结测试集上留下系统性跨用户失败时启动。模型输入为归一化 Pose 序列、时间间隔和缺失掩码，顺序比较 MLP、TCN；不训练原始视频模型，不微调 MediaPipe。

Gate：候选必须在相同 held-out participant/session 上优于规则基线，并同时报告 per-class precision/recall/F1、混淆矩阵、unknown 行为、误触发率和延迟；否则保留规则方案。

### A5：单人集成交付

交付固定配置、资产 hash、observation/event fixture、回放命令、测试、评估摘要、fallback 和未冻结项。集成人只消费稳定表面，不导入 Part A 内部 MediaPipe 对象。

## 5. 移动双目屏幕距离预案

本节是候选实施预案，不代表双目硬件、测距能力或正式阈值已经冻结。其目标是在 DeskMate 移动时直接测量人脸参考点到屏幕有效平面的垂直距离，而不是测量人到机器人或相机的距离。

### 5.1 几何前提与距离语义

两台摄像头必须刚性固定、完成内参/畸变和双目外参标定、使用可验证的同步时间戳，并且在预期 DeskMate 移动范围内同时看到：

- 可在两路图像中对应的人脸参考点；
- 至少三个不共线且属于同一屏幕有效平面的参考点，工程上优先使用四个屏幕边框标记；
- 足够视差，且人脸和屏幕点没有长期落在图像边缘、遮挡区或退化几何位置。

双目在同一相机坐标系中重建人脸点 `P_face` 和屏幕平面 `(n_screen, Q_screen)`，距离定义为：

```text
distance_m = abs(dot(n_screen, P_face - Q_screen))
```

这里计算的是到屏幕平面的垂直距离，不是到屏幕中心、DeskMate 或任一摄像头的欧氏距离。首版人脸参考点候选为双眼中心；是否改用鼻梁或其他头部参考点、屏幕保护层与有效显示平面的偏移，均需在采集前冻结。

DeskMate 自身运动无需单独从距离公式中扣除，因为人脸和屏幕在同一同步双目坐标系内同时重建。但如果两者不是同一时刻共同可见，不能用不同时刻的三角化结果冒充该性质。

### 5.2 候选数据流与边界

```text
left/right camera capture
  -> bounded synchronized pair + per-camera timestamps/drop counters
  -> calibration hash verification + rectification
  -> Face correspondences ---------> stereo face reference point
  -> screen reference detection ---> stereo screen points -> robust plane fit
  -> metric distance observation + validity/quality evidence
  -> smoothing + entry/exit duration + hysteresis + cooldown
  -> screen_too_close event candidate
```

未来共享输入不能表示为两个互不关联的普通 `FramePacket`。它需要一个版本化的同步 pair domain record，至少携带左右 source/frame ID、各自采集时间戳、pair 时间戳、时间偏差、drop/missing 状态、标定版本与 source rig ID。原始 OpenCV 图像、视差图、MediaPipe 对象和设备句柄不得进入 Part A 的 observation/event 边界。

距离 observation 至少保留以下非图像证据：

- `distance_m`，仅在 metric calibration 和完整质量门通过时存在；
- 人脸参考点类型、screen-plane reference 版本和 stereo calibration SHA-256；
- 左右时间偏差、三角化/重投影误差、屏幕平面拟合残差；
- 可用人脸点数、屏幕点数、视差/深度有效性、证据时间戳和 stale 状态；
- 明确的 `valid`、`missing` 或 `error` 及原因。

只有这一新 observation 经过目标硬件验收后，`screen_too_close` 才能把 `absolute_distance_claimed` 设为 `true`。现有基于人脸面积比的 laptop 单目路径继续作为探索性开发证据，不得与 metric stereo 数值静默混用。

### 5.3 分阶段实施与 Gate

| 阶段 | 工作 | 退出 Gate |
| --- | --- | --- |
| S0 几何可行性 | 用实际 DeskMate、屏幕和用户工作区测量视场、重叠区域、遮挡、工作距离及运动包络；冻结距离参考点与误差定义 | 两路在整个声明工作区持续看到人脸与不少于三个屏幕平面点；若不成立，先改安装位姿/镜头，不写测距代码 |
| S1 双目资产 | 冻结相机型号、刚性基线、分辨率/FPS、同步方式；生成版本化内参、畸变、外参和 rectification map | 两次独立标定结果可重复，校准文件 hash 固定；验收重投影/极线误差线由团队冻结 |
| S2 离线测距原型 | 对同步 recorded pairs 完成人脸对应、屏幕参考点、三角化、平面拟合和质量拒绝；建立不含媒体的标量回放 | 在静态已知距离和角度上报告误差分布、有效率、unknown 原因与确定性重复回放，不只展示单次读数 |
| S3 移动验证 | 在独立物理参考下覆盖前后/横向移动、旋转、启停、振动、运动模糊、遮挡和标记丢失 | 报告分距离/角度/速度的 MAE、P95 绝对误差、有效率、stale/miss、P95 延迟和错误告警；不合格时回到几何/同步，而不是调事件阈值掩盖 |
| S4 事件闭环 | 将有效 metric observation 接入独立 `screen_too_close` 状态机和候选事件回放 | 长负样本 FPH、正例检测延迟、进入/退出、cooldown、unknown 暂停与恢复全部通过冻结验收线 |
| S5 集成交付 | 由最终集成人增加同步 pair/shared-domain adapter，登记校准和运行资产，验证离线启动 | 模型回放、距离 observation 模拟和控制端消费可独立测试；缺失或错误资产启动失败，不静默切换语义 |

S0–S2 不需要机器人通电运动。S3 先使用受控手动位姿或台架完成；真正的底盘运动仅在协议/mock 和静态证据通过后进行，并遵守 operator、clear area、low speed、distance limit、collision protection、watchdog 和 emergency/manual stop 要求。

### 5.4 失败策略与模型决策

- 任一摄像头掉线、pair 超时/不同步、标定 hash 不匹配、视差退化、三角化点在相机后方、人脸或屏幕点不足、平面残差超限、数据 stale 时，距离为 `unknown`，不能确认或清除告警。
- 两台摄像头都朝向人、但屏幕不在共同视野时，只能得到人到 rig 的距离，不满足本预案。
- 只有 DeskMate 位于已验证的固定 dock 位姿时，才允许显式切换到独立标识的单目相对距离 fallback；DeskMate 移动时不得沿用旧中性标定。
- 双目匹配、平面拟合和时序判断优先使用标定几何与可解释算法，不启动新的端到端距离网络。只有真实失败证据表明 Face 对应点是主瓶颈时，才评估替代关键点模型；不因存在 RTX 4070 而预先训练。
- 若共同视野在实际运动包络内无法成立，本方案判定为几何不可行，备选为前向 RGB-D 加 robotics 屏幕位姿，或增加独立朝屏幕的定位摄像头；不得用增加事件持续时间掩盖测距错误。

### 5.5 仍需 robotics/产品冻结的输入

1. 两台摄像头型号、镜头/FOV、分辨率、FPS、快门形式和硬件/软件同步能力；
2. 刚性 baseline、安装高度/俯仰/朝向以及 DeskMate 的声明移动包络；
3. 屏幕参考方式：边框 AprilTag/ArUco、自然边框特征或其他可测试目标；
4. 人脸参考点、目标工作距离范围、允许角度、metric error 和 valid-rate 验收线；
5. 标记遮挡或用户/屏幕不在共同视野时，产品是否接受 `unknown`；
6. 标定检查频率、碰撞/拆装后的失效判定和校准资产管理责任。

## 6. 当前未冻结项

- 最终 robotics 双目摄像头、刚性安装、同步、共同视野及传输契约；
- Full/Lite 的目标环境选择；
- Face blendshapes/matrix 的最终启用组合；
- 双目屏幕距离的参考点、平面标记、标定/质量门和验收误差，以及头向轴符号/方向标签；
- 静坐、姿态、距离、眨眼、亮度、音量的正式阈值；
- 误触发、漏检、P95延迟和有效率验收线；
- 音频是否只报告 dBFS，还是由硬件校准后报告相对 SPL；
- acknowledgement 与 condition-cleared 的最终事件生命周期。

未冻结项不会阻止 A1/A2 采集原始指标，但会阻止 release 选择和 Gate A3/A5 的最终通过。

## 7. Part A 独占路径

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
