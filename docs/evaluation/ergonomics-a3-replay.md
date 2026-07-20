# Part A A3 标量回放、候选事件与评估口径

日期：2026-07-20

状态：开发契约；不代表目标摄像头验收、正式阈值或 Gate A3 关闭

## 1. 本轮闭环

本轮建立以下纯离线路径：

```text
去媒体化、拟匿名标量 JSONL
  -> 严格 schema / hash / timestamp / source 校验
  -> 八路独立 ErgonomicsRuleEngine
  -> Part A event candidate（非 UnifiedEvent）
  -> 连续行为摘要
```

回放输入只保留规则真正需要的标量、时间戳、缺失状态与模型来源。
文件中禁止图像、landmark、音频采样、真实设备序列号和直接身份信息；
真实录像派生的标量仍属于隐私数据，只能放在忽略目录或批准的 artifact
store。Git 中只允许明确标记为 `synthetic_contract_test` 的小型合成 fixture。
解析器能证明严格标量结构及 header 中的隐私声明成立，但不能仅凭声明证明
source ID 或自由文本绝无直接标识符，因此摘要明确写为
`direct_identifier_absence=declared_not_verified`。

## 2. 已冻结的 Part A 边界

- 一个文件只允许一个匿名 camera source/device 和一个单调时钟域；
- 第一行必须是 header，之后只能是严格递增且连续编号的 scalar sample；
- event config 使用加载文件的精确 SHA-256，不能只凭文件名或目录名；
- event/perception config 与 model manifest 从同一份有界、已哈希字节严格解析，
  拒绝重复字段、非有限数、过深 JSON、模型 identity 或资产路径漂移；
- feature bundle 覆盖 Pose/Face adapter、observation、亮度/音频 signal、实时音频
  接入及 Pose/Face feature 代码，避免标量生产逻辑变化后沿用旧 extractor hash；
- `ran`、缓存时间、age 和 stale 由回放器交叉校验，不能盲信记录值；
- Pose、Face、亮度和音频的 `missing/error/stale` 保持为未知证据；
- 八个事件使用独立状态 lane，同一时间可以并行 active，不在 Part A
  设优先级；
- 所有 duration 只由记录中的单调 timestamp 计算，缺失/断流区间不累计；
- Part A candidate 不包含 `suggested_action`、电机参数、舵机角度、Arduino
  命令或控制端仲裁结果；其 evidence 字段采用 v1 allowlist；
- replay 和 candidate 均有不可放大的行数、记录数和总字节硬上限，并先复制到
  匿名临时文件再解析，避免 hash 后按路径重读的 TOCTOU；
- candidate 输出只允许写到忽略的 `artifacts/`，两个输出一起暂存、失败回滚，
  且不得覆盖输入、配置、manifest 或已有硬链接。

## 3. Candidate 的语义

Candidate schema 与最终 `UnifiedEvent` 分开版本化。其生命周期仅表达检测器
自身的状态变化：

- `start`：某一路完成进入确认，开始一个新的检测 episode；
- `update`：同一 episode 的有界心跳或重要相位变化；
- `unknown`：证据缺失、过期或断流；若 episode 已 active，保留同一
  episode 且暂停 duration，不得据此 clear；
- `available`：idle lane 从 unknown 恢复到已知证据；它不是 clear，也不创建
  warning episode；
- `clear`：检测条件已连续恢复到退出时长，结束该检测 episode。

这里的 `clear` 只表示 `condition_exit_confirmed`，不表示用户已经确认提醒、
控制器已经收件、UI 已关闭或最终 `UnifiedEvent.cleared`。这些关系仍是项目
未冻结项。

当前没有 held-out participant/session 上的语义概率校准器，因此候选中的
`confidence.value` 必须为 `null`，状态明确为 `uncalibrated`；`unknown`
使用 `unavailable`。v1 对 `calibrated` fail closed；只有先版本化校准器来源
契约后才可引入。规则阈值命中、MediaPipe score、关键点有效率均不能冒充
校准后的语义正确概率。

producer 生成 candidate 临时文件后，CLI 会在提交前用独立的严格 consumer
重新解析：验证 SHA、精确 schema、固定 source/run context、单调
sequence/timestamp、逐 lane episode/confirmed-at/duration 生命周期及同一帧
重复 lane。摘要同时记录输入 hash、producer/feature/config/model hash、资产
验证状态、Git commit/dirty-state hash 和 Python/平台环境。

## 4. 连续行为指标口径

评估器按 session 和 event lane 独立流式累计，不跨 session 拼接。相邻两个
快照之间只有在时间间隔不超过最大证据间隔且两端都不是 `UNKNOWN` 时才计为
known；否则整段计入 unknown。active duration 也只累计证据连续的区间。

可报告的开发指标包括：warning entry/clear 数、known/unknown duration、
active valid duration、active 中 unknown pause、并行 active 数、cooldown
违规和 episode censoring。

- 有逐通道时间标注的 negative 区间，才可计算
  `false_triggers_per_hour`；分母必须排除 unknown、transition 和
  `do_not_score` 区间。
- 只有场景名、没有时间标注的长录像，只能报告
  `warning_entries_per_observed_hour`，不得称为误触发率。
- detection latency 只在带 `onset_ns` 与 `eligible_at_ns` 的 positive
  episode 上计算；分别保留原始延迟和扣除设计 dwell/window 后的 excess
  latency。
- aggregate `formal_effect_metric_eligible` 只有在八个 lane 都同时具备有效的
  negative FPH 分母和 positive latency 标注时才为 true；仅有八路负样本或
  仅有八路正样本都不能把整体效果口径标成完备。
- 合成 fixture 的结果必须标记为 `synthetic_contract_test`，只证明 parser、
  状态机、candidate 和公式可复现，不证明真实准确率或产品性能。

### 4.1 标注 artifact 与 CLI

正式标注使用独立的 `deskmate.ergonomics.annotations/1.0` JSON artifact。它必须：

- 由调用方提供精确 lowercase SHA-256；
- 绑定被评估 scalar replay 的精确 SHA-256 和 camera source；
- 声明 `monotonic/session_relative/ns` 时钟；
- 使用逐 lane、不重叠的半开时间区间；
- `positive` 同时提供 `onset_ns`、`eligible_at_ns` 和 `offset_ns`；
- 不携带图像、landmark、音频采样或直接身份字段。

严格读取器限制文件大小、标注条数和 JSON 深度，并拒绝重复 key、非有限数字、重复 event ID、错误 schema、错误 hash、source/replay 漂移及 lane 内区间重叠。`labeled_evidence` replay 缺少标注或标注 hash 时 fail closed；unlabeled/synthetic replay 不接受标注，避免把同一输入静默改变为正式效果证据。

```powershell
.\.venv\Scripts\python.exe scripts/ergonomics/replay_part_a.py run `
  artifacts/<session>/scalar-replay.jsonl `
  --sha256 <replay-sha256> `
  --data-status labeled_evidence `
  --annotations artifacts/<session>/annotations.json `
  --annotations-sha256 <annotation-sha256>
```

摘要回写 annotation schema、set ID、artifact/replay SHA、source 和记录数。只有标注 negative known-time 分母存在时才报告正式 FPH；只有 positive 标注存在时才报告正式 latency。整体资格仍要求八个 lane 同时满足两类指标。

## 5. 仍未冻结或仍需真实证据的项目

- robotics 最终摄像头、传输、分辨率、帧率、色彩空间和时间戳契约；
- Full/Lite 在目标摄像头上的正式选择；
- 八路正式阈值、候选 heartbeat、误触发/漏检/延迟验收线；
- calibration 与 confidence calibrator 的正式数据和方法；
- acknowledgement、condition clear、controller ack 与 UnifiedEvent lifecycle
  的映射；
- 有标注目标场景正例及长负样本结果。

标注 artifact parser、replay/source 绑定和 CLI 已完成；本项剩余的是人审真实标注及其效果结果，而不是软件入口。

因此，即使本轮所有合成契约测试通过，A3 Gate 仍保持 open。

## 6. 复现与验证

跟踪的最小 schema/provenance fixture：

```text
tests/fixtures/ergonomics/scalar-replay-valid-v1.jsonl
SHA-256: 6c4a73c3f17432c0260e7762003a6eb1e64dcadaa64a7cbb4cfecb6cdca29dc0
```

它包含三个合成 scalar snapshot：fresh evidence、latest-only cache 和显式
drop/gap。它的用途是验证 schema、hash、cache/stale 和 unknown 传播，不是
效果数据。

PowerShell 复现命令：

```powershell
$fixture = "tests/fixtures/ergonomics/scalar-replay-valid-v1.jsonl"
$fixtureSha = (Get-FileHash $fixture -Algorithm SHA256).Hash.ToLowerInvariant()

.venv\Scripts\python.exe scripts/ergonomics/replay_part_a.py validate `
  $fixture --sha256 $fixtureSha

.venv\Scripts\python.exe scripts/ergonomics/replay_part_a.py run `
  $fixture --sha256 $fixtureSha `
  --data-status synthetic_contract_test `
  --candidates artifacts/a3-replay-smoke/candidates.jsonl `
  --summary artifacts/a3-replay-smoke/summary.json --overwrite
```

预期摘要：输入 `3` 个 snapshot、累计 `dropped_before=1`；drop/gap 使八路
进入 unknown；candidate 输出为 `part-a-event-candidate/1.0`，当前小 fixture
产生 `11` 条有界 unknown 记录；连续评估必须同时显示
`contract_only=true` 和 `formal_effect_metric_eligible=false`。

完整的 all-lane 合成契约测试在临时目录生成 42 秒、250 ms 间隔的标量流，
不向 Git 写入另一份大数据副本：

```powershell
.venv\Scripts\python.exe -m pytest -q `
  tests/ergonomics/test_replay_pipeline.py
```

测试先逐路只扰动一个输入，要求目标 lane 恰好一个 `start/clear`、其他七路
不得串线；再要求八路同时 active 时 `parallel_max_active == 8`。gap 组合测试
还验证 gap 当帧不 clear、episode ID 不变，并从 26,000 ms wall duration 中
精确排除两个 250 ms unknown 区间，terminal duration 为 25,500 ms。相同输入
的 summary 和 candidate JSONL 必须逐字节一致。当前 pipeline 专项结果为
`12 passed`。

本轮 A3 replay/candidate/evaluation/output 专项为 `82 passed`，完整仓库测试为
`188 passed`。另已运行真实 CLI smoke，3-record fixture 生成 `11` 条 candidate，
consumer 复核 `11` 条且 SHA 一致；`contract_only=true`、
`formal_effect_metric_eligible=false`。这些仍只是合成契约证据。
