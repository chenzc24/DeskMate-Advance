# Stage 2A：四人真人验收执行包

状态：`prepared_not_executed`。本文件只说明明日如何执行，不记录或暗示任何真人结果。机器可读用例是
[`stage2a_four_player_live_acceptance_v1.json`](../../configs/evaluation/stage2a_four_player_live_acceptance_v1.json)。

## 1. 验收边界

- 四名参与者分别明确同意本场内存注册；不得替他人代为确认。
- Laptop 摄像头一次只面向当前测试者；真实机器人旋转仍未连接。
- JSONL 只保存事件、席位、动作、状态版本、拒绝原因和单调时间戳。
- 帧、音频、人脸图像和 embedding 落盘数量必须为零。
- 本次通过只证明 Laptop 四人纵向闭环，不录取模型、不冻结目标相机、不授权物理运动。

## 2. 测试前检查

1. 先运行[验收前基础设施](stage2a-prevalidation-infrastructure.md)中的设备/资产预检并创建同一个匿名 `session_group` 记录。
2. 四人按 A、B、C、D 固定登记，测试过程中不交换身份。
3. 确认摄像头为 `0`、英文麦克风为 `1`；若设备号变化，先运行各自的 device-list 命令。
4. 保证画面中只有被要求的人；多人用例除外。
5. 每个用例单独启动，完成指定现象后按 `Q`，不要在一份日志里混用多个 case。
6. 不得临场修改模型阈值。发现失败只记录，不边测边调参。

## 3. 固定正常序列

Button 为 A，因此翻牌前行动顺序固定为：

| 步骤 | 当前席位 | 中文角色 | 动作 | 版本变化 | 下一席位 |
| --- | --- | --- | --- | --- | --- |
| 1 | D | 枪口位（UTG） | call | 0 → 1 | A |
| 2 | A | 庄家位（Button） | call | 1 → 2 | B |
| 3 | B | 小盲位（SB） | call | 2 → 3 | C |
| 4 | C | 大盲位（BB） | check | 3 → 4 | 无；进入 `round_complete` |

任何拒识、身份错误或模态冲突都不得改变这里的版本序列。

## 4. 用例顺序

| Case | 目标 | 现场停止条件 |
| --- | --- | --- |
| FPA-00 | 少注册一人时四人模式不能开始 | 出现 `hand_start_blocked` |
| FPA-01 | D→A→B→C 正常完整轮 | 出现 `ROUND_COMPLETE` |
| FPA-02 | 当前期待 D 时让已注册 A 入镜并尝试动作 | 出现稳定 `seat_mismatch`，版本仍为 0 |
| FPA-03 | D 动作窗口开启后加入第二张脸 | 窗口关闭，版本仍为 0 |
| FPA-04 | D 动作窗口开启后离开画面超过 1 秒 | 窗口关闭，版本仍为 0 |
| FPA-05 | D 只说 `call`，等待后按 `C` | 按 C 前不推进，按 C 后仅推进至 A |
| FPA-06 | 在融合窗口内分别给出 `call` 与 `fold` | 冲突拒绝，版本仍为 0 |
| FPA-07 | 两分钟拿牌、摸脸、喝水等普通动作 | 全程没有状态转换 |
| FPA-08 | 每席先出现错误玩家，再由正确玩家完成动作 | 有至少四次 mismatch，最终仍按正常序列结束 |

详细操作句和机器断言以 JSON 协议为准。

## 5. 启动方式

先预览命令和输出位置，不占用设备：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\perception\run_four_player_acceptance.py FPA-01 --session-group fourp-20260722-01 --consent-confirmed --dry-run
```

正式执行单个用例：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\perception\run_four_player_acceptance.py FPA-01 --session-group fourp-20260722-01 --index 0 --backend dshow --speech-device 1 --consent-confirmed
```

若本次验收使用机器人 MJPEG 相机，以 `--stream-url` 替代
`--index/--backend`：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\perception\run_four_player_acceptance.py FPA-01 --session-group fourp-20260722-robotcam-01 --stream-url http://100.80.46.54:5000/video_feed --speech-device 1 --consent-confirmed
```

每次会生成一个新的、不会覆盖旧结果的目录：

```text
runs/stage2a_four_player_acceptance/<session-group>/<session-id>/
  FPA-01.jsonl
  FPA-01.report.json
  operator_observation.json  # live 后由现场记录命令生成
```

`runs/` 被 Git 忽略。若 live runtime 异常退出但日志已生成，仍会运行分析器并把 runtime `error` 判为失败。每个 attempt 还必须按 runner 输出的命令补齐 `operator_observation.json`；否则九 Case batch 汇总保持失败/不完整。

## 6. 单独复核日志

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\perception\analyze_four_player_acceptance.py `
  runs\stage2a_four_player_acceptance\<session-id>\FPA-01.jsonl `
  --case FPA-01 `
  --output runs\stage2a_four_player_acceptance\<session-id>\FPA-01.report.json
```

分析器检查：统一 session/hand/case 标签、完整 ready/summary、零媒体持久化、零物理机器人、必需/禁止/有序事件、精确状态转换、版本连续性，并汇总身份状态、窗口关闭原因、接受/拒绝和从身份门打开到状态提交的 P50/P95 延迟。

## 7. 现场纪律与恢复

- case 失败后先按 `Q` 保存完整 summary，再另起 session 重测；不要删除失败日志。
- 注册错误时退出当前 case 后重开，不用 `X` 掩盖已经发生的错误。
- 错误玩家或丢脸后，必须重新由当前正确玩家通过身份门，不能手动切换 focus。
- 语音识别失败记录 transcript/confidence；不要当场扩大词表。
- 若发生闪退，保留 `.jsonl` 和终端 stderr，记录发生在注册、身份门、动作窗还是轮次转换。

## 8. 明日完成定义

完成不等于九个 case 全部 PASS。明日必须交付：九份原始 JSONL、九份分析报告、失败现象摘要、参与者/session 分离说明和不含生物特征的汇总指标。任何 FAIL 都先归因到交互、身份、landmark、语音、融合或状态门，再决定调参、改交互或启动 compact TCN；不得只凭现场观感宣布训练需求。
