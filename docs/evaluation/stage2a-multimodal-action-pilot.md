# Stage 2A 多模态玩家动作 Pilot

日期：2026-07-21

当前结论：`GESTURE REVISION PASS / OFFLINE SPEECH PIPELINE PASS / HUMAN VOICE MATRIX OPEN / DEVELOPMENT ONLY`

本轮不改变五种扑克动作语义，也不改变状态机权威。它只做两项开发期可行性补充：把真人测试失败的 `Closed_Fist -> bet` 改成预训练模型已可靠支持的 `Victory -> bet`；增加离线英文封闭词表语音证据，使不使用手势的玩家仍可表达动作。

## 手势修订

| 动作 | 当前开发手势 |
| --- | --- |
| fold / 弃牌 | `Thumb_Down` |
| check / 过牌 | `Open_Palm` |
| call / 跟注 | `Thumb_Up` |
| bet / 下注 | `Victory` |
| raise / 加注 | `Pointing_Up` |

`Closed_Fist` 因真人 Laptop 测试频繁识别为 `None` 而退出映射。配置、五动作覆盖、unknown 拒绝和时序确认测试均已更新；最终交互语法仍需 participant/session 验收。

## 语音基线

| 项目 | 当前值 |
| --- | --- |
| model ID | `player-action-vosk-small-en-us` |
| 状态 | `development` |
| 模型 | `vosk-model-small-en-us-0.15`，官方目录标注约 40 MB、Apache-2.0 |
| Runtime | Vosk 0.3.45、SoundDevice 0.5.5、Python 3.13、CPU |
| 输入 | 16 kHz、mono、int16、bounded queue；不保存 PCM |
| 模型 tree SHA-256 | `57929637421baa20ff74ffb194f48e7c4a5bd0c09eac1c79a3c305ddf32db038` |
| 官方模型目录 | [Vosk Models](https://alphacephei.com/vosk/models) |
| 官方麦克风示例 | [test_microphone.py](https://github.com/alphacep/vosk-api/blob/master/python/example/test_microphone.py) |

封闭词表是 `fold/check/call/bet/raise/cancel/confirm`。七个词均已通过 Vosk 动态语法词表初始化且没有 missing-vocabulary 警告；任何词表外文本、低于 0.90、控制词或重复 cooldown 都不能成为动作候选。

## 权威、座位和融合边界

- 语音和手势都只输出 schema 1.0 `PlayerActionObservation`。
- 两者一致可形成 agreement evidence；单通道高置信可独立形成 candidate；冲突输出 `ambiguous`。
- game 继续检查 hand、state version、acting seat、confidence、supporting frames、duration 和 legal action，提交成功后才切换座位与账本。
- 语音 audit source 使用合同中已有的 `voice_adapter`；没有降低 Stage 1 的 0.90、3 evidence frames、200 ms promotion gate。
- Laptop 麦克风只在状态机给定的当前席 listening window 中开启。它能缩小时间范围，但不能证明谁说话；不使用声纹或生物身份。
- `cancel/confirm` 当前只作为控制 evidence，不直接提交 poker action；确认 UX 尚未冻结。

## 当前验证与开放项

- 两份配置可解析，五种手势和五个语音命令各自完整覆盖动作枚举。
- MediaPipe 与 Vosk 资产均离线加载并通过精确哈希验证。
- 语音模型可用限定英文语法初始化；静音 PCM 不产生 candidate。
- action/fusion 目标测试：23 passed；完整仓库：116 passed。
- Laptop 麦克风阵列 device 1 以 16 kHz mono 运行 5.50 秒：16 audio blocks、0 dropped、0 utterance（测试时未说话）、0 candidate、0 saved bytes。它证明采集链路，不代表真人英文命令准确率。
- 不保存视频或音频，不连接机器人，不触发物理动作。

尚未完成真人七词混淆矩阵、四席串话/邻席泄漏、口音/距离/噪声测试、`confirm/cancel` UX 和 held-out participant/session 指标。因此 Vosk 不能进入 `candidate/release`，也不能把现场一次识别当作模型录取。

## 运行入口

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\perception\live_action_pilot.py --index 0 --backend dshow --max-seconds 600
.\.venv\Scripts\python.exe scripts\perception\live_speech_pilot.py --list-devices
.\.venv\Scripts\python.exe scripts\perception\live_speech_pilot.py --device 1 --max-seconds 120 --emit-partials
```
