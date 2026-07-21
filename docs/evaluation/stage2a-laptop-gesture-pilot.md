# Stage 2A Laptop 手势模型 Pilot

测试日期：2026-07-21

结论：`FEASIBILITY PIPELINE PASS / DEVELOPMENT MODEL ONLY / LIVE FIVE-ACTION MATRIX OPEN`

Laptop 单摄像头、单当前席 ROI 的静态手势 baseline 已实现。官方 MediaPipe Gesture Recognizer 能离线加载，输出经过项目自有记录、ROI、五语义映射、多帧确认、release/cooldown 和 `PlayerActionObservation` schema，再由 Stage 1 引擎复核当前席与动作合法性。该结果不是四席模型录取，也没有冻结最终手势语法。

## 模型与可追溯性

| 项目 | 当前值 |
| --- | --- |
| model ID | `player-action-mediapipe-canned-gesture` |
| 状态 | `development` |
| 官方 bundle | MediaPipe Gesture Recognizer float16 latest，下载于 2026-07-21 |
| 本地文件 | `models/assets/gesture_recognizer.task`，Git ignored |
| 大小 | 8,373,440 bytes |
| SHA-256 | `97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482` |
| Runtime | Python 3.13.1、MediaPipe 0.10.35、OpenCV 5.0.0、CPU XNNPACK |
| 官方说明 | [Gesture Recognizer](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer) |
| model card | [Hand Gesture Classification model card](https://storage.googleapis.com/mediapipe-assets/gesture_recognizer/model_card_hand_gesture_classification_with_faireness_2022.pdf) |

模型清单保留 admission blockers：没有 held-out participant/session、没有五动作真人指标、没有四席目标相机证据，且该模型只识别静态手势。

## Pilot 映射与门控

| Canned gesture | Poker evidence 语义 |
| --- | --- |
| `Thumb_Down` | `fold` |
| `Open_Palm` | `check` |
| `Thumb_Up` | `call` |
| `Victory` | `bet` |
| `Pointing_Up` | `raise` |
| `None/Closed_Fist/ILoveYou` | `unknown`，不产生 Poker candidate |

2026-07-21 真人验收发现 `Closed_Fist` 经常落入 `None`。它被明确记录为失败手势，开发映射改为公开样例已正确识别的 `Victory -> bet`；这仍不是最终语法冻结。

当前未冻结阈值为：score ≥ 0.60、至少 5 帧、至少 250 ms、连续 3 帧释放、1,000 ms cooldown。它们只用于 feasibility，不得作为最终校准值。

## 可执行验证

| 验证 | 结果 |
| --- | --- |
| `pytest -q tests/perception/actions` | 23 passed（含后续语音/融合测试） |
| 配置与 model manifest 解析 | 通过 |
| 模型 SHA-256 与离线加载 | 通过 |
| 空白图像 | 无手、无 candidate |
| Schema + Stage 1 engine | 稳定 `Thumb_Up` 生成 `call` evidence；引擎合法提交后才推进 focus/ledger |
| 非当前 ROI/低分/未映射类别 | `out_of_roi/unknown`，不产生 candidate |
| 持续保持动作 | 仅发一次 candidate；释放与 cooldown 后才重新 arm |

官方 Colab 的四张原始样例均检测到手并得到正确 canned label：

| 样例 | 结果 | score | 单次冷启动 inference |
| --- | --- | ---: | ---: |
| `thumbs_down.jpg` | `Thumb_Down` | 0.773 | 25.1 ms |
| `victory.jpg` | `Victory -> bet` | 0.908 | 26.8 ms |
| `thumbs_up.jpg` | `Thumb_Up` | 0.732 | 29.4 ms |
| `pointing_up.jpg` | `Pointing_Up` | 0.820 | 17.4 ms |

这些是正向 smoke，不是本项目 participant/session 指标。

## Laptop 摄像头实测

- Windows DirectShow index 0：1280×720，报告约 30 FPS；index 1 不可用。
- Headless 8 秒：180 帧、0 missing reads、平均推理 11.63 ms、P95 18.63 ms。
- Preview 1,000 帧：0 missing reads、有效 27.34 FPS、平均推理 10.69 ms、P95 12.98 ms、最大 27.51 ms。
- 两次运行都未在画面中检测到手，所以 0 candidate；不能据此判断五动作真人准确率。
- `frames_saved = 0`；没有保存、提交或登记任何摄像头画面。

## 当前正确结论与下一 Gate

已证明：模型资产、Camera→FramePacket→MediaPipe→owned evidence→temporal confirmation→schema→game legality 的纵向链路在 Laptop 上可运行，性能足以进入交互 pilot。

尚未证明：五动作是否适合真人表达、相互混淆、普通拿牌动作误触发、不同用户/左右手/距离/光照表现，以及四席相机覆盖。下一步需要真人在镜头前按动作分别重复，并加入 no-action、拿牌、摸脸、遮挡、取消和邻席干扰。只有 participant/session split 与完整指标完成后，才能决定直接保留 canned baseline、训练静态 landmark classifier，或升级 landmark sequence + compact TCN。

复现入口：

```powershell
.\.venv\Scripts\python.exe scripts\perception\smoke_action_model.py
.\.venv\Scripts\python.exe scripts\perception\live_action_pilot.py --index 0 --backend dshow --max-seconds 60
```

实时脚本有时间/帧数上限，不保存帧，不连接机器人，也不直接产生正式 `PlayerAction`。
