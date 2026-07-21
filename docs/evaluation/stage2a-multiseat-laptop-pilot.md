# Stage 2A 四席 Laptop 手势归属 Pilot

日期：2026-07-21

结论：`FOUR-SEAT ROI ROUTING PASS / CAMERA PIPELINE PASS / HUMAN MULTISEAT MATRIX OPEN / DEVELOPMENT ONLY`

本 Pilot 在同一 Laptop 摄像头画面中检测最多四只手，并按手部质心归属到固定 `Seat A/B/C/D` ROI。席位是配置区域，不是人员身份；不使用人脸、声纹或 biometric identity。只有状态机/测试控制器当前选择的 `focus_seat` 可以进入手势时序确认，其他席位只显示原始 evidence。

## Laptop 测试布局

```text
+-------------------+-------------------+
| Seat B   [key 2]  | Seat C   [key 3]  |
|                   |                   |
+--------- gap -----+----- gap ---------+
| Seat A   [key 1]  | Seat D   [key 4]  |
|                   |                   |
+-------------------+-------------------+
```

四个 ROI 互不重叠，中央保留 4% normalized gap。落在 gap/画面外的手是 `unassigned`；同一当前席同时出现多只手输出 `unknown`，不选择其中一只猜测。该象限仅用于 Laptop 交互测试，明确标记为 `laptop_quadrant_pilot_not_target_geometry`，不能替代目标牌桌相机标定。

## 权威边界

- MediaPipe 可输出最多四只手的 label、score、centroid 和 handedness。
- `SeatRoiRouter` 仅用固定 ROI 与质心分配席位，不识别人是谁。
- 非当前席的完整合法手势不能产生当前席 candidate。
- 按 `1/2/3/4` 是 Laptop simulator 对 acting-seat focus 的切换；切换时 state version 增加并清空旧 temporal state。
- 只有当前席稳定动作才成为 `PlayerActionObservation`，之后仍需 game 检查 hand/state/seat/legal action 才能改变账本。
- 不保存视频帧，不连接机器人，不触发实体动作。

## 自动化与摄像头结果

- Part A action/speech/fusion/multiseat 目标测试：29 passed。
- 完整仓库回归：122 passed。
- 四席配置、非重叠、顺时针象限路由、中心 gap、非当前席隔离、同席多手拒绝和四手模型离线加载均通过。
- DirectShow camera 0：1280x720，nominal 30 FPS。
- 8.51 秒 headless：113 frames、0 missing、13.29 effective FPS。
- 四手模型 inference：mean 18.98 ms、P95 24.99 ms、max 36.14 ms。
- 测试时无人出现在摄像头中，所以四席真人动作矩阵仍开放，不能报告席位归属准确率。
- `frames_saved = 0`，`biometric_identity_used = false`。

## 尚未关闭的 Gate

- 真人把手分别移动到 A/B/C/D 并完成五动作矩阵。
- 同时两席、三席和四席出手的检测/归属。
- 中央边界、遮挡、左右手、拿牌和普通手部活动负样本。
- 状态机自动 focus 轮转，而不是测试键盘选择。
- 真实四人牌桌 ROI、俯视/机器人相机、距离和跨席泄漏指标。
- held-out participant/session 与 target-camera admission。

## 运行入口

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\perception\live_multiseat_action_pilot.py --backend dshow --max-seconds 600 --emit-all
```

窗口按 `1/2/3/4` 选择当前席，按 `Q` 或 `Esc` 退出。
