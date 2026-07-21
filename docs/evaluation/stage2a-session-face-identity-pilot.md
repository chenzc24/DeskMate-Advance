# Stage 2A：本场人脸身份核验 Pilot

状态：`development_feasibility_only`。它验证以下链路是否可行：

```text
Laptop 摄像头
  -> YuNet 人脸检测
  -> SFace 对齐与 embedding
  -> 本场、显式同意、仅内存注册库
  -> player_id / unknown / seat_mismatch
```

## 权限边界

- `acting_seat` 仍只由确定性游戏状态机给出；机器人朝向只是该状态的物理执行结果。
- 人脸模块只核验“镜头中的人是否为本场登记到当前席位的玩家”，不得自行选择或切换席位。
- `unknown`、`ambiguous`、多人入镜和 `seat_mismatch` 均保持当前关注对象，等待重试或人工处理。
- 输出 observation 不含人脸图像、landmark 或 embedding，也不得改变动作、牌局、筹码或账本。
- 所有参与者必须先明确同意。注册模板仅存在进程内存，按 `X` 或退出进程时清除。
- 当前不含活体检测，照片/屏幕重放攻击仍是已知风险，因此本模块还不能作为安全身份认证。

## Laptop 验收步骤

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\perception\live_face_identity_pilot.py --index 0 --backend dshow --consent-confirmed --max-seconds 600
```

窗口按键：

- `1`–`4`：模拟状态机把关注席位切到 `seat_a`–`seat_d`；切换会清空时序确认窗口，但不会删除注册库。
- `E`：为当前席位注册默认 ID `player_a`–`player_d`，连续采集 5 个合格样本。
- `X`：立刻清空整个本场注册库。
- `Q` / `Esc`：退出并清空注册库。

每次注册时镜头中必须恰好一张、尺寸合格的人脸。注册后，匹配需要达到余弦相似度阈值、与第二名的 margin，并连续稳定 5 帧且至少 300 ms。阈值 `0.45/0.08` 是保守 Pilot 默认值，不是最终校准结果。

## 当前模型与待补证据

模型资产由 `models/manifest.yaml` 固定哈希，运行期禁止下载：

- YuNet `face_detection_yunet_2023mar`：人脸框与五点 landmark。
- SFace `face_recognition_sface_2021dec`：对齐人脸 embedding。

进入 candidate 前必须按“参与者 + 完整 session”拆分数据，报告 false match、false non-match、unknown rejection、多人入镜、跨席位泄漏、不同头姿/光照/遮挡、旋转停止后的确认延迟，以及照片和屏幕重放攻击。需另行决定活体策略、阈值、注册 UX、不同玩家换位规则和 mismatch 恢复流程。

本 Pilot 不保存任何帧，不能代替目标机器人摄像头、真实旋转视角或四人实测。
