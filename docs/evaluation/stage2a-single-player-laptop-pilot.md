# Stage 2A：单人 Laptop 纵向闭环 Pilot

状态：`development_feasibility_only`。该模式用于四名参与者暂时无法到场时，先由一人在 Laptop 上验证玩家归属、手势、语音和状态提交链路。它不是单人德州扑克规则，也不能替代四人 Core 验收。

## 模式边界

启动参数为 `--player-mode single_player_pilot`。该模式：

- 要求且只允许注册一个席位；不会让同一张脸重复注册到四个席位。
- 根据所选席位自动配置临时 Button，使四人状态机的首个 `acting_seat` 正好是该注册席位。
- 当前玩家完成一次合法动作后进入明确的 `pilot_complete` 边界，不会转向未注册玩家。
- 使用 Laptop 摄像头、Laptop 麦克风和模拟 `rotate_to` ACK，不触发机器人运动。
- 保持 Core 的状态机、合法性、ActorBinding、隐私和拒绝策略；仅缩短参与者与运行长度。

## 启动

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe scripts\perception\live_sequential_part_a.py `
  --player-mode single_player_pilot `
  --index 0 --backend dshow `
  --speech-device 1 --consent-confirmed `
  --max-seconds 600
```

## 操作顺序

1. 按 `1`–`4` 任选一个测试席位。
2. 按 `E`，保持单人正脸，完成人脸注册。
3. 若测试语音，按 `V` 后说三段短英文命令，完成本场内存声纹注册。
4. 按 `S`；等待模拟转向 ACK 后的新画面稳定、人脸匹配和人体绑定。
5. 手势路径：展示一个已映射动作并保持稳定。
6. 语音路径：说 `fold/check/call/bet/raise`，再由同一人说 `confirm`；`cancel` 用于撤销。
7. 一个合法动作被接纳后，窗口显示 `pilot_complete`。按 `Q` 退出。

若测试的是首轮首位玩家，`check` 通常不是合法动作；建议使用 `call`，避免把规则层拒绝误判成模型失败。

## 能证明与不能证明的内容

单人 Pilot 可以证明 Laptop 设备和一名参与者的“注册→绑定→动作 evidence→合法性→状态提交”能够连通，也可以测试纯声纹确认、手势确认、取消、超时和遮挡拒绝。

它不能证明：跨玩家人脸/声纹拒识、邻座手归属、四席顺序、跨席泄漏、多人噪声、机器人摄像头迁移或真实运动 ACK。上述项目仍必须在后续四人目标硬件验收中完成。
