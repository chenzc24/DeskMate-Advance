# cocino_car API 与游戏状态机整合计划

状态：`已补充公共牌定时停车与 request_id 终态；实体运动未执行/未验证`

日期：`2026-07-27`

负责人：`wuqi-ii`

关联计划：

- `plan/wuqi-ii/2026-07-26-table-route-state-machine/plan.md`
- `plan/wuqi-ii/2026-07-27-navigation-motion-gap/plan.md`

## 目标

读取项目 `C:\Users\ASUS\Desktop\cocino_car` 的正式
`/api/robotics/v1/*` 接口，在 Poker Dealer 项目中完成以下工作：

1. 用有限有向图把状态机的目标座位、公共牌区目标转换为小车原子动作。
2. 将循线、定角转向、人脸对中、回归白线和停止 API 分配到相应状态机等待节点。
3. 对每个原子动作轮询完成状态，成功后才能产生状态机可接受的
   `NavigationAck`。
4. 两次连续物理移动之间保持至少 `2500 ms`。
5. 保持发牌、识别和游戏规则的权威边界；小车 API 不得直接推进游戏状态。
6. 对当前小车 API 无法安全闭环的能力拒绝成功，并在交付说明中列出所需车端修改。

## Owned paths

- `src/poker_dealer/robotics/navigation/`
- `src/poker_dealer/runtime/hand_runtime.py`
- `src/poker_dealer/runtime/sequential_part_b.py`
- `src/poker_dealer/domain/game.py`
- `tests/robotics/`
- `tests/runtime/`
- `tests/domain/`
- 本计划文件
- `C:\Users\ASUS\Desktop\cocino_car\pi_service\robot_web\control\robotics_gateway.py`
- `C:\Users\ASUS\Desktop\cocino_car\pi_service\tests\test_robotics_gateway.py`
- `C:\Users\ASUS\Desktop\cocino_car\pi_service\README.md`

## Dirty read-only paths

下列现有修改与本目标无关，不覆盖、不整理：

- `configs/runtime/network_endpoints.json`
- `scripts/perception/train_card_detector.py`
- 卡牌数据、模型与训练计划
- `src/track_line/live_line_detection.py`

小车仓库仅修改上述正式 API 请求终态文件及测试、说明；其余路径保持只读。

## 外部依赖与已核对事实

- 小车正式 API 版本：`1.0`
- 正式动作：
  `follow_line_to_end`、`face_turn_start/heartbeat/stop`、
  `line_recenter_start/stop`、`preset_turn`、`dispense_one`、`stop`
- 动作接收不等于动作完成；路线动作完成必须轮询 `/status`。
- 人脸是否居中由上位机视觉判断，小车端只执行脉冲转向和心跳保护。
- `dispense_one` 仅有 Arduino 命令接收证据，没有实体出牌传感器证据。
- 用户确认发牌机构每次调用能够保证只出一张牌；本轮据此采用
  “一次 `dispense_one` 对应一个牌槽”的前提。
- 公共牌位置按用户指定采用定时方案：调用现有巡线，1 秒后调用现有 `stop`。
- API 请求结果新增与 `request_id` 关联的 `request_status` 和 `terminal`。

## 实施范围

1. 实现只依赖 Python 标准库的 HTTP 客户端，并校验 API 版本和 capabilities。
2. 实现规范姿态图、最短合法路径和每条边对应的小车原子动作。
3. 实现 fail-closed 的小车导航适配器：
   - 检查起始姿态与版本；
   - 使用唯一且可重试的请求 ID；
   - 轮询状态；
   - 错误、超时、断联和未知状态均返回失败 ACK；
   - 每个成功原子移动后执行 2500 ms 冷却。
4. 状态机明确解析当前 Button 对固定 Seat 的角色映射：
   - 行动/Showdown：导航到玩家正面姿态；
   - 底牌：Button、SB、BB、UTG 使用规划中的发牌姿态；
   - Flop/Turn/River：导航到公共牌区并原地等待牌面识别。
5. 将底牌步骤改为每名玩家连续两张，以便一次定位后连续发两张。
6. 公共牌全部识别后进入 1000 ms 的内部等待节点，结束后才恢复巡线。
7. 小车 API 保存正在执行的路线/出牌请求，并按 `request_id` 返回
   `running/succeeded/failed/cancelled` 终态。
8. 增加纯模拟与假 HTTP 传输测试，不访问真实树莓派、不打开电机门控。

## 明确不做

- 不向树莓派发送请求。
- 不启用电机门控。
- 不声称实体运动、公共牌区停车或真实出牌已经通过验证。
- 不提交、不推送、不创建分支。

## 验证

- 路径图覆盖所有规范姿态到目标玩家的合法路径。
- 两次右转的 SB→BB、UTG→Button 路径保持约束。
- HTTP 客户端请求字段、API 版本、幂等 ID 和轮询终态测试。
- 状态机目标姿态和每人连续两张底牌测试。
- Flop 三张发完后原地等待三张牌识别的回归测试。
- 运行相关测试、可行时运行完整 Python 测试。
- `git diff --check`
- `git status --short --branch`

## Physical-motion status

`未授权、未执行、未验证`。本计划只完成协议、规划器、适配器和模拟验证。

## Commit intent

用户已于 `2026-07-27` 明确要求将本轮状态机与小车 API 整合结果推送到
`main`。只提交本计划以及关联桌面路线、导航等待计划拥有的文件；不纳入工作区
中既有的卡牌训练、模型、端点配置和巡线修改。

## 实施结果

- 已实现 cocino_car API v1.0 客户端、请求关联和超时后按 request ID 查询。
- 已实现 I 形桌面有限有向图与所有规范姿态间的合法路径。
- 已实现真实导航适配器的循线、转向、人脸对中、回线、轮询和故障停车。
- 已实现 Button→SB→BB→UTG 注册巡航和 UTG→I_E 归位。
- 已实现当前 Button 对固定 Seat 的动态物理姿态解析。
- 已将底牌改为每位玩家一次定位、连续两次单张发牌 ACK。
- 已保留 Flop 三张、Turn/River 一张发完后原地等待牌面确认的门控。
- 已实现公共牌阶段巡线 1 秒后 `stop`，识牌完成后再等待 1 秒才恢复巡线。
- 已实现小车动作按 `request_id` 查询 `request_status`、`terminal` 和完成时间；
  被显式 `stop` 中止的巡线请求会得到 `cancelled` 终态。
- 已实现 cocino_car `dispense_one` Dealer adapter；用户已明确选择不增加
  实体出牌依据，因此等待 request 终态后产生显式
  `arduino_command_ack_only` ACK；传感器字段保持未知，不伪造物理验证。
- 按用户明确选择，不增加车端姿态版本或实体出牌传感器，也不修改现有脉冲参数。

## 验证结果

- 本轮定向状态机、导航与 replay 测试：`31 passed`
- `domain/game/runtime/robotics/contracts` 范围：`293 passed, 2 failed`
- 小车正式 `pi_service` 全套：`96 passed`
- 小车路径/导入验证：通过
- `python -m compileall -q src scripts`：通过
- 本目标 owned tracked paths 的 `git diff --check`：通过
- 全工作区 `git diff --check`：被无关既有文件
  `src/track_line/live_line_detection.py:26` 的 trailing whitespace 阻断；
  本任务按只读脏路径规则未修改该文件。

当前范围中仍存在的非本目标失败：

- `scripts/game/demo_stage1.py` 直接作为子进程运行时没有把 `src` 放入模块路径。
- 用户已有 `configs/runtime/network_endpoints.json` 修改与旧测试预期 IP 不同。
