# 2026-07-22 Runtime 全线 Review 收口

状态：软件问题已实现，目标相机标定、模型准入和实体机构验证仍属于后续 Gate，不能由本次软件测试代替。

## 问题与实现对应

| Review 项 | 收口实现 | 主要位置 |
| --- | --- | --- |
| Runtime 未读取权威配置 | Composition Root 启动时读取 `core_v1.json`，规则与初始筹码显式传入单手/整场 Runtime | `game/config.py`、`game/rules.py`、`runtime/live_hand_app.py` |
| 单一牌面 ROI | Profile 引用完整 13 槽几何；当前槽使用独立 ROI；多检测结果按中心点一对一绑定，越界、重叠和同槽多牌拒绝 | `perception/cards/geometry.py`、`configs/perception/card_slots_development_v1.json` |
| 未完成日志可误判通过 | Checker 默认只接纳 `SETTLED`；诊断必须显式 `--allow-incomplete` | `runtime/event_log.py`、`scripts/runtime/check_hand_log.py` |
| 身份阶段无超时 | 从转向成功起建立统一 attention deadline，覆盖画面稳定、身份核验和动作等待 | `runtime/sequential_part_a.py` |
| 冻结 roster 被丢弃 | `seat -> player_id` 进入 `HandRuntime`，身份与 ActorBinding 必须对照冻结玩家 | `runtime/hand_runtime.py`、`runtime/sequential_part_a.py` |
| ACK 审计不完整 | 原始 ACK 永不直接 accepted；关联、传感器和单调版本通过后另记 `dealer_command_completed`；发牌成功要求 deck present | `domain/dealer.py`、`game/engine.py`、`runtime/event_log.py` |
| 归属置信度未准入 | 低于 Runtime 阈值的 attributed action 不进入 Engine | `runtime/sequential_part_a.py` |
| 注册不顺滑 | 注册独立 900 秒默认 deadline；取消清除临时样本；N/P 安全换角色 | `scripts/runtime/run_hand.py`、`runtime/registration.py`、`runtime/live_perception.py` |
| 缺少跨手 Session | 整场持有 roster、筹码、Button、手牌 ID、rebuy 审计和手工清桌门；Replay/Live 支持注册一次连续多手、作废/重试/结束 | `runtime/session_runtime.py`、`runtime/session_control.py`、`runtime/live_session.py`、`scripts/runtime/run_hand.py` |
| 跨手结果只在内存 | 新增独占 Session JSONL 和独立 Checker，回查每手文件 Hash 与单手 Checker，不信任声明结果 | `runtime/session_log.py`、`scripts/runtime/check_session_log.py` |
| Laptop/机器人共用占位 ROI | Laptop 使用独立 13 槽开发配置，并提供固定截图交互标定入口；机器人几何保持待定 | `configs/perception/card_slots_laptop_development_v1.json`、`scripts/perception/calibrate_card_slots.py` |
| 暂停后不能继续 | 暂停保存恢复点；操作员确认实体/软件一致后重建 lane；冲突槽须先人工清空并审计，或整手作废 | `game/engine.py`、`runtime/hand_runtime.py` |
| 麦克风 override 锁错资源 | CLI override 先写回 Runtime Profile，再创建资源锁和 Live 感知 | `scripts/runtime/run_hand.py` |
| 游戏阶段绕开 ControlSource | C/F/Enter 与 Backspace/X 统一产生 `ControlObservation`，Hand Loop 记录并路由到感知；不直接改牌局 | `domain/controls.py`、`runtime/hand_loop.py`、`runtime/live_perception.py` |
| 旧入口容易误用 | 正式入口保持 `scripts/runtime/run_hand.py`；`HandEngine.start_predealt_fixture()` 明确为测试 fixture，`start()` 仅兼容旧测试 | `game/engine.py` |
| 核心文件职责过重 | 规则配置、Fixed-Limit 参数、hash-chain EventLog、13 槽几何和跨手 Session 已拆为独立模块；Live 控制通过 port 路由 | `game/config.py`、`game/rules.py`、`game/event_log.py`、`perception/cards/geometry.py`、`runtime/session_runtime.py` |
| 只有 check/call 纵向回放 | 新增 fold 直接结算、raise + short all-in 全链回放 | `tests/runtime/test_hand_loop_replay.py` |

## 13 槽几何的真实状态

仓库现在具备完整的数据结构、配置校验、逐槽裁剪和多牌空间绑定，但提供的几何文件是 development template，`target_geometry_validated=false`。Laptop 和机器人目标相机都必须分别完成实桌标定与 held-out 验证，才能把各自 Profile 标成已验证。未标定前，软件功能存在不等于牌面模型 Gate 已通过。

## 恢复语义

暂停不会自动猜测或继续。恢复有三条明确路径：

1. 操作员核对实体牌、机构位置和软件快照完全一致，审计后 retry；Runtime 重建当前 Part A/Part B lane，并生成新命令 ID。
2. 牌槽冲突时，操作员先物理清空指定槽并执行 `reconcile_card_slot`，所有冲突消除后才能恢复并重新观察。
3. 无法确认一致时执行 `void`；投入退回数字账本，Button 保持不变，人工收牌清桌后重发。

真实机构的 `reset_fault`、homing 和低速恢复仍由 Robotics Controller 实现；本仓库只保留语义命令、ACK、安全门和审计边界，本次没有授权实体动作。

## 验证边界

软件测试包括配置覆盖、roster 错配、身份超时、低归属置信度、ACK 关联/版本/传感器、13 槽绑定、跨手筹码与 Button、恢复审计以及多种纵向 Replay。连续 20 手、目标相机指标、牌副/参与者 held-out 数据、实体发牌 200 次与安全见证仍是 Stage 4/5 验收工作，不在本次结果中宣称通过。
