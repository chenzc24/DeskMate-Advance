# Core 共享接口

本文冻结 Stage 0A 语义，不冻结 MCU 的 JSON/二进制 framing；wire encoding 由 Stage 3 在保持 schema 语义兼容的前提下确定。机器 schema 位于 `configs/contracts/`。移除 `burn_tray` 是目标词汇的破坏性变化，当前 dealer message schema 为 v2.0；v1.0 adapter 必须在运动前因版本不匹配而拒绝连接。

## 逻辑槽位

机械目标为 9 个：`seat_a…seat_d`、`board_flop_1…3`、`board_turn`、`board_river`。Core v1 不烧牌，也没有 `burn_tray` 目标。牌面视觉槽仍为 13 个：五个 board ID 加四席各两个 hole IDs；玩家行为另有 `seat_a_action…seat_d_action` 四个固定 region。毫米位置、角度和 ROI polygon 尚未冻结，不得写入 release config。

## 发牌命令

当前项目域对象位于 `poker_dealer.domain.dealer`：

- `home(command_id, timeout_ms)`
- `rotate_to(command_id, target_slot, timeout_ms)`
- `dispense_one(command_id, timeout_ms)`
- `stop(command_id, timeout_ms)`
- `get_status(command_id, timeout_ms)`
- `reset_fault(command_id, timeout_ms)`（只允许人工安全确认后调用）

每条命令恰有一个终态 ACK：`succeeded`、`failed`、`rejected` 或 `timed_out`。ACK 带 device state/version 和传感器证据；非成功 ACK 必须带稳定错误码/原因。冻结错误词汇：`not_homed`、`invalid_target`、`deck_empty`、`feed_jam`、`double_feed`、`position_timeout`、`interlock_open`、`emergency_stop`、`transport_lost`、`protocol_error`。

协议必须支持：关联 ID、版本、CRC/帧完整性、重复请求幂等、未知命令拒绝、heartbeat/watchdog、状态查询和人工复位。软件超时不能假定马达已停止；失联时 MCU 自己也必须进入安全状态。

## 牌面观察

`CardObservation` 包含 observation ID、冻结槽位、单调时间、`confirmed|unknown|empty|face_down|face_up_unconfirmed|occluded`、可选 card、置信度、模型/校准版本、稳定帧数和质量 flags。

- `confirmed` 必须有 rank/suit。
- `unknown` 表示看见但证据不足、遮挡、失焦、域外牌等。
- `empty` 只能在空槽模型/规则有足够证据时输出；不能把读取失败当 empty。
- 一个稳定确认由多帧时序聚合产生，不由单帧最高分直接产生。
- 同一手中出现相同 `CardIdentity` 两次时，runtime 必须暂停，不能挑一个置信度较高者继续。

状态机按阶段维护每槽 `expected_empty -> delivery_pending -> present_face_down/reveal_pending -> face_up_unconfirmed -> confirmed -> cleared`；`unknown` 和 `conflict` 是保持当前预期并暂停/重观察的异常状态。机器人 ACK 只能证明出牌动作，视觉只能证明占用、朝向和可见牌面身份，二者不得互相冒充。

## 玩家行为观察

`PlayerActionObservation` 是行为模型或时序确认器的证据合同，包含 observation/hand ID、预期 state version、证据窗口起止时间、`focus_seat`、证据状态、可选动作、校准置信度、稳定时长/帧数、模型/标定版本和质量 flags。证据状态固定为 `no_action|action_start|candidate|ambiguous|occluded|out_of_roi|unknown`。

- `game` 是 `acting_seat` 的唯一权威，runtime 只把机器人已转向的该席观察窗口交给正式动作确认路径。S0-21 的 session face identity 是可选 verification evidence；`unknown/mismatch` 保持当前席并请求恢复，匹配也不能自行推进状态。
- 只有 `candidate` 可携带 `fold|check|call|bet|raise`；其余状态不得偷偷携带动作。
- `candidate` 仍不是正式动作。必须匹配当前 hand/state/seat，通过多帧与校准门槛并由 game 再次验证合法性后，adapter 才能创建 `PlayerAction`。
- 非当前席活动、旧窗口、遮挡、冲突或低置信证据不得改变状态、账本或关注席位；系统保持当前玩家并给出可替换的反馈/重试提示。
- 关注席位只在动作和账本被原子提交、`state_version` 增加后切换。具体手势、窗口、阈值、冷却及是否要求显式确认等待目标用户证据。

## 玩家动作

Stage 1 的语义动作固定为 `fold`、`check`、`call`、`bet`、`raise`。输入实现可来自 Laptop UI、physical controls、gesture/voice adapter 或 simulator。UI 只展示 game 返回的 `legal_actions`，但 game 仍需再次验证请求，非法请求不得改变任何状态或账本。

Action schema 保留 `amount_units`：Fixed-Limit Core 中必须为空，金额由街道和配置推导。动作记录包含 hand ID、seat、action ID、动作、source、接收时间、应用前后的 state version。

## Laptop 与机器人控制

`ControlObservation` 是 Laptop fallback 与未来机器人按钮的共享语义：`confirm|cancel|start|clear|next_option|previous_option`，并携带 observation/control ID、单调时间、source 和 device state version。注册 runtime 与下注 runtime 消费同一合同；机器人 wire transport 不得直接调用 gallery 或 game reducer。

- 注册阶段：`confirm` 开始当前角色人脸采集，`start` 只在 Button/SB/BB/UTG 均已注册时冻结 roster。
- Fixed-Limit 下注阶段：`next_option/previous_option` 只能在 game 返回的 `legal_actions` 内循环，`confirm` 生成当前 hand/state/acting seat 的正式动作请求。独立 Laptop pilot 只测试规则/账本；最终集成必须先通过 Part A 当前角色身份/actor gate 才打开按钮下注窗口。
- Laptop 键盘和机器人按钮只更换 adapter；状态、合法动作和数字账本仍由 runtime/game 持有。
- 旧 action window、重复 control ID 或倒退的 device state version 不得改变账本。

## 数字筹码与审计

数字账本是 Core 唯一余额权威。第一版由 Laptop 或机器人按钮/其他已验证 adapter 提交 Fixed-Limit 动作并保留线上筹码。Plan A 可增加固定下注区视觉核对，但 `observed_chip_units` 只能验证，不能覆盖账本；Plan B 的权威实体筹码仍是后续研究。余额只能由经过验证的玩家动作或带 operator ID/reason 的显式人工调整改变；动作、每席 street/hand contribution、main/side pots、余额和新 `state_version` 必须原子提交。Rebuy/补充筹码是独立审计事件，否则结束该玩家会话，不能静默修改初始 stack。

## 牌局快照与恢复

快照至少包含：hand ID/state version、street、Button/SB/BB/acting seat、四席状态、当前 street/整手投入、数字筹码、main/side pots 及 eligible seats、牌槽生命周期和已确认公共牌/摊牌底牌、待完成 command ID、合法动作、暂停原因和规则版本。未被任何对手匹配的 excess 退回；其余分层投入形成可竞争 side pots。

每手使用 append-only log 保存原始行为 evidence、被接受/拒绝的正式动作、机器人 command/ACK、牌槽观察、状态迁移、人工调整、暂停/恢复和最终结算。快照可由日志恢复，但不得通过直接编辑快照绕过证据链。

恢复只允许：重发幂等查询、补做未执行的物理动作、接受重新观察、人工作废本手。不得直接编辑赢家或跳过一个未 ACK 的发牌步骤。
