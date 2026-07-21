# Stage 3：单张发牌机构与安全控制

## 目标

把“向目标槽发一张牌”变成可测、可确认、可停止的物理能力。Stage 3 不连接真实牌局决策；先由测试脚本和协议 simulator 控制。

## 入口条件与并行边界

- 受控原型入口：S0-01/03/04/09/10/13 已有明确候选、危险分析和操作员；10 个逻辑 target/命令语义保持冻结。
- design freeze 入口：四座尺寸、drop polygons、feeder/reveal 比较、传感器/BOM 和 wire mock 已通过 Stage 00B 对应子 Gate。
- Stage 3 可与 Stage 2A/2B 并行；只共享桌面几何、相机遮挡/光照约束和 target IDs，不导入模型或 game 逻辑。
- 禁止：真人牌局驱动机构、无人运动、以电机运行时间代替传感器成功证据、让 MCU 推断牌局阶段。

## 机械建议

Core 优先采用固定中央旋转底座 + 单张 feeder，而不是移动机器人：四个玩家目标、burn 和五个 board 位置共 10 个目标，仍是可标定的有限角度。feeder 可从摩擦轮/分离片方案开始，但必须用实际多副牌测试新旧程度、厚度、静电、翘曲和表面摩擦。

S0-13 要求明确比较两类 reveal：选择性翻转通道，或独立 board reveal 模块。Hole/burn 必须 face-down，board 最终必须 face-up；若只支持人工翻牌，产品只能声明“自动分发、人工揭示”，不能声明全自动发公共牌。

最小传感器：home/零位、目标位置到位、牌仓有牌、出口过牌、double-feed/jam 可观测证据、保护罩/互锁、急停。仅靠“电机转了固定时长”不能算成功 ACK。

## 交付物

- CAD/BOM/电气图、危险点与保护罩/禁手区。
- MCU 状态机：BOOT、NOT_HOMED、READY、MOVING、DISPENSING、FAULT、E_STOP。
- 有版本、ID、CRC、ACK、heartbeat、状态查询和错误码的 wire protocol。
- 上位机 `SimulatedDealer` 与真实 transport 使用同一语义接口。
- homing、rotate、dispense、stop、fault reset 的协议/固件测试。
- Hole/burn face-down 与 board face-up 的方向控制、检测和失败恢复报告。
- 操作员 checklist、卡堵恢复、急停复位和断电流程。

## 工作包

| 工作包 | 核心产物 | Gate 前证据 |
| --- | --- | --- |
| S3.0 Hazard + interfaces | 危险分析、运动包络、10 target 坐标合同、无马达协议 mock | software/MCU 互解析和错误矩阵 |
| S3.1 Feeder bench | 单张分离、牌仓/出口/double-feed/jam 传感 | 多牌副、牌量、静电/翘曲批量结果 |
| S3.2 Positioning | home、旋转定位、软硬限位、落点校准 | 每 target 分布和 P95/P99 时延 |
| S3.3 Reveal | hole/burn 背面和 board 正面方案 | orientation、错向检测、卡堵/落点报告；或冻结 manual fallback |
| S3.4 MCU safety | 状态机、interlock、E-stop、watchdog、失联安全 | 低速见证测试和复位约束 |
| S3.5 Transport | version/ID/CRC/ACK/heartbeat/query/idempotency | 乱序/重复/损坏/断连测试 |
| S3.6 Endurance | 保护罩内循环、温升、磨损、恢复 runbook | 可靠性摘要和维护间隔候选 |

S3.1、S3.2、S3.5 可在接口冻结后并行；S3.3 会影响桌面/相机遮挡，选择前必须把样件结果回传给 Stage 2B。S3.4 的安全状态机是任何带牌运动测试的前置，不得留到最后补。

## 验证顺序

1. 不装牌、断开执行器：解析器、错误命令、CRC、重复 ID、断连/heartbeat。
2. 低速空载、有限角度：home、定位、软/硬限位、E-stop。
3. 加保护罩和单张牌：出口传感器与 ACK 时序。
4. 多副牌的批量单张测试：正常、低牌量、弯曲、静电、故意堵塞。
5. 10 个目标槽精度测试；不得有人手进入轨迹。
6. Board reveal 方向测试：正确朝向、未翻、双翻、卡堵和落点漂移。
7. 长时间循环与温升/磨损检查。

## 建议 Gate 3

- 连续 200 次 `dispense_one`：0 次双张，单张成功率 ≥ 99%；所有漏发/卡堵被传感器检测且不得误报成功。
- 每个目标槽至少 50 次定位/发牌，落入定义区域 ≥ 99%；失败可安全恢复。
- Board reveal 候选必须单独报告 face-up 成功率、方向错误率和方向错误检测率；没有证据时不得选定自动揭示方案。
- home、软件 stop、物理 E-stop、interlock 和 watchdog 均有实测视频/日志；E-stop 后不得自动复位。
- 断开上位机、乱序/重复命令、CRC 错误时机构进入或保持安全状态。
- P95/P99 动作时延被测量并用于上位机 timeout，而不是拍脑袋设置。
- 全部运动测试有操作员、清场、低速/限力和保护；没有 unattended test。

数值可在 Gate 0 根据机构现实调整，但“检测到失败才允许 ACK failed、绝不把未知当成功”的原则不可降低。

## Gate 3 交付包与回退

- CAD/BOM/electrical/firmware/protocol 版本、校准、传感器证据、批量原始摘要、故障注入和 operator runbook 可追溯。
- 真实 transport 与 `SimulatedDealer` 对相同命令序列给出语义兼容终态；所有成功 ACK 有规定传感器证据。
- feeder 或定位不稳定：退回对应 bench，不允许在上位机增加盲重试掩盖失败。
- reveal 不达标：冻结“自动分发、人工揭示”fallback，并同步产品声明/Stage 2B lifecycle；不得带着方向未知进入集成。
- 安全或失联行为不通过：停止全部实物联调，Gate 3 失败，不以较低速度替代必要保护。
