# Stage 5：Core 验收与演示交付

## 目标

证明 Poker Dealer 不是一次性的 scripted trick，而是在已声明约束内可重复完成牌局、遇到问题会安全停下、由新操作员按文档也能运行的 Core 产品。

## 入口条件与验收纪律

- Gate 4 已签字，S0-07/16/18 与 board reveal 产品声明均冻结；不存在 candidate 模式冒充 release。
- Release candidate 的源码、配置、模型、相机标定、firmware、机构和操作流程全部可哈希且离线可恢复。
- 验收用 deck/session/participants 不得来自模型训练或选择集；操作员至少一名不是主要开发者。
- 先执行安全/故障见证，再开始连续手数 qualification；现场短演示不能替代验收。

## 验收前冻结

- Git commit、dirty state、Python/firmware/CAD 版本。
- `core_v1` 配置 hash、wire/schema 版本、相机校准/ROI hash。
- 模型 manifest、权重/export hash、数据/test manifest hash。
- 实体牌副 ID、桌垫、相机/镜头/支架、光照、机构 BOM 与固件。
- 演示脚本、恢复脚本、E-stop 和人工裁决原则。

## 正式验收

1. 冷启动且断网，完成启动自检和 homing。
2. 四名玩家连续完成至少 20 手牌；状态机逐席切换 action ROI，模型/adapter 确认的动作覆盖 fold、check/call、候选 bet/raise，并覆盖 Button 轮转、all-in、main/side pots、flop/turn/river、多人 showdown 和 tie。
3. 人工注入至少一次当前席动作遮挡/歧义、一次非当前席同时动作、一次牌面遮挡、一次 illegal action、一次可恢复 feeder fault 和一次通信/timeout 模拟；系统均不得错误推进、切换 attention 或修改账本。
4. 对每手保存结构化 hand log，由独立 checker 重放并核对行为 evidence、动作接受/拒绝、focus 切换、street、发牌数、card identity、赢家和数字账本。
5. 记录行为模型 per-action 指标、false accepted actions per hour/hand、跨席泄漏、unknown/拒识与 P95 确认延迟，以及发牌成功/双张/漏张/落点、牌面 precision/unknown/latency、规则错误、恢复成功率和内存。
6. 另一名非开发操作员按 runbook 完成一次启动、牌局、故障恢复和关机。

### 连续 20 手如何计数

- 只有从 setup 开始并进入 `settled`、日志可独立重放且账本守恒的手牌计入 20 手；voided/redeal 不计入已完成手数。
- 正确拒识或安全暂停不是产品错误；完成规定恢复并合法结算后可以计入。若需要作废，则恢复后重新开始该手的计数。
- 错误接受非当前席动作、错误账本/赢家、未 ACK/未确认推进、日志无法恢复或危险运动，使本次 qualification 失败；修复后必须以新 release candidate 重新开始连续计数。
- 计划故障注入可独立于 20 手 qualification 执行；若穿插执行，必须预先声明注入点和成功条件，不能把临时人工改状态当恢复。

## 工作包与交付节奏

| 工作包 | 产物 | 完成条件 |
| --- | --- | --- |
| S5.0 RC freeze | compatibility matrix、资产清单、离线安装包 | 干净机器冷启动通过，hash 全部解析 |
| S5.1 Independent checker | 牌局 log replay、ledger/card/deal verifier | 与 runtime 独立实现并对黄金日志一致 |
| S5.2 Witnessed faults | action/card/dealer/通信/E-stop 故障记录 | 每个故障进入预期安全状态并按 runbook 恢复 |
| S5.3 Qualification | 连续 20 个合法 settled hands | 覆盖矩阵满足且无失败条件 |
| S5.4 Operator handoff | 安装、开关机、reset、恢复、安全/裁决手册 | 非开发操作员独立 dry-run |
| S5.5 Presentation | 短演示脚本、fallback 和已知限制 | 使用同一 RC，不修改阈值或规则 |

## 演示叙事

现场先展示四席 Button/SB/BB/UTG 和当前行动者，演示只有当前席 ROI 的稳定动作会被接受、邻席同时动作被忽略，然后运行一手完整牌局；随后展示一次“遮住公共牌，机器人拒绝判定；移开手后继续”、一次非法动作拒绝和一次 side-pot 分配。实体筹码如摆在桌面仅为非权威道具，Core 账本以屏幕为准，机器人不识别或触碰筹码。

## Gate 5 / Core 完成定义

- 20 手牌中 0 次非当前席或未确认行为导致的状态/账本变化、0 次非法规则推进、0 次未确认发牌后推进、0 次错误赢家/账本、0 次未被拦截的重复牌。
- 机构满足冻结后的 Gate 3 可靠性和全部安全要求；E-stop/保护措施现场有效。
- 行为动作确认与牌身份确认分别达到 Gate 2A/2B；所有失败样例被归档为紧凑报告而不是只给总 accuracy。
- 所有必需资产离线可加载；新机器安装和操作步骤已 dry-run。
- simulator、recorded replay、协议 mock、独立 hand-log checker 和完整测试随交付可复现。
- 已知限制明确列出，Plus backlog 不被描述成已完成。

交付包包含：源代码、锁定配置/schema、模型 metadata 与离线资产说明、固件/CAD/BOM 引用、测试/评估摘要、操作与安全 runbook、演示脚本、恢复/人工裁决流程、Core/Plus 边界。

## 验收失败后的回退

- 规则/账本/attention 错误回 Stage 1；行为或牌面错误回 Stage 2A/2B；发牌/安全错误回 Stage 3；版本、时序或恢复错误回 Stage 4。
- 修复后重跑受影响 Gate，并生成新的 release candidate/hash；不得覆盖失败 RC 的日志和指标。
- 只有演示呈现问题且不改变语义/阈值/安全时，才允许只重做 S5.4/5.5；任何运行逻辑变化都必须回对应工程 Gate。
