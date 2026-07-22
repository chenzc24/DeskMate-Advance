# Stage 2A Part A Handoff Checklist

状态：`template_ready / handoff_not_ready`。机器模板是 [`stage2a_part_a_handoff.template.json`](../../configs/evaluation/stage2a_part_a_handoff.template.json)。模板中的 `null/false/open_blockers` 是有意保留的真实未完成项，不能为了交接而填写猜测值。

## 交接前必须具备

- 九 Case batch 报告，以及所有原始 JSONL、机器报告和操作员现场记录。
- 10,000+ no-action deterministic replay；若有真人长录像，还需真实 false acceptance/hour。
- source manifest、resolved split、landmark view manifest 的 SHA-256；participant/session 不跨 split。
- 每动作 P/R/F1、confusion、拒识覆盖、false acceptance/hand/hour、跨席、取消和 P50/P95 延迟。
- 最终选中的 development/candidate 模型、fallback、配置和资产/export hash。
- reference/export 一致性结果；若不训练 TCN，明确记录“基线足够”的失败证据审计。
- 已知失败模式、恢复方式和不得推进 state/ledger 的拒绝规则。
- target-camera 迁移清单；真实 robot ACK 仍属于联合 Gate，不得在 Part A handoff 中伪造。

## Bundle 禁止内容

- 原始人脸、视频、音频、embedding、真实姓名或同意记录。
- runtime 下载地址中的签名 URL、凭据或本地绝对私有路径。
- 直接 motor/GPIO/servo 参数。
- 把 face identity、手势或语音描述为 acting-seat/game authority。

## 状态晋升

`development -> candidate` 需要全部离线/真人指标和 immutable hash；`candidate -> release` 还需 target-camera、系统 replay、导出一致性及联合集成 Gate。四人单次现场演示即使全部 PASS，也不能独自触发晋升。
