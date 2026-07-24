# Chip V2 筹码定位与 10/20 面额识别

## 结论

本轮使用 `chip_v2(1)` 的 40 张用户采集图完成了一个新的开发候选：

- YOLO 定位从现有 `hard-negative-v3` 权重低学习率微调；
- 面额范围固定为 `10` 和 `20`，运行时不再输出 `1` 或 `5`；
- 面额仍由外圈颜色、透视矫正后的数字模板和多帧确认共同决定；
- 结果仍为 `development`，没有替换正式模型、默认路径或
  `models/manifest.yaml`。

该候选只能输出感知证据，不能直接修改游戏状态、数字账本或控制机构。

## 数据快照和划分

原图被复制到忽略目录
`data/raw/chips/2026-07-24-chip-v2-source/`，保持原始字节不变。
40 张图片均可解码且没有精确重复，共人工复核 75 个筹码实例：

- 31 张斜视图：`10` 序列、`20` 序列和混合序列；
- 9 张正视图：4 张 `10`、5 张 `20`，只用于模板候选；
- 数据中没有 `1` 或 `5`，本轮没有生成或借用这两类样本。

定位数据按完整采集组先划分、再增强：

- `chip_v2:10` 和 `chip_v2:mix` 进入训练，每张原图保留一个原版并生成
  7 个确定性增强版本；
- 完整 `chip_v2:20` 序列的 13 张图、23 个实例只用于目标相机验证；
- 正视图不进入 YOLO 训练或验证；
- 原有测试集保持不变。

最终数据量为训练 2241 张、验证 442 张、测试 460 张。增强包括有限角度
旋转、轻度透视、缩放和平移、曝光/伽马/色温、阴影、反光、模糊、噪声与
JPEG 退化，不做水平或垂直翻转。

- 数据集清单 SHA-256：
  `bf088a7f856c727d4b827b00d8676e824d48442382ab307a7ff41ebed1ece32b`
- 原始快照记录 SHA-256：
  `ef8a07a441ca47581fe301eb90ee57c939e0483f59a42ff8de83faa8a785cb92`

## 定位结果

两套权重均以 960 像素、相同数据和相同评估程序复测。

| 数据 | 模型 | Precision | Recall | F1 | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 保留的斜视 `20` 序列，13 图/23 实例 | hard-negative-v3 | 0.265 | 0.391 | 0.316 | 0.200 | 0.101 |
| 保留的斜视 `20` 序列，13 图/23 实例 | chip-v2-v1 | 0.997 | 1.000 | 0.999 | 0.995 | 0.607 |
| 原有测试集，460 图/1514 实例 | hard-negative-v3 | 0.947 | 0.881 | 0.913 | 0.927 | 0.825 |
| 原有测试集，460 图/1514 实例 | chip-v2-v1 | 0.930 | 0.896 | 0.912 | 0.930 | 0.817 |

新模型明显改善了本次目标相机和斜视场景；原测试集的召回率和 mAP50
略升，但精度和 mAP50-95 略降，F1 基本不变。因此它适合先作为开发候选
实拍回放，不应直接晋升为 release。

候选权重：
`models/assets/chip_recognition/yolo11n-localization-hard-negative-v3/best.pt`

权重 SHA-256：
`80998949eb499a1c2f82045439757fdb697739fd9ab54df78fe4118109db5b20`

## 10/20 面额结果

正视模板首先全部生成，再通过 66 个斜视实例做确定性子集筛选。最终只新增
一张 `10` 模板；新增全部模板反而会降低结果，因此没有采用。评估时 matcher
明确限制为 `(10, 20)`。

| 方案 | 接受 | 正确 | 错误 | 拒识 | 接受样本准确率 | 全部样本正确率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原模板，限制 10/20 | 47 | 45 | 2 | 19 | 95.74% | 68.18% |
| 筛选后的模板，限制 10/20 | 52 | 51 | 1 | 14 | 98.08% | 77.27% |

分面额结果：

- `10`：正确从 21/33 提升到 27/33，错误从 2 降到 1；
- `20`：保持 24/33 正确、0 个错误，其余拒识；
- 两次重复评估得到完全相同的计数。

候选模板库：
`data/work/chips/2026-07-24-chip-v2-optimization/selected_denomination_library/`

这次模板是用同一批斜视数据筛选的，存在选择偏差。正式晋升前必须再拍一段
未参与筛选的 10/20 混合视频，按完整会话保留作独立测试。

## 本地实拍命令

树莓派流可用下列命令测试；这只打开感知显示，不发送机器人动作：

```powershell
python chip_recognition_workspace/live_chip_yolo11.py `
  --model models/assets/chip_recognition/yolo11n-localization-hard-negative-v3/best.pt `
  --template-library data/work/chips/2026-07-24-chip-v2-optimization/selected_denomination_library `
  --allowed-denominations 10 20 `
  --stream-url http://100.80.46.54:5050/
```

如当前服务实际暴露的是 `/video_feed`，将最后一个 URL 改为
`http://100.80.46.54:5050/video_feed`。

## 复现入口

- 构建数据：
  `python scripts/data/build_chip_v2_optimization_view.py`
- 训练：
  `python chip_recognition_workspace/train_chip_yolo11n.py --config chip_recognition_workspace/chip_yolo11n_chip_v2_v1.json`
- 定位评估：
  `python scripts/evaluation/evaluate_chip_localization.py --help`
- 模板构建：
  `python scripts/data/build_chip_v2_template_library.py`
- 模板筛选：
  `python scripts/evaluation/select_chip_v2_templates.py`
- 面额评估：
  `python scripts/evaluation/evaluate_chip_v2_denomination.py --help`
