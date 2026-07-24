# 10/20 筹码外圈颜色二分类版本

## 新版本边界

该版本是独立入口，不覆盖原来的“外圈颜色＋透视矫正＋数字模板”版本：

- 原版本：
  `chip_recognition_workspace/live_chip_yolo11.py`
- 新版本：
  `chip_recognition_workspace/live_chip_yolo11_rim_color.py`

两者共用同一个 YOLO 单类筹码定位权重。新版本在框出筹码后，直接在 YOLO
框内构造椭圆环带，只读取外圈颜色分布：

- 蓝色与肉色交替：`10`
- 绿色与深绿色交替：`20`

它不会运行 OCR、数字模板匹配、GrabCut 或透视矫正，也不会输出 `1/5`。
接受的单帧结果仍需通过现有 `track_id` 多帧投票后才显示为稳定面额。

## 特征和数据

二分类器使用 40 维固定特征：饱和度加权色相直方图、HSV/Lab 分布，以及
蓝、绿、肉色、深绿、亮色和暗色在外圈中的占比。运行时只是标准化和一次
逻辑回归点积，不依赖 scikit-learn。

参数文件：
`models/assets/chip_recognition/rim-colour-binary-10-20-v1/model.json`

训练来源仍是 2026-07-24 的 10/20 固定设计筹码，共 75 个复核实例。复核时
发现并修正了两处原面额标注错误：

- `chip_v2/mixv3.png` 左侧绿色筹码实际为 `20`；
- `chip_v2/mixv6.png` 上排右侧绿色筹码实际为 `20`。

## 结果

在 66 个斜视实例上，最终直接外圈版本得到：

| 指标 | 结果 |
| --- | ---: |
| 接受 | 66 |
| 正确 | 66 |
| 错误 | 0 |
| 拒识 | 0 |
| 单筹码平均颜色处理延迟 | 3.68 ms |

此前颜色＋数字版本在相同实例上的平均面额处理约为 114 ms。新的直接取色
版本不再做数字分割和旋转模板搜索，因此更适合多筹码实时画面。

这 66 个实例参与了最终运行参数的拟合，`100%` 是开发集结果，不能视为
独立泛化成绩。最终参数拟合前，完整 `chip_v2:20` 采集组曾作为临时保留组，
结果为 21/23；正式晋升仍需重新拍摄一段未参与开发的 10/20 混合视频。

## 树莓派视频流测试

当前本机映射的视频地址为：
`http://127.0.0.1:5000/video_feed`

启动命令：

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python chip_recognition_workspace/live_chip_yolo11_rim_color.py `
  --model models/assets/chip_recognition/yolo11n-localization-hard-negative-v3/best.pt `
  --template-library data/work/chips/2026-07-24-chip-v2-optimization/selected_denomination_library `
  --allowed-denominations 10 20 `
  --stream-url http://127.0.0.1:5000/video_feed
```

`--template-library` 在新入口中只为兼容原命令行结构；实际二分类参数来自
`models/assets/chip_recognition/rim-colour-binary-10-20-v1/model.json`。按 `q` 或
`Esc` 退出。

该程序只显示非权威感知结果，不修改游戏账本、不推进状态机，也不连接机器人。
