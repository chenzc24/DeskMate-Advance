# 筹码定位与面额识别开发程序

本文说明如何复现 Poker Dealer 当前的实体筹码开发检测程序。它使用
YOLO11n 定位筹码，再通过最佳帧、外圈颜色、透视/椭圆矫正、中心数字模板
和多帧确认识别 `1`、`5`、`10`、`20` 面额。

该程序属于 Plus 感知实验。输出只能作为桌面可见筹码证据，不能修改游戏
状态、玩家余额或权威数字账本，也不会发送机器人指令。

## 运行资产

| 资产 | 版本 | 路径 |
|---|---|---|
| 筹码定位模型 | `chip-localization-yolo11n@hard-negative-v3-20260723` | `models/assets/chip_recognition/yolo11n-localization-hard-negative-v3/best.pt` |
| 固定设计面额模板 | `chip-denomination-fixed-design-template@las-vegas-v1-20260723` | `models/assets/chip_recognition/las-vegas-denomination-templates-v1/` |

定位模型通过 Git LFS 管理。模板包包含 18 张 128×128 二值数字掩码和
颜色特征，不包含原始筹码照片。模板只适用于本项目当前的
`LAS VEGAS POKER CLUB` 版 `1/5/10/20` 筹码，不是通用赌场筹码规则。

## 首次安装

在仓库根目录执行：

```powershell
git lfs install
git lfs pull
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pip install ultralytics==8.4.104
```

已有项目虚拟环境时，只需执行 `git lfs pull` 并确认 Ultralytics 版本。

检查模型是否已下载：

```powershell
Get-FileHash models\assets\chip_recognition\yolo11n-localization-hard-negative-v3\best.pt -Algorithm SHA256
```

期望 SHA-256：

```text
7ab3a870bf6865127e3c65fc6c8e771a3c990aed94e97f6977bfdba1ac09e3c3
```

## 启动方式

### 电脑摄像头

默认使用摄像头 `0`、MSMF 后端：

```powershell
.\.venv\Scripts\python.exe chip_recognition_workspace\live_chip_yolo11.py
```

指定其他摄像头或 DroidCam：

```powershell
.\.venv\Scripts\python.exe chip_recognition_workspace\live_chip_yolo11.py --camera-index 1 --backend dshow
```

先用以下命令或系统相机程序确认 DroidCam 对应的设备编号。

### 树莓派 MJPEG 视频流

```powershell
.\.venv\Scripts\python.exe chip_recognition_workspace\live_chip_yolo11.py --stream-url http://<PI_TAILSCALE_IP>:5000/video_feed
```

网络视频流使用 FFmpeg，不要同时传入 `--backend`。确保电脑能访问树莓派
Tailscale IP，且树莓派上的 MJPEG 服务监听 `5000` 端口。

### 有界无窗口检查

```powershell
.\.venv\Scripts\python.exe chip_recognition_workspace\live_chip_yolo11.py --camera-index 0 --max-frames 100 --headless --emit-all
```

窗口模式按 `Q` 或 `Esc` 退出。程序不保存画面。

## 画面与输出

程序按以下顺序处理：

```text
YOLO 定位
  -> track_id
  -> 轨迹窗口中选择最佳原始帧
  -> 外圈颜色证据
  -> 椭圆/透视矫正为正圆
  -> 中心数字模板匹配
  -> 多帧投票确认
  -> 可见数量与非权威面额总数
```

框标签显示定位置信度、轨迹 ID、面额状态和已确认面额。底部的
`observed_visible_total` 只统计当前被识别且未拒绝的可见筹码；出现
`partial` 时不可当作完整总数。

加上 `--emit-all` 后，终端会逐帧输出 JSON。重要字段包括：

- `visible_chip_count`：当前定位到的可见筹码数量；
- `denomination_counts`：已确认的 `1/5/10/20` 数量；
- `observed_visible_total`：已确认面额之和；
- `total_complete`：当前所有定位框是否均获得稳定面额；
- `value_state`：`collecting`、`confirmed` 或拒识状态；
- `value_rejection_reason`：例如 `too_far`、`no_fresh_match`。

## 常用参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--confidence` | `0.25` | YOLO 最低定位置信度 |
| `--nms-iou` | `0.45` | 重叠框抑制阈值 |
| `--imgsz` | `960` | YOLO 推理尺寸 |
| `--device` | `0` | CUDA 设备；无 CUDA 时使用 `cpu` |
| `--best-frame-window` | `5` | 每条轨迹选择最佳帧的采样数 |
| `--value-score` | `0.58` | 数字模板最低得分 |
| `--value-margin` | `0.035` | 第一、第二候选的最低差距 |
| `--value-min-minor-axis` | `42` | 过远小筹码的最小短轴像素 |
| `--value-min-aspect-ratio` | `0.38` | 允许的最低椭圆宽高比 |
| `--value-cache-frames` | `8` | 面额结果缓存帧数 |

不要为了提高接收率同时大幅降低 `value-score` 和 `value-margin`，否则
`1` 与 `5` 的混淆和错误稳定会增加。过远筹码优先改善相机距离、焦点和
分辨率。

## 验证

运行筹码模块测试：

```powershell
.\.venv\Scripts\python.exe -m pytest chip_recognition_workspace -q
```

模型、模板版本、哈希、指标和限制记录在 `models/manifest.yaml`。当前
模板评估来自已有开发拍摄，不是独立相机/场次留出集，因此模型和模板都
保持 `development` 状态。

## 常见问题

- `fine-tuned model is missing`：执行 `git lfs pull`。
- `template library is missing`：确认仓库已更新，并且没有覆盖默认
  `--template-library` 路径。
- 树莓派 `stream timeout`：检查 Tailscale 连通性和 `5000` 端口服务。
- 能框到但没有面额：查看 `value_rejection_reason`；常见原因是距离过远、
  角度太扁、遮挡或最佳帧仍模糊。
- 无 CUDA：增加 `--device cpu`，速度会下降但输出合同不变。
- 非当前筹码设计：必须重新采集正面模板并进行独立评估，不能复用本模板
  的颜色与数字阈值宣称可靠识别。
