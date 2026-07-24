# OpenCV机器视觉巡线原型

该目录是电脑端OpenCV巡线视觉模块。默认目标环境为绿色绒布上的白色
引导线，当前只完成视觉感知：

```text
电脑摄像头/视频/图片
  -> 地面ROI
  -> HSV白色线提取
  -> 绿色布面范围约束
  -> 形态学去噪
  -> 近、中、远三段候选选择
  -> 连续性与置信度检查
  -> LineObservation
```

它不会调用GPIO、串口、电机、机器人适配器或游戏状态机。

## 文件结构

- `observations.py`：稳定的巡线观察数据结构。
- `config.py`：参数、JSON加载和参数校验。
- `detector.py`：纯OpenCV三段ROI检测器。
- `visualization.py`：调试画面，不参与控制结果。
- `live_line_detection.py`：电脑摄像头、视频和图片入口。
- `config.white_on_green.json`：绿色绒布白线的默认参数。
- `config.dark_line.json`：保留的浅色地面黑线回退参数。
- `../../tests/track_line/test_detector.py`：无需真实摄像头的合成测试。

## 环境

项目已经声明NumPy和OpenCV依赖。在项目环境中确认：

```powershell
python -c "import cv2, numpy; print(cv2.__version__)"
```

## 电脑摄像头

```powershell
python src/track_line/live_line_detection.py --source 0
```

如果DroidCam在Windows中是第二个摄像头：

```powershell
python src/track_line/live_line_detection.py --source 1
```

按`q`或`Esc`退出，空格暂停/继续。

## 视频和图片

视频回放：

```powershell
python src/track_line/live_line_detection.py `
  --source C:\path\to\track.mp4
```

图片检测并保存标注结果：

```powershell
python src/track_line/live_line_detection.py `
  --source C:\path\to\track.jpg `
  --headless `
  --output C:\path\to\track_result.jpg
```

无窗口运行前100帧：

```powershell
python src/track_line/live_line_detection.py `
  --source C:\path\to\track.mp4 `
  --headless `
  --max-frames 100
```

程序定期输出JSON，其中：

- `offset`：横向误差，负数在线左侧、正数在线右侧。
- `heading`：远处线路相对近处线路的方向。
- `curvature`：三段中心的二阶变化。
- `confidence`：0到1的检测证据强度。
- `line_lost`：线路不可安全使用。
- `points_normalized`：远、中、近中心的归一化坐标。

`line_lost=true`时，三个误差字段为`null`，下游不得猜测方向。

## 绿色绒布和白线分割

默认模式不再使用普通灰度Otsu。程序先在HSV空间中提取：

- 低饱和度、高亮度的白色线候选；
- 指定色相、饱和度范围内的绿色布面。

程序使用主要绿色区域生成地面凸包，只保留凸包内部的白色候选。因此画面
边缘的白色杯子、墙面等不应直接成为线路。默认阈值位于
`config.white_on_green.json`：

- `white_saturation_max`：白色允许的最大饱和度；
- `white_value_min`：白色最低亮度；
- `green_hue_min/max`：绿色布面色相范围；
- `minimum_green_roi_ratio`：绿色地面不足时直接拒绝整帧。

当前阈值根据提供的无白线绿布照片和合成白线建立，只是开发初值。最终必须
使用固定在小车上的摄像头，拍摄真实白线、阴影和不同距离后再校准。

## 测试

```powershell
python -m pytest tests/track_line/test_detector.py -q
```

## 调参顺序

1. 固定最终摄像头高度和俯角。
2. 调整`roi_top_ratio`，只保留地面区域。
3. 先调整`white_value_min`和`white_saturation_max`，确保只留下白线。
4. 调整绿色HSV范围，确保布面稳定形成地面范围。
5. 调整面积和宽度上下限，过滤小白点和大块白色杂物。
6. 最后调整`minimum_confidence`和`continuity_scale`。

先录制最终安装视角的视频并离线回放。检测稳定后，再单独增加PD/PID
控制器和模拟电机适配器；首次物理测试必须低速、有人值守并具备急停。
