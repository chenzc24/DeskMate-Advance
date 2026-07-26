# 统一手牌 Runtime 模拟验证

日期：2026-07-22

## 已验证

- 冻结四人 roster 后进入 `DEALING_HOLE`，不再假定底牌已存在。
- 底牌按 Button→SB→BB→UTG 定位四次，每位连续执行两次单张发牌 ACK，
  并把两个逻辑槽依次标记为背面在位。
- Pre-flop、Flop、Turn、River 四轮 Part A 均由引擎给出唯一
  `acting_seat` 和合法动作。
- Flop 连续发 3 张后在一个共享画面中批量确认，Turn 1 张、River 1 张分别确认，无烧牌。
- Showdown 逐名未弃牌玩家确认两个底牌槽，并只从已确认槽结算。
- 完整手牌恰好执行十三次 `dispense_one`。
- Flop 只定位一次，因此测试牌局产生 40 个 command/ACK 对：13 次单张
  发牌均有独立 ACK，四名玩家的两张底牌共享一次定位，Flop 的三次发牌共享
  一次定位；四轮各四次
  玩家转向、Showdown 四次玩家转向。
- 重复 ACK 幂等；未知牌保持；错命令、命令/视觉/动作超时和牌面冲突暂停。
- 第一张底牌 ACK 后重启会恢复为同一玩家第二张发牌，公共牌 ACK 后恢复为等待
  视觉，不重复发牌；未决命令重启会暂停。
- 直接按钮到 game 的入口必须显式声明 `allow_direct_engine_pilot=True`。

## 验证命令

```powershell
.\.venv\Scripts\python.exe -m compileall -q src scripts
.\.venv\Scripts\python.exe -m pytest -q
git diff --check
```

结果：`261 passed`，`git diff --check` 通过，仅有 Git 的 LF/CRLF 提示。

## 尚未声称

- 没有连接实体 dealer，没有授权实体运动。
- 没有把现有 Live Part A UI 和 Live Card Pilot 装配成统一实机入口。
- 没有完成目标相机 ROI、牌面模型 release、MCU 协议和 Stage 3 安全 Gate。
- 没有实现从 `PAUSED_RECOVERY` 原地继续；当前安全路径是操作员检查后作废/
  重发，避免猜测物理桌面。
