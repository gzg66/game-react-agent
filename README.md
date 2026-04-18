# 游戏 ReAct 自动化智能体

一个基于 ReAct 模式的游戏自动化测试项目，结合 Poco / Airtest、页面状态图和 Gemini 模型，实现：

- 冷启动探索游戏 UI，构建页面状态图
- 基于缓存路径做低成本导航
- 在运行时通过工具调用执行点击、滑动、等待等操作
- 在必要时结合截图和 UI 树进行多模态分析

## 当前特性

- 命令行输出、日志文案、模型提示词已统一为中文
- 支持模拟设备模式，便于本地验证流程
- 支持真实设备模式，用于接入实际游戏包进行探索
- 支持将历史导出的 `outputs/` JSON 和 `graph.db` 批量迁移为中文内容

## 环境要求

- Python `>= 3.11`
- 可选真实设备依赖：
  - `airtest`
  - `pocoui`

## 安装

基础依赖：

```bash
pip install -e .
```

如果需要连接真实设备：

```bash
pip install -e ".[device]"
```

如果需要开发依赖：

```bash
pip install -e ".[dev]"
```

## 主要目录

- `src/game_agent/`：核心代码
- `scripts/run_agent.py`：交互式智能体入口
- `scripts/explore_cold_start.py`：冷启动探索入口
- `config/`：配置文件
- `data/`：日志、图数据库等运行数据
- `outputs/`：探索导出结果

## 配置说明

默认配置位于 `config/default.yaml`，示例真机配置位于 `config/xttc_poco.yaml`。

重点配置项：

- `device`：设备平台、串号、是否使用模拟设备
- `game`：包名、Activity、Poco 地址、设备 URI
- `exploration`：探索步数、页面上限、输出目录
- `vision`：视觉模型开关及参数
- `gemini`：文本模型参数
- `graph`：状态图数据库路径与探索深度

## 常用命令

使用模拟设备运行冷启动探索：

```bash
python scripts/explore_cold_start.py --mock --max-steps 20
```

使用真实设备运行冷启动探索：

```bash
python scripts/explore_cold_start.py --config config/xttc_poco.yaml
```

启动交互式智能体：

```bash
python scripts/run_agent.py --config config/xttc_poco.yaml --task "进入英雄界面"
```

## 历史数据中文化

如果此前已经导出过英文版 `outputs/*.json` 或 `graph.db`，可以执行：

```bash
python scripts/localize_history.py
```

脚本会尝试处理：

- `outputs/**/*.json`
- `outputs/**/*.db`
- `data/graph.db`

当前支持的迁移内容主要包括：

- 页面分类英文值转中文
- 默认页面名 `page_xxx` 转为 `页面_xxx`
- 默认页面描述 `Auto-discovered at step N` 转为中文
- 汇总状态 `completed` 转为 `已完成`

## 说明

- 第三方服务返回的原始错误消息可能仍然包含英文，这是外部依赖返回内容，不一定能完全本地化。
- 如果当前仓库里还没有历史 `outputs/` 或 `graph.db`，迁移脚本会直接跳过并输出中文提示。
