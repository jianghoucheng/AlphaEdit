# Qwen AlphaEdit

这是一个仅面向稠密 Qwen 系列因果语言模型的 AlphaEdit 实现。项目保留
AlphaEdit 的零空间约束知识编辑逻辑，已移除其他编辑算法和论文对比框架。

## 支持范围

- Qwen2、Qwen2.5、Qwen3 以及具有相同稠密 MLP 布局的 Qwen3.5 文本模型。
- 模型必须具有单一的 `mlp.down_proj`；MoE 和量化模型暂不支持。
- 编辑过程当前要求模型完整放在一张 CUDA GPU 上。
- 请求 prompt 必须包含一个 `{}`，用于填入 subject。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Qwen3.5 需要 Transformers 5.x。对于只使用 Qwen2/Qwen2.5 的旧环境，可以
根据实际模型兼容性调整 `pyproject.toml` 中的 Transformers 版本。

## 编辑请求

参考 [examples/edits.json](examples/edits.json)：

```json
[
  {
    "case_id": 0,
    "prompt": "{} is located in",
    "subject": "The Eiffel Tower",
    "target_new": {"str": " Rome"}
  }
]
```

`target_new` 也可以直接写成字符串。程序会自动补充目标前的空格。

## 运行

```bash
qwen-alphaedit \
  --model Qwen/Qwen2.5-7B \
  --config configs/qwen2.5-7b.json \
  --requests examples/edits.json \
  --stats-dir data/stats \
  --cache-dir data/cache \
  --output-dir outputs/qwen2.5-7b-edited \
  --device cuda:0
```

第一次运行会从 Wikipedia 计算各编辑层的二阶矩统计，耗时和磁盘占用都较大。
之后会直接读取 `--stats-dir` 中的缓存。

输出目录包含：

- 编辑后的模型和 tokenizer；
- `alphaedit_config.json`：解析后的实际模型模块配置；
- `alphaedit_state.pt`：零空间投影与累计 key 协方差。

继续进行顺序编辑时，同时加载上次保存的模型和状态：

```bash
qwen-alphaedit \
  --model outputs/qwen2.5-7b-edited \
  --state outputs/qwen2.5-7b-edited/alphaedit_state.pt \
  --config configs/qwen2.5-7b.json \
  --requests next_edits.json \
  --output-dir outputs/qwen2.5-7b-edited-2
```

## Python API

```python
from alphaedit import AlphaEditConfig, AlphaEditor

config = AlphaEditConfig.from_json("configs/qwen2.5-7b.json")
editor = AlphaEditor(
    model,
    tokenizer,
    config,
    stats_dir="data/stats",
    cache_dir="data/cache",
)
edited_model = editor.edit(requests)
editor.save_state("outputs/alphaedit_state.pt")
```

## 开发检查

```bash
python -m compileall -q alphaedit tests
pytest
ruff check alphaedit tests
```

完整的端到端编辑仍需要真实 Qwen 权重、Wikipedia 数据和 CUDA GPU；单元测试只
覆盖配置解析、Qwen 模块检测、请求校验和矩阵形状处理。

## 来源

核心方法来自 *AlphaEdit: Null-Space Constrained Knowledge Editing for
Language Models*。协方差统计与 PyTorch hook 工具由原始 AlphaEdit/MEMIT
研究代码整理而来。
