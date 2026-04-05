# Transformer Implementation from Scratch

一个从零开始实现的Transformer模型，包含完整的Transformer Block组件和 Attention Sink 实验研究。

## 📋 项目概述

本项目提供了一个完整、可用的Transformer Block实现，遵循现代大语言模型的设计模式。代码简洁明了，适合学习、研究和实验使用。

**核心特性**：
- ✅ 完整的Transformer Block实现（RoPE + 多头自注意力 + SwiGLU + Pre-RMSNorm）
- ✅ Attention Sink 现象研究与可视化
- ✅ 模块化设计，易于扩展和实验
- ✅ 详细的Jupyter Notebook讲解
- ✅ 中文文档和示例

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- 其他依赖见 `requirements.txt`

### 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/Transformer.git
cd Transformer

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

## 📁 项目结构

```
Transformer/
├── src/
│   ├── transformer_module.py    # 核心Transformer模块实现
│   ├── attention_sink_module.py # Attention Sink 实验模块
│   └── main.py                  # 主程序，运行 Attention Sink 实验
├── notebooks/
│   ├── Transformer_and_attention_sink.ipynb    # Transformer构建讲解笔记
│   └── Transformer_and_attention_sink.pdf      # 笔记PDF版本
├── requirements.txt             # Python依赖
├── output.png                   # 注意力可视化示例
└── README.md                    # 本文档
```

## 🔧 核心模块

### 1. RoPE（旋转位置编码）

**文件**: `src/transformer_module.py`
**类**: `RoPE`

实现旋转位置编码（Rotary Position Embedding），仅作用于Query和Key，不作用于Value。

```python
rope = RoPE(d_k=64, max_seq_len=4096)
Q = rope(Q)  # Q: [batch, heads, seq_len, d_k]
K = rope(K)  # K: [batch, heads, seq_len, d_k]
```

**特点**：
- 预计算`cos/sin`并注册为buffer，不参与训练
- 自动跟随设备移动
- 支持可变序列长度

### 2. 多头自注意力（Multi-Head Attention）

**文件**: `src/transformer_module.py`
**类**: `MultiHeadAttention`

实现因果自回归注意力（Causal Attention），包含RoPE和因果掩码。

```python
mha = MultiHeadAttention(num_heads=8, d_model=512, max_seq_len=4096)
output = mha(x)  # x: [batch, seq_len, d_model]
```

**特点**：
- 下三角因果掩码（上三角为`-inf`）
- 缩放点积注意力
- 注意力矩阵捕获（用于可视化）

### 3. SwiGLU前馈网络

**文件**: `src/transformer_module.py`
**类**: `SwiGLU`

实现SwiGLU激活函数的前馈网络，比标准FFN更高效。

```python
ffn = SwiGLU(d_model=512, d_hidden=2048)  # 默认 multiplier=4
output = ffn(x)  # x: [batch, seq_len, d_model]
```

**公式**: `SwiGLU(x) = W2 · (SiLU(W1·x) ⊙ (V·x))`

### 4. Transformer Block

**文件**: `src/transformer_module.py`
**类**: `TransformerBlock`

完整的Transformer Block，采用Pre-RMSNorm架构。

```python
block = TransformerBlock(num_heads=8, d_model=512, max_seq_len=4096, multiplier=4)
output = block(x)  # x: [batch, seq_len, d_model]
```

**计算流程**：
1. `x_norm1 = RMSNorm(x)`
2. `h = MHA(x_norm1)`
3. `x1 = x + h`（残差连接）
4. `x_norm2 = RMSNorm(x1)`
5. `x_out = FFN(x_norm2)`
6. `x2 = x1 + x_out`（残差连接）

### 5. ToyModel（完整模型）

**文件**: `src/transformer_module.py`
**类**: `ToyModel`

包含嵌入层、多层Transformer Block和语言模型头的完整模型。

```python
model = ToyModel(num_blocks=6, num_heads=8, d_model=512, max_seq_len=4096, vocab_size=10000)
logits = model(input_ids)  # input_ids: [batch, seq_len]
```

**特点**：
- 可配置的Transformer层数
- 注意力矩阵捕获和可视化支持
- 完整的语言模型架构

## 🔬 Attention Sink 实验

### 什么是Attention Sink ？

Attention Sink 是Transformer模型中的一个有趣现象：模型倾向于将大量注意力分配给序列开头位置的Token（如`<pad>`、重复模式等），即使这些Token的信息量不大。这种现象在长序列生成中尤其明显。

### 实验模块

**文件**: `src/attention_sink_module.py`
**类**: `AttentionSinkExperiment`

提供完整的实验流程：
- 模型训练
- 注意力可视化
- 文本生成
- 检查点保存/加载

### 运行实验

```python
from src.attention_sink_module import AttentionSinkExperiment

# 创建实验对象
experiment = AttentionSinkExperiment(
    num_blocks=6,        # Transformer层数
    corpus=corpus,       # 语料库（字符集）
    num_heads=8,         # 注意力头数
    d_model=512,         # 模型维度
    max_seq_len=4096,    # 最大序列长度
    learning_rate=1e-4   # 学习率
)

# 训练模型
texts_for_train = ["我我我喜欢学习人工智能。", "这个这个这个是一个测试句子。"]
experiment.train(texts_for_train, epochs=100, save_path="checkpoint.pth")

# 可视化注意力
experiment.visualize_attention("你好，你好，你好，很高兴认识你！", layer_idx=-1, head_idx='mean')

# 文本生成
generated = experiment.generate("你好，", max_new_tokens=20)
print(generated)
```

### 使用预训练模型

```bash
# 直接运行主程序
python src/main.py
```

主程序会自动：
1. 检查是否存在预训练检查点
2. 加载或训练模型
3. 可视化注意力图
4. 生成示例文本

## 📊 实验结果示例

### 注意力可视化

运行`src/main.py`后，会生成类似下图的注意力热力图：

![Attention Sink Visualization](output.png)

**观察到的现象**：
- 序列开头的重复Token获得较高的注意力分数
- 注意力呈现"汇"状分布，向开头位置集中
- 不同注意力头可能捕获不同的模式

### 文本生成示例

输入: `"你好，你好，你好，很高兴认识你！"`

输出可能为: `"你好，你好，你好，很高兴认识你！这个这个这个是一个测试句子。我我我喜欢学习人工智能。开始开始开始今天天气很好。"`

## 📚 学习资源

### Jupyter Notebook

`notebooks/Transformer_and_attention_sink.ipynb` 提供了详细的讲解：

1. **数学公式推导**
   - RoPE旋转位置编码公式
   - 注意力计算流程
   - SwiGLU激活函数

2. **代码实现解析**
   - 模块设计思路
   - PyTorch实现技巧
   - 调试和可视化方法

3. **实验结果分析**
   - Attention Sink 现象分析
   - 模型训练策略
   - 性能优化建议

### PDF文档

`notebooks/Transformer_and_attention_sink.pdf` 是Notebook的PDF版本，方便离线阅读。

## 🧪 API参考

### `transformer_module.py`

#### `RoPE(d_k, max_seq_len=4096, b=10000)`
- `d_k`: 每个注意力头的维度
- `max_seq_len`: 支持的最大序列长度
- `b`: RoPE的基础频率

#### `MultiHeadAttention(num_heads, d_model, max_seq_len=4096)`
- `num_heads`: 注意力头数量
- `d_model`: 模型维度
- `max_seq_len`: 最大序列长度

#### `TransformerBlock(num_heads, d_model, max_seq_len=4096, multiplier=4)`
- `num_heads`: 注意力头数量
- `d_model`: 模型维度
- `max_seq_len`: 最大序列长度
- `multiplier`: FFN隐藏层维度倍数

#### `ToyModel(num_blocks, num_heads=8, d_model=512, max_seq_len=4096, vocab_size=10000)`
- `num_blocks`: Transformer层数
- `num_heads`: 注意力头数量
- `d_model`: 模型维度
- `max_seq_len`: 最大序列长度
- `vocab_size`: 词汇表大小

### `attention_sink_module.py`

#### `AttentionSinkExperiment(num_blocks, corpus=None, num_heads=8, d_model=512, max_seq_len=4096, learning_rate=3e-4, load_from=None)`
- `num_blocks`: Transformer层数
- `corpus`: 语料库字符串（字符集）
- `num_heads`: 注意力头数量
- `d_model`: 模型维度
- `max_seq_len`: 最大序列长度
- `learning_rate`: 学习率
- `load_from`: 预训练检查点路径

#### 方法
- `train(texts_for_train, epochs=100, log_interval=10, save_path=None)`: 训练模型
- `visualize_attention(text_for_test, layer_idx=-1, head_idx='mean')`: 可视化注意力
- `generate(text_for_prompt, max_new_tokens=10)`: 文本生成

## 🛠️ 开发和扩展

### 实验配置

修改`src/main.py`中的参数：

```python
# 调整模型参数
experiment = AttentionSinkExperiment(
    num_blocks=12,       # 增加层数
    num_heads=16,        # 增加注意力头
    d_model=768,         # 增大模型维度
    learning_rate=5e-5   # 调整学习率
)

# 调整训练数据
texts_for_train = [
    # 添加更多训练样本
    "重复重复重复重复重复重复",
    "模式模式模式模式模式模式",
    # ...
]
```

### 性能优化

1. **设备选择**: 自动检测MPS（Apple Silicon）、CUDA或CPU
2. **内存优化**: 使用梯度检查点、混合精度训练
3. **批处理**: 支持批处理训练和推理


---

**Happy Coding! 🚀**

如果这个项目对你有帮助，请考虑给它一个⭐️星标！

