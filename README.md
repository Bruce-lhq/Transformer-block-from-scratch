# Transformer（从零搭建单个 Transformer Block）

这个仓库用于用 **PyTorch** 手写实现一个最小可用的 Transformer Block（偏 LLaMA 风格）组件，并在笔记本中配套解释其计算流程。

当前已实现（见 [src/transformer_module.py](src/transformer_module.py)）：
- RoPE 旋转位置编码：[`RoPE`](src/transformer_module.py)
- 多头自注意力（含因果 Mask）：[`MultiHeadAttention`](src/transformer_module.py)
- 前馈网络 SwiGLU：[`SwiGLU`](src/transformer_module.py)
- 预归一化 + 残差的 Transformer Block：[`TransformerBlock`](src/transformer_module.py)

配套讲解见：  
- [notebooks/Transformer_building.ipynb](notebooks/Transformer_building.ipynb)

---

## 环境依赖

见 [requirements.txt](requirements.txt)：

- torch

安装（建议在虚拟环境中）：
```bash
pip install -r requirements.txt
```

---

## 快速使用

下面是构造并前向运行 [`TransformerBlock`](src/transformer_module.py) 的最小例子：

```python
import torch
from src.transformer_module import TransformerBlock

batch_size = 2
seq_len = 16
d_model = 128
num_heads = 8

x = torch.randn(batch_size, seq_len, d_model)

block = TransformerBlock(num_heads=num_heads, d_model=d_model, max_seq_len=4096, multiplier=4)
y = block(x)

print("x:", x.shape)  # [2, 16, 128]
print("y:", y.shape)  # [2, 16, 128]
```

---

## 模块说明（实现要点）

### 1) RoPE（旋转位置编码）
- 仅作用于 Q/K，不作用于 V
- 通过预计算 `cos/sin` 并 `register_buffer`，避免参与训练且自动跟随设备移动  
见：[`RoPE.forward`](src/transformer_module.py)

### 2) Multi-Head Attention（含因果 Mask）
- `W_Q/W_K/W_V/W_O` 线性投影后 reshape 成 `[batch, heads, seq, d_k]`
- 使用 RoPE 修正 Q/K
- 使用下三角 causal mask（上三角为 `-inf`）保证自回归不可见未来信息  
见：[`MultiHeadAttention.forward`](src/transformer_module.py)

### 3) TransformerBlock（Pre-RMSNorm）
结构为：
1. `x_norm1 = RMSNorm(x)`
2. `h = MHA(x_norm1)`
3. `x1 = x + h`（残差）
4. `x_norm2 = RMSNorm(x1)`
5. `x_out = FFN(x_norm2)`
6. `x2 = x1 + x_out`（残差）  
见：[`TransformerBlock.forward`](src/transformer_module.py)

---

## 目录结构

```
notebooks/
  Transformer_building.ipynb    # 讲解笔记（公式 + 代码）
src/
  transformer_module.py         # RoPE/MHA/SwiGLU/TransformerBlock 实现
requirements.txt
```