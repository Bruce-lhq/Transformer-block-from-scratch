import torch
import torch.nn as nn
import torch.nn.functional as F
class RoPE(nn.Module):
    def __init__(self, d_k, max_seq_len = 4096, b=10000):
        super().__init__()
        theta_i = 1/(b**(torch.arange(0,d_k,2).float()/d_k))  # \theta_i = b^{-\frac{2i}{d}}, \quad i \in \{0,1,\dots,\frac{d}{2}-1\}
        m = torch.arange(max_seq_len).float()  # m = [0,1,...,seq_len-1]
        m_theta_i = torch.outer(m, theta_i)  # [seq_len, d_k/2]
        cos = torch.cos(torch.cat((m_theta_i, m_theta_i), dim=-1))  # [seq_len, d_k]
        sin = torch.sin(torch.cat((m_theta_i, m_theta_i), dim=-1))  # [seq_len, d_k]
        self.register_buffer('cos', cos[None, None, :, :])  # 好处：cos和sin不需要更新参数，注册为buffer后会自动放到正确的设备上
        self.register_buffer('sin', sin[None, None, :, :])  # 扩充维度以适应后续计算

    def forward(self, x): # 适用于Q、K (V不需要位置编码!)
        # x: [batch_size, num_heads, seq_len, d_k]
        seq_len = x.shape[2]
        d_2 = x.shape[-1] // 2
        cos = self.cos[:, :, :seq_len, :]  # [1, 1, seq_len, d_k]
        sin = self.sin[:, :, :seq_len, :]
        x_first_half = x[..., :d_2]
        x_second_half = x[..., d_2:]
        x_flip = torch.cat((-x_second_half, x_first_half), dim=-1)  
        x_out = x * cos + x_flip * sin  # 旋转位置编码
        return x_out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, max_seq_len=4096):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        # 维度属性
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        # 权重矩阵
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False) 
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        # RoPE位置编码
        self.rope = RoPE(self.d_k, max_seq_len=max_seq_len)
        # 因果掩码
        tril = torch.tril(torch.ones(max_seq_len, max_seq_len)).bool() # torch.tril (triangular lower)提取下三角元素，并把上三角元素置0
        mask = torch.zeros(max_seq_len, max_seq_len).masked_fill(~tril, float('-inf')) # 下三角元素置0，上三角元素置-inf
        self.register_buffer('mask', mask[None, None, :, :])  # [1, 1, seq_len, seq_len]

    def forward(self, x): # x: [batch_size, seq_len, d_model]
        # 线性投影与分头
        batch_size, seq_len, _ = x.shape
        Q = self.W_Q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # [batch_size, num_heads, seq_len, d_k]
        K = self.W_K(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # 交换(1, 2)是因为之后softmax时要进行矩阵乘法(只乘后两维)
        V = self.W_V(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # 在seq_len维度上分配注意力
        # RoPE位置编码
        Q = self.rope(Q)
        K = self.rope(K)
        # 缩放点积注意力与因果掩码 Causal Masking
        ## Query-Key 点积计算注意力得分
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)  # [batch_size, num_heads, seq_len, seq_len]
        ## 因果掩码和 softmax 计算注意力权重
        Attention = F.softmax(scores + self.mask[..., :seq_len, :seq_len], dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
        self.captured_attention = Attention  # **新增**：捕获注意力矩阵，用hook记录下来，方便后续分析
        ## 乘以 Value 完成加权求和
        Out = torch.matmul(Attention, V)  # [batch_size, num_heads, seq_len, d_k]
        # 多头拼接并乘以输出矩阵(必须先内存连续化(contiguous)再view，否则会报错)
        Out = Out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)  # [batch_size, seq_len, d_model]
        H = self.W_O(Out)  # [batch_size, seq_len, d_model]
        return H


class StreamingMultiHeadAttention(MultiHeadAttention):
    def __init__(self, num_heads, d_model, sink_size=4, window_size=10, max_seq_len=4096):
        super().__init__(num_heads, d_model, max_seq_len=max_seq_len)  
        self.sink_size = sink_size
        self.window_size = window_size
        self.max_cache_len = sink_size + window_size
        self.register_buffer('k_cache', None, persistent=False) 
        self.register_buffer('v_cache', None, persistent=False)
        self.seq_len = 0 # 当前已经处理的总序列长度，初始为0

    def forward(self, x_new): 
        batch_size = x_new.shape[0]
        seq_len_new = x_new.shape[1] 
        # 无论哪种情况，都需要先计算出纯净的 Q, K, V (借用父类的线性层)
        Q_new = self.W_Q(x_new).view(batch_size, seq_len_new, self.num_heads, self.d_k).transpose(1, 2)
        K_new = self.W_K(x_new).view(batch_size, seq_len_new, self.num_heads, self.d_k).transpose(1, 2)
        V_new = self.W_V(x_new).view(batch_size, seq_len_new, self.num_heads, self.d_k).transpose(1, 2)
        # 初始化预分配固定内存 (Ring Buffer)
        if self.k_cache is None or self.k_cache.shape[0] != batch_size:
            self.k_cache = torch.zeros((batch_size, self.num_heads, self.max_cache_len, self.d_k), device=x_new.device, dtype=K_new.dtype)
            self.v_cache = torch.zeros_like(self.k_cache)
            self.seq_len = 0 
        # ============================================================
        # 分成两大阶段：预填充阶段(prefill)和流式解码阶段(decoding)
        if seq_len_new>1:
            # 预填充阶段(prefill),输入一段长提示词
            if seq_len_new <= self.max_cache_len:
                ### 还没超长，全部存下
                self.k_cache[:, :, :seq_len_new, :] = K_new
                self.v_cache[:, :, :seq_len_new, :] = V_new
            else:
                ### 超长了，存满 sink_size 的部分，剩余的部分只保留最后 window_size 的部分
                self.k_cache[:, :, :self.sink_size, :] = K_new[:, :, :self.sink_size, :]
                self.v_cache[:, :, :self.sink_size, :] = V_new[:, :, :self.sink_size, :]
                self.k_cache[:, :, self.sink_size:, :] = K_new[:, :, -self.window_size:, :]
                self.v_cache[:, :, self.sink_size:, :] = V_new[:, :, -self.window_size:, :]
            ## 更新当前序列长度
            self.seq_len = seq_len_new
            ## 按照父类的逻辑继续计算注意力
            Q_rotated = self.rope(Q_new)
            K_rotated = self.rope(K_new)
            ### 缩放点积注意力与因果掩码 Causal Masking
            #### Query-Key 点积计算注意力得分
            scores = torch.matmul(Q_rotated, K_rotated.transpose(-2, -1)) / (self.d_k ** 0.5)  # [batch_size, num_heads, seq_len, seq_len]
            #### 因果掩码和 softmax 计算注意力权重
            Attention = F.softmax(scores + self.mask[..., :seq_len_new, :seq_len_new], dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
            self.captured_attention = Attention  # **新增**：捕获注意力矩阵，用hook记录下来，方便后续分析
            #### 乘以 Value 完成加权求和
            Out = torch.matmul(Attention, V_new)  # [batch_size, num_heads, seq_len, d_k]
            ### 多头拼接并乘以输出矩阵(必须先内存连续化(contiguous)再view，否则会报错)
            Out = Out.transpose(1, 2).contiguous().view(batch_size, seq_len_new, -1)  # [batch_size, seq_len, d_model]
            H = self.W_O(Out)  # [batch_size, seq_len, d_model]
            return H
        # ============================================================
        else:
            # 流式解码阶段(decoding),每次输入一个新 token
            ## 计算插入索引并原位更新
            if self.seq_len < self.max_cache_len:
                insert_idx = self.seq_len
            else:
                insert_idx = self.sink_size + (self.seq_len-self.sink_size) % self.window_size
            self.k_cache[:, :, insert_idx:insert_idx+1, :] = K_new
            self.v_cache[:, :, insert_idx:insert_idx+1, :] = V_new
            ## 提取出当前有效的 Cache 进行 Attention 计算
            if self.seq_len < self.max_cache_len:
                K_active = self.k_cache[:, :, :self.seq_len+1, :]
                V_active = self.v_cache[:, :, :self.seq_len+1, :]
            else:
                K_active = torch.cat((self.k_cache[:, :, :self.sink_size, :], self.k_cache[:, :, insert_idx+1:, :], self.k_cache[:, :, self.sink_size:insert_idx+1, :]), dim=2)
                V_active = torch.cat((self.v_cache[:, :, :self.sink_size, :], self.v_cache[:, :, insert_idx+1:, :], self.v_cache[:, :, self.sink_size:insert_idx+1, :]), dim=2)
            self.seq_len += 1
            ## RoPE位置编码
            K_active = self.rope(K_active)
            pad_len = K_active.shape[2]
            dummy_Q = torch.zeros((batch_size, self.num_heads, pad_len, self.d_k), device=x_new.device, dtype=x_new.dtype)
            dummy_Q[:, :, -1:, :] = Q_new  # 只在最后一个位置放入新计算的 Q，其他位置全是0，这样就能正确地应用 RoPE 位置编码
            dummy_Q = self.rope(dummy_Q)
            Q_new = dummy_Q[:, :, -1:, :]  # 提取出最后一个位置的 Q，形状仍然是 [batch_size, num_heads, 1, d_k]
            ## 缩放点积注意力
            ### 单步生成时，不需要 Causal Mask！因为所有的 K 都是历史或现在，根本不存在未来的信息泄露问题了。
            scores = torch.matmul(Q_new, K_active.transpose(-2, -1)) / self.d_k ** 0.5
            Attention = F.softmax(scores, dim=-1)
            self.captured_attention = Attention
            Out = torch.matmul(Attention, V_active)
            Out = Out.transpose(1, 2).contiguous().view(batch_size, 1, -1)
            H = self.W_O(Out)
            return H
        

class AttentionProbe:
    def __init__(self):
        self.captured_data = []
    def __call__(self, module, input, output):
        attention = module.captured_attention.detach().cpu().numpy()
        self.captured_data.append(attention)
    def reset(self):
        self.captured_data = []


class HiddenStateProbe:
    """捕获 TransformerBlock 每次调用的输出（即每层的隐藏状态）"""
    def __init__(self):
        self.captured_data = []
    def __call__(self, module, input, output):
        self.captured_data.append(output.detach().cpu())
    def reset(self):
        self.captured_data = []


class SwiGLU(nn.Module):
    def __init__(self,d_model,d_hidden):
        super().__init__()
        self.W = nn.Linear(d_model,d_hidden,bias=False)
        self.V = nn.Linear(d_model,d_hidden,bias=False)
        self.W2 = nn.Linear(d_hidden,d_model,bias=False)
    def forward(self,x):
        x1 = F.silu(self.W(x))
        x2 = self.V(x)
        x_out = self.W2(x1 * x2)
        return x_out


class TransformerBlock(nn.Module):
    def __init__(self, num_heads, d_model, max_seq_len=4096, multiplier=4, sink_size=4, window_size=10):
        super().__init__()
        # 用 RMSNorm 类而不是函数，因为它自动包含了可学习的缩放参数
        self.rmsnorm1 = nn.RMSNorm(d_model)  # 第一次归一化 Pre-RMSNorm
        self.rmsnorm2 = nn.RMSNorm(d_model)  # 第二次归一化
        self.attention = StreamingMultiHeadAttention(num_heads, d_model, sink_size=sink_size, window_size=window_size, max_seq_len=max_seq_len)
        self.ffn = SwiGLU(d_model, d_model * multiplier)
    
    def forward(self, x):
        # 第一次归一化 Pre-RMSNorm
        x_norm1 = self.rmsnorm1(x)
        # 全局信息交互 Multi-Head Attention (MHA)
        h = self.attention(x_norm1)
        # 第一次残差连接
        x1 = x + h
        # 第二次归一化
        x_norm2 = self.rmsnorm2(x1)
        # 前馈网络 Feed-Forward Network (FFN) 
        x_out = self.ffn(x_norm2)
        # 第二次残差连接
        x2 = x1 + x_out
        return x2


class SimpleTokenizer:
    def __init__(self, corpus): # 输入语料文本
        self.chars = ['<pad>'] + sorted(list(set(corpus))) # 把用于填充空位的特殊token <pad>放在第0位，确保它的id为0
        self.vocab_size = len(self.chars)
        self.char_to_id = {char:id for id, char in enumerate(self.chars)}
        self.id_to_char = {id:char for id, char in enumerate(self.chars)}
    
    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        ids = [[0]+[self.char_to_id[char] for char in text[:511] if char in self.char_to_id] for text in texts] # 在每个文本的开头加一个<pad> token，截断到512字符，跳过未知字符
        # 填充0使得所有序列长度相同，这样才能转化为tensor
        max_len = max(len(id_seq) for id_seq in ids)
        batch_size = len(ids)
        padded_ids = torch.zeros((batch_size, max_len),dtype=torch.long)  # 使用0进行填充，因为0对应<pad> token
        for i, id_seq in enumerate(ids):
            padded_ids[i, :len(id_seq)] = torch.tensor(id_seq)
        return padded_ids
    
    def decode(self, id_seqs):
        if isinstance(id_seqs, torch.Tensor):
            id_seqs = id_seqs.tolist()
        texts = []
        for id_seq in id_seqs:
            chars = [self.id_to_char[id] for id in id_seq if id != 0] # 解码时忽略<pad> token
            text = ''.join(chars)
            texts.append(text)
        return texts
    

class ToyModel(nn.Module):
    def __init__(self, num_blocks, num_heads=8, d_model=512, max_seq_len=4096, vocab_size=10000, sink_size=4, window_size=10):
        super().__init__()
        self.transformer_block = TransformerBlock(num_heads, d_model, max_seq_len=max_seq_len, sink_size=sink_size, window_size=window_size) # 这里保证了各层权重始终相同
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.probe = AttentionProbe()
        self.transformer_block.attention.register_forward_hook(self.probe)
        self.hidden_probe = HiddenStateProbe()
        self.transformer_block.register_forward_hook(self.hidden_probe)
        self.captured_attention = None
        self.sink_size = sink_size
        self.window_size = window_size
        self.embedding = nn.Embedding(vocab_size, d_model)  # 新增：词嵌入层，将离散的 token 转换为连续的向量表示
        # self.embedding的作用: 输入是离散的整数(代表离散的Token)，输出是连续的embedding向量(每个Token对应一个d_model维的向量)，这个映射是可学习的
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # 新增：预测头，将 [batch_size, seq_len, d_model] 的 x 转换成 [batch_size, seq_len, vocab_size] 的 logits，表示每个位置上预测下一个 token 的未归一化的对数概率分布。
        self._layer_caches = {}  # 为每个虚拟层保存独立的 KV cache

    def _load_layer_cache(self, layer_idx):
        attn = self.transformer_block.attention
        if layer_idx in self._layer_caches:
            state = self._layer_caches[layer_idx]
            attn.k_cache = state['k']
            attn.v_cache = state['v']
            attn.seq_len = state['seq_len']
        else:
            attn.k_cache = None
            attn.v_cache = None
            attn.seq_len = 0

    def _save_layer_cache(self, layer_idx):
        attn = self.transformer_block.attention
        self._layer_caches[layer_idx] = {
            'k': attn.k_cache,
            'v': attn.v_cache,
            'seq_len': attn.seq_len,
        }

    def reset_cache(self):
        self._layer_caches.clear()

    def forward(self, input_ids): # input_ids 的形状为 [batch_size, seq_len]
        self.probe.reset() # 每次前向传播前重置 probe，清空上一次的观测数据
        self.hidden_probe.reset()
        x = self.embedding(input_ids) # 将输入的 token ids 转换为嵌入向量,形状为 [batch_size, seq_len, d_model]
        for i in range(self.num_blocks):
            self._load_layer_cache(i)   # 换入第 i 个虚拟层的 cache
            x = self.transformer_block(x)
            self._save_layer_cache(i)   # 换出保存
        self.captured_attention = list(self.probe.captured_data) # 显式copy一份数据，避免后续被修改
        logits = self.lm_head(x) # [batch_size, seq_len, vocab_size]
        return logits

    def forward_hidden(self, input_ids):
        """只返回 transformer 隐藏状态，不经过 lm_head（供分类头使用）"""
        self.probe.reset()
        self.hidden_probe.reset()
        x = self.embedding(input_ids)
        for i in range(self.num_blocks):
            self._load_layer_cache(i)
            x = self.transformer_block(x)
            self._save_layer_cache(i)
        return x

    def forward_from_embedding(self, x):
        """直接接受 embedding 后的输入，跳过 token embedding（供非文本下游任务使用）"""
        self.probe.reset()
        self.hidden_probe.reset()
        for i in range(self.num_blocks):
            self._load_layer_cache(i)
            x = self.transformer_block(x)
            self._save_layer_cache(i)
        return x


class TextClassifier(nn.Module):
    """在预训练语言模型 backbone 上添加文本分类头（冻结 backbone）"""
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        # 冻结 backbone，只训练分类头
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.head = nn.Linear(backbone.d_model, num_classes)

    def forward(self, input_ids):
        with torch.no_grad():
            x = self.backbone.forward_hidden(input_ids).detach()
        # 用最后一个非 padding token 的隐藏状态做池化
        # 因果模型中该位置 attended to 全部前文，天然是序列摘要
        lengths = (input_ids != 0).sum(dim=1)  # [batch]
        batch_idx = torch.arange(input_ids.shape[0], device=input_ids.device)
        pooled = x[batch_idx, lengths - 1]  # [batch, d_model]
        return self.head(pooled)


class HandwrittenClassifier(nn.Module):
    """手写图像分类器：将图像的每一行视为一个 token，通过 backbone transformer 提取特征"""
    def __init__(self, backbone, patch_dim, num_classes):
        super().__init__()
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.patch_embed = nn.Linear(patch_dim, backbone.d_model)
        self.head = nn.Linear(backbone.d_model, num_classes)

    def forward(self, images):
        # images: [batch, seq_len, patch_dim] (e.g., [batch, 28, 28] for MNIST)
        x = self.patch_embed(images)  # [batch, seq_len, d_model]
        with torch.no_grad():
            h = self.backbone.forward_from_embedding(x).detach()
        # 用最后一行（因果模型中它看到了全部 28 行）
        pooled = h[:, -1, :]  # [batch, d_model]
        return self.head(pooled)


