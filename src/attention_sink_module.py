from transformer_module import *
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
class AttentionSinkExperiment:
    def __init__(self, num_blocks, corpus=None, num_heads=8, d_model=512, max_seq_len=4096, learning_rate=3e-4, load_from=None):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"Experiment initialized. Using device: {self.device}")
        if load_from is not None:
            # 加载检查点时，corpus 参数不是必需的
            if corpus is not None:
                print("Warning: corpus parameter ignored when loading from checkpoint")
            # 从检查点加载
            checkpoint = torch.load(load_from, map_location='cpu')
            # 恢复tokenizer
            self.tokenizer = SimpleTokenizer('')  # 临时tokenizer，下面覆盖其内部字典
            self.tokenizer.chars = checkpoint['tokenizer_chars']
            self.tokenizer.char_to_id = checkpoint['tokenizer_char_to_id']
            self.tokenizer.id_to_char = checkpoint['tokenizer_id_to_char']
            self.tokenizer.vocab_size = checkpoint['vocab_size']
            self.vocab_size = checkpoint['vocab_size']
            # 重建模型
            self.model = ToyModel(
                num_blocks=checkpoint['num_blocks'],
                num_heads=checkpoint['num_heads'],
                d_model=checkpoint['d_model'],
                max_seq_len=checkpoint['max_seq_len'],
                vocab_size=checkpoint['vocab_size']
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            # 重建优化器
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # 如果提供了新的学习率，更新优化器中的学习率
            if learning_rate != 3e-4:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = learning_rate
            print(f"Model loaded from {load_from}")
        else:
            if corpus is None:
                raise ValueError("corpus parameter is required when load_from is None")
            self.tokenizer = SimpleTokenizer(corpus)
            self.vocab_size = self.tokenizer.vocab_size
            self.model = ToyModel(num_blocks, num_heads, d_model, max_seq_len, self.vocab_size)
            self.model.to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    
    def train(self, texts_for_train, epochs=100, log_interval=10, save_path=None):
        self.model.train()
        with torch.no_grad():
            original_ids = self.tokenizer.encode(texts_for_train).to(self.device) # [batch_size, seq_len]
            # 根据全部之前的输入预测下一个输出
            input_ids = original_ids[:, :-1]  # 输入序列，去掉最后一个token
            target_ids = original_ids[:, 1:]  # 目标序列，去掉第一个token

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            output_logits = self.model(input_ids) # [batch_size, seq_len-1, vocab_size]
            # cross-entropy loss 期望的输入是 [batch_size * (seq_len-1), vocab_size] 的 output_logits 和 [batch_size * (seq_len-1)] 的 target_ids，所以需要reshape
            loss = self.criterion(output_logits.reshape(-1, self.vocab_size), target_ids.reshape(-1))
            loss.backward()
            self.optimizer.step()
            if epoch % log_interval == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        if save_path is not None:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'tokenizer_chars': self.tokenizer.chars,
                'tokenizer_char_to_id': self.tokenizer.char_to_id,
                'tokenizer_id_to_char': self.tokenizer.id_to_char,
                'num_blocks': self.model.num_blocks,
                'num_heads': self.model.num_heads,
                'd_model': self.model.d_model,
                'max_seq_len': self.model.max_seq_len,
                'vocab_size': self.model.vocab_size
            }
            torch.save(checkpoint, save_path)
            print(f"Model saved to {save_path}")

    def visualize_attention(self, text_for_test, layer_idx=-1, head_idx='mean'):
        self.model.eval()
        with torch.no_grad():
            input_ids = self.tokenizer.encode(text_for_test).to(self.device)  # [1, seq_len]
            _ = self.model(input_ids)  
            attention = self.model.captured_attention[layer_idx]  # [1, num_heads, seq_len, seq_len]
            if isinstance(head_idx, int):
                attention_2d = attention[0, head_idx, :, :]  # [seq_len, seq_len]
            elif head_idx == 'mean':
                attention_2d = attention.mean(axis=1)[0] # [seq_len, seq_len]

            # 可视化 attention_2d，例如使用 seaborn 的 heatmap         
            tokens = [self.tokenizer.id_to_char[id.item()] for id in input_ids[0]]  
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                attention_2d, 
                cmap="viridis",        # 使用天文/物理界常用的翠绿色调，对数值渐变非常敏感
                square=True,           # 强制每个单元格为正方形
                xticklabels=tokens,    # X 轴标签（Key）
                yticklabels=tokens,    # Y 轴标签（Query）
                cbar_kws={"shrink": .8}# 缩小一点颜色条，更美观
            )
            head_title = f"Head {head_idx}" if isinstance(head_idx, int) else "Mean of All Heads"
            # 适应中文字体
            plt.rcParams['font.sans-serif'] = ['Songti SC']  # 设置中文字体为宋体
            plt.title(f"Attention Map (Layer {layer_idx}, {head_title})")
            plt.xlabel("Key")
            plt.ylabel("Query")
            plt.xticks(rotation=0, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.show()

    def generate(self, text_for_prompt, max_new_tokens=10):
        self.model.eval()
        with torch.no_grad():
            input_ids = self.tokenizer.encode(text_for_prompt).to(self.device)  # [1, seq_len]
            for _ in range(max_new_tokens):
                output_logits = self.model(input_ids)  # [1, seq_len, vocab_size]
                next_token_logits = output_logits[:, -1, :]  # [1, vocab_size]
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)  # [1, 1]
                input_ids = torch.cat((input_ids, next_token_id), dim=1)  # 将新预测的 token id 添加到输入序列末尾
            generated_text = self.tokenizer.decode(input_ids.cpu())  # 解码成文本
            return generated_text[0]