from transformer_module import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
class AttentionSinkExperiment:
    def __init__(self, num_blocks, corpus=None, num_heads=8, d_model=512, max_seq_len=4096, learning_rate=3e-4, sink_size=4, window_size=10, load_from=None, log_dir=None):
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
                vocab_size=checkpoint['vocab_size'],
                sink_size=checkpoint['sink_size'],
                window_size=checkpoint['window_size']
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
            self.model = ToyModel(num_blocks, num_heads, d_model, max_seq_len, self.vocab_size, sink_size=sink_size, window_size=window_size)
            self.model.to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
        if log_dir is not None:
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None
    
    def train(self, texts_for_train, batch_size=8, epochs=100, log_interval=10, epoch_interval=5, save_path=None):
        self.model.train()
        with torch.no_grad():
            original_ids = self.tokenizer.encode(texts_for_train) # [batch_size, seq_len]
            dataset_size = original_ids.shape[0]

        for epoch in range(epochs):
            # 每次 Epoch 打乱数据顺序
            indices = torch.randperm(dataset_size)
            original_ids = original_ids[indices]
            epoch_loss = 0 # 记录当前 epoch 的总损失
            num_batches = 0 # 记录当前 epoch 处理的 batch 数量
            # 引入 Mini-batch 内部循环, 每次处理 batch_size 个样本
            for i in range(0, dataset_size, batch_size):
                batch_ids = original_ids[i : i + batch_size].to(self.device)  # [batch_size, seq_len]  
                input_ids = batch_ids[:, :-1]  # [batch_size, seq_len-1]
                target_ids = batch_ids[:, 1:]  # [batch_size, seq_len-1]
                self.optimizer.zero_grad()
                output_logits = self.model(input_ids) # [batch_size, seq_len-1, vocab_size]
                # cross-entropy loss 期望的输入是 [batch_size * (seq_len-1), vocab_size] 的 output_logits 和 [batch_size * (seq_len-1)] 的 target_ids，所以需要reshape
                loss = self.criterion(output_logits.reshape(-1, self.vocab_size), target_ids.reshape(-1))
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() # 累加当前 batch 的损失
                num_batches += 1
                if i % log_interval == 0:
                    self.writer.add_scalar('Loss/train', loss.item(), epoch * (dataset_size // batch_size) + num_batches) if self.writer is not None else None
                    print(f"  Step {num_batches} Loss: {loss.item():.4f}")
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
                            'vocab_size': self.model.vocab_size,
                            'sink_size': self.model.sink_size,
                            'window_size': self.model.window_size
                        }
                        torch.save(checkpoint, save_path)

            avg_loss = epoch_loss / num_batches # 计算当前 epoch 的平均损失
            if epoch % epoch_interval == 0:
                print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
                self.writer.add_scalar('Loss/epoch', avg_loss, epoch) if self.writer is not None else None

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
                            'vocab_size': self.model.vocab_size,
                            'sink_size': self.model.sink_size,
                            'window_size': self.model.window_size
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
            self.model.reset_cache()  # 清空旧 cache，开始新的生成
            input_ids = self.tokenizer.encode(text_for_prompt).to(self.device)  # [1, seq_len]
            # Prefill: 一次性处理完整 prompt
            output_logits = self.model(input_ids)  # [1, seq_len, vocab_size]
            next_token_logits = output_logits[:, -1, :]  # [1, vocab_size]
            next_token_logits[:, 0] = float('-inf')  # 禁止生成 padding token
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)  # [1, 1]
            generated_tokens = [next_token_id]
            # Decode: 每次只传入最新 token，利用 KV cache + sink tokens
            for _ in range(max_new_tokens - 1):
                output_logits = self.model(next_token_id)  # seq_len=1, 走 streaming 解码路径
                next_token_logits = output_logits[:, -1, :]
                next_token_logits[:, 0] = float('-inf')
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                generated_tokens.append(next_token_id)
            all_ids = torch.cat([input_ids] + generated_tokens, dim=1)
            generated_text = self.tokenizer.decode(all_ids.cpu())
            return generated_text[0]

    def evaluate_ppl(self, texts, batch_size=32):
        """在给定文本上计算困惑度 (Perplexity)"""
        self.model.eval()
        self.model.reset_cache()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            all_ids = self.tokenizer.encode(texts).to(self.device)
            for i in range(0, all_ids.shape[0], batch_size):
                batch_ids = all_ids[i:i + batch_size]
                input_ids = batch_ids[:, :-1]
                target_ids = batch_ids[:, 1:]
                logits = self.model(input_ids)
                loss_per_token = F.cross_entropy(
                    logits.reshape(-1, self.vocab_size),
                    target_ids.reshape(-1),
                    reduction='none'
                )
                mask = (target_ids != 0).reshape(-1)
                total_loss += loss_per_token[mask].sum().item()
                total_tokens += mask.sum().item()
        avg_nll = total_loss / max(total_tokens, 1)
        ppl = math.exp(avg_nll)
        print(f"PPL: {ppl:.2f} (evaluated on {total_tokens} non-padding tokens)")
        return ppl

    def evaluate_cloze(self, texts, batch_size=32):
        """评估字符级填空准确率：给定上文预测下一个字符"""
        self.model.eval()
        self.model.reset_cache()
        correct = 0
        total = 0
        with torch.no_grad():
            all_ids = self.tokenizer.encode(texts).to(self.device)
            for i in range(0, all_ids.shape[0], batch_size):
                batch_ids = all_ids[i:i + batch_size]
                input_ids = batch_ids[:, :-1]
                target_ids = batch_ids[:, 1:]
                logits = self.model(input_ids)
                mask = (target_ids != 0)
                preds = logits.argmax(dim=-1)
                correct += ((preds == target_ids) & mask).sum().item()
                total += mask.sum().item()
        accuracy = correct / max(total, 1)
        print(f"Cloze Accuracy: {accuracy:.4f} ({correct}/{total} tokens)")
        return accuracy

    def evaluate_sink_rate(self, texts, batch_size=32):
        """评估注意力沉降率：sink tokens 获得的平均注意力比例"""
        self.model.eval()
        self.model.reset_cache()
        sink_size = self.model.sink_size

        all_ids = self.tokenizer.encode(texts).to(self.device)
        layer_sink_rates = []  # [batch_count, num_layers]

        with torch.no_grad():
            for i in range(0, all_ids.shape[0], batch_size):
                batch_ids = all_ids[i:i + batch_size]
                _ = self.model(batch_ids)
                batch_rates = []
                for attn in self.model.captured_attention:
                    # attn: [batch, num_heads, seq_len, seq_len] (numpy)
                    seq_len = attn.shape[-1]
                    # 每个 query 位置对前 sink_size 个 key 的注意力之和
                    sink_attn = attn[:, :, :, :min(sink_size, seq_len)].sum(axis=-1)
                    rate = sink_attn.mean().item()
                    batch_rates.append(rate)
                layer_sink_rates.append(batch_rates)

        n_layers = len(layer_sink_rates[0])
        per_layer = [sum(b[i] for b in layer_sink_rates) / len(layer_sink_rates) for i in range(n_layers)]
        overall = sum(per_layer) / len(per_layer)
        print(f"Sink Rate (overall, sink_size={sink_size}): {overall:.4f}")
        for i, r in enumerate(per_layer):
            print(f"  Layer {i}: {r:.4f}")
        return overall, per_layer

    def train_text_classification(self, train_texts, train_labels, num_classes, val_texts=None, val_labels=None, epochs=10, batch_size=32, lr=1e-3, save_path=None):
        """训练文本分类头（冻结 backbone），支持验证集监控，可选保存到 .pth"""
        from sklearn.metrics import f1_score, accuracy_score

        self.model.eval()
        self.model.reset_cache()
        classifier = TextClassifier(self.model, num_classes).to(self.device)
        optimizer = torch.optim.Adam(classifier.head.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        # 编码训练数据
        train_ids = self.tokenizer.encode(train_texts).to(self.device)
        train_labels_t = torch.tensor(train_labels, dtype=torch.long).to(self.device)

        # 编码验证数据（如果提供）
        val_ids = None
        if val_texts is not None:
            val_ids = self.tokenizer.encode(val_texts).to(self.device)

        # 训练分类头
        classifier.train()
        best_val_f1 = 0.0
        for epoch in range(epochs):
            indices = torch.randperm(len(train_labels_t), device=self.device)
            epoch_loss = 0.0
            for i in range(0, len(indices), batch_size):
                idx = indices[i:i + batch_size]
                logits = classifier(train_ids[idx])
                loss = criterion(logits, train_labels_t[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # 验证集评估
            if val_ids is not None:
                classifier.eval()
                val_preds = []
                with torch.no_grad():
                    for i in range(0, len(val_ids), batch_size):
                        logits = classifier(val_ids[i:i + batch_size])
                        preds = torch.argmax(logits, dim=-1)
                        val_preds.extend(preds.cpu().tolist())
                val_f1 = f1_score(val_labels, val_preds, average='macro')
                val_acc = accuracy_score(val_labels, val_preds)
                print(f"  Epoch {epoch + 1}/{epochs}, train_loss: {epoch_loss:.4f}, val_f1: {val_f1:.4f}, val_acc: {val_acc:.4f}")
                # 保存最佳模型
                if save_path is not None and val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    torch.save({
                        'head_state_dict': classifier.head.state_dict(),
                        'num_classes': num_classes,
                        'type': 'text',
                    }, save_path)
                    print(f"  Best model saved to {save_path} (val_f1: {val_f1:.4f})")
                classifier.train()
            else:
                print(f"  Epoch {epoch + 1}/{epochs}, train_loss: {epoch_loss:.4f}")

        # 如果没有验证集，在训练结束后保存最终模型
        if save_path is not None and val_ids is None:
            torch.save({
                'head_state_dict': classifier.head.state_dict(),
                'num_classes': num_classes,
                'type': 'text',
            }, save_path)
            print(f"  Classification head saved to {save_path}")

        return classifier

    def eval_text_classification(self, test_texts, test_labels, num_classes, head_path=None, batch_size=32):
        """加载分类头并在测试集上评估"""
        from sklearn.metrics import f1_score, accuracy_score

        self.model.eval()
        self.model.reset_cache()

        classifier = TextClassifier(self.model, num_classes).to(self.device)
        if head_path is not None:
            checkpoint = torch.load(head_path, map_location=self.device, weights_only=True)
            classifier.head.load_state_dict(checkpoint['head_state_dict'])
            print(f"Classification head loaded from {head_path}")

        test_ids = self.tokenizer.encode(test_texts).to(self.device)

        classifier.eval()
        all_preds = []
        with torch.no_grad():
            for i in range(0, len(test_ids), batch_size):
                logits = classifier(test_ids[i:i + batch_size])
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().tolist())

        f1 = f1_score(test_labels, all_preds, average='macro')
        acc = accuracy_score(test_labels, all_preds)
        print(f"Text Classification Results — F1 (macro): {f1:.4f}, Accuracy: {acc:.4f}")
        return {'f1': f1, 'accuracy': acc, 'predictions': all_preds}

    def train_handwritten_classification(self, train_images, train_labels, patch_dim, num_classes, val_images=None, val_labels=None, epochs=10, batch_size=32, lr=1e-3, save_path=None):
        """训练手写图像分类头（冻结 backbone，训练 patch_embed + head）"""
        from sklearn.metrics import f1_score, accuracy_score

        self.model.eval()
        self.model.reset_cache()
        classifier = HandwrittenClassifier(self.model, patch_dim, num_classes).to(self.device)
        optimizer = torch.optim.Adam(
            list(classifier.patch_embed.parameters()) + list(classifier.head.parameters()),
            lr=lr
        )
        criterion = nn.CrossEntropyLoss()

        train_images = train_images.to(self.device)
        train_labels_t = torch.tensor(train_labels, dtype=torch.long).to(self.device)

        best_val_f1 = 0.0
        history = {'train_loss': [], 'val_f1': [], 'val_acc': []}
        for epoch in range(epochs):
            indices = torch.randperm(len(train_labels_t), device=self.device)
            epoch_loss = 0.0
            for i in range(0, len(indices), batch_size):
                idx = indices[i:i + batch_size]
                logits = classifier(train_images[idx])
                loss = criterion(logits, train_labels_t[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if val_images is not None:
                classifier.eval()
                val_images_dev = val_images.to(self.device)
                val_preds = []
                with torch.no_grad():
                    for i in range(0, len(val_images_dev), batch_size):
                        logits = classifier(val_images_dev[i:i + batch_size])
                        preds = torch.argmax(logits, dim=-1)
                        val_preds.extend(preds.cpu().tolist())
                val_f1 = f1_score(val_labels, val_preds, average='macro')
                val_acc = accuracy_score(val_labels, val_preds)
                history['train_loss'].append(epoch_loss)
                history['val_f1'].append(val_f1)
                history['val_acc'].append(val_acc)
                print(f"  Epoch {epoch + 1}/{epochs}, train_loss: {epoch_loss:.4f}, val_f1: {val_f1:.4f}, val_acc: {val_acc:.4f}")
                if save_path is not None and val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    torch.save({
                        'patch_embed_state_dict': classifier.patch_embed.state_dict(),
                        'head_state_dict': classifier.head.state_dict(),
                        'patch_dim': patch_dim,
                        'num_classes': num_classes,
                        'type': 'handwritten',
                    }, save_path)
                    print(f"  Best model saved to {save_path} (val_f1: {val_f1:.4f})")
                classifier.train()
            else:
                print(f"  Epoch {epoch + 1}/{epochs}, train_loss: {epoch_loss:.4f}")

        if save_path is not None and val_images is None:
            torch.save({
                'patch_embed_state_dict': classifier.patch_embed.state_dict(),
                'head_state_dict': classifier.head.state_dict(),
                'patch_dim': patch_dim,
                'num_classes': num_classes,
                'type': 'handwritten',
            }, save_path)
            print(f"  Handwritten classification head saved to {save_path}")

        return classifier, history

    def eval_handwritten_classification(self, test_images, test_labels, patch_dim, num_classes, head_path=None, batch_size=32):
        """加载手写分类头并评估"""
        from sklearn.metrics import f1_score, accuracy_score

        self.model.eval()
        self.model.reset_cache()

        classifier = HandwrittenClassifier(self.model, patch_dim, num_classes).to(self.device)
        if head_path is not None:
            checkpoint = torch.load(head_path, map_location=self.device, weights_only=True)
            classifier.patch_embed.load_state_dict(checkpoint['patch_embed_state_dict'])
            classifier.head.load_state_dict(checkpoint['head_state_dict'])
            print(f"Handwritten classification head loaded from {head_path}")

        test_images = test_images.to(self.device)

        classifier.eval()
        all_preds = []
        with torch.no_grad():
            for i in range(0, len(test_images), batch_size):
                logits = classifier(test_images[i:i + batch_size])
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().tolist())

        f1 = f1_score(test_labels, all_preds, average='macro')
        acc = accuracy_score(test_labels, all_preds)
        print(f"Handwritten Classification Results — F1 (macro): {f1:.4f}, Accuracy: {acc:.4f}")
        return {'f1': f1, 'accuracy': acc, 'predictions': all_preds}