from attention_sink_module import AttentionSinkExperiment
import warnings
import os
import random
import numpy as np
import torch
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
os.environ["HF_DATASETS_OFFLINE"] = "1"  # 强制使用本地数据集，避免下载
from datasets import load_dataset
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn.utils") # 忽略 seaborn 的中文字体警告，避免干扰输出
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager") # 忽略 matplotlib 的中文字体警告，避免干扰输出
raw_datasets = load_dataset("wikitext", "wikitext-103-v1")
# 在训练集中随机选取一些文本进行训练，确保文本长度适中，并构建字符集
idx = random.sample(range(len(raw_datasets['train'])), 1000)  # 从训练集中随机选取100条文本
texts_for_train = [raw_datasets['train']['text'][i] for i in idx if len(raw_datasets['train']['text'][i].strip()) > 10]
corpus = "".join(set("".join(texts_for_train)))
corpus += "<pad>" # 确保包含你在 SimpleTokenizer 中需要的特殊 token
if __name__ == "__main__":
    # 如果之前已经训练过模型并保存了检查点，就从检查点继续训练，否则从头开始训练
    if os.path.exists("attention_sink_checkpoint.pth"):
        load_from = "attention_sink_checkpoint.pth"
    else:
        load_from = None

    # 创建实验对象
    experiment = AttentionSinkExperiment(num_blocks=6, corpus=corpus, load_from=load_from, num_heads=8, d_model=512, learning_rate=1e-4, sink_size=1, window_size=30, log_dir="runs/attention_sink_experiment")

    # ====== 使用 validation split（模型训练时完全未见过）======
    val_texts = [t for t in raw_datasets['validation']['text'] if len(t.strip()) > 10]
    val_texts = random.sample(val_texts, min(200, len(val_texts)))

    # ====== 评估 ======
    print("\n=== Perplexity Evaluation ===")
    ppl = experiment.evaluate_ppl(val_texts)

    print("\n=== Cloze Evaluation ===")
    cloze_acc = experiment.evaluate_cloze(val_texts)

    print("\n=== Sink Rate Evaluation ===")
    sink_overall, sink_per_layer = experiment.evaluate_sink_rate(val_texts)

    # ====== Scaling Law ======
    print("\n=== Scaling Law Experiment ===")
    scaling = {'num_blocks': [], 'ppl': [], 'cloze_acc': [], 'sink_rate': []}
    for n_blocks in [1, 2, 3, 4, 5, 6]:
        print(f"\n--- num_blocks = {n_blocks} ---")
        experiment.model.num_blocks = n_blocks
        experiment.model.reset_cache()
        s_ppl = experiment.evaluate_ppl(val_texts)
        s_acc = experiment.evaluate_cloze(val_texts)
        s_sink, _ = experiment.evaluate_sink_rate(val_texts)
        scaling['num_blocks'].append(n_blocks)
        scaling['ppl'].append(s_ppl)
        scaling['cloze_acc'].append(s_acc)
        scaling['sink_rate'].append(s_sink)

    experiment.model.num_blocks = 6  # 恢复

    # ====== 拟合分析 ======
    L = np.array(scaling['num_blocks'], dtype=float)
    PPL = np.array(scaling['ppl'])
    ACC = np.array(scaling['cloze_acc'])
    SINK = np.array(scaling['sink_rate'])

    # A. PPL 幂律拟合: PPL(L) = A * L^(-alpha) + C
    def ppl_power_law(x, A, alpha, C):
        return A * np.power(x, -alpha) + C
    popt_ppl, _ = curve_fit(ppl_power_law, L, PPL, p0=[50, 1.5, 5], maxfev=10000)
    A_fit, alpha_fit, C_fit = popt_ppl
    print(f"\n=== A. PPL 幂律拟合: PPL(L) = A·L^(-α) + C ===")
    print(f"  A = {A_fit:.4f}, α = {alpha_fit:.4f}, C = {C_fit:.4f}")
    print(f"  不可还原熵 C = {C_fit:.2f} (模型深度→∞时的 PPL 下界)")

    # B. Accuracy vs ln(1/PPL): Acc = β·ln(1/PPL) + K
    inv_ppl = np.log(1.0 / PPL)
    coeffs_b = np.polyfit(inv_ppl, ACC, 1)
    beta_fit, K_fit = coeffs_b
    # 逐段计算 β（PPL 每下降1单位能换多少准确率）
    print(f"\n=== B. PPL-Accuracy 关系: Acc = β·ln(1/PPL) + K ===")
    print(f"  β = {beta_fit:.4f}, K = {K_fit:.4f}")
    print(f"  转换效率: PPL 每下降 1 单位 → 准确率提升约 {beta_fit / np.mean(PPL):.4f}")
    # 逐段 β
    for i in range(1, len(PPL)):
        d_acc = ACC[i] - ACC[i-1]
        d_inv_ppl = inv_ppl[i] - inv_ppl[i-1]
        local_beta = d_acc / d_inv_ppl if d_inv_ppl != 0 else 0
        print(f"  L={L[i-1]:.0f}→{L[i]:.0f}: 局部β = {local_beta:.4f}" +
              (" ← 饱和" if local_beta < beta_fit * 0.5 else ""))

    # C. Sink Rate 指数衰减: SinkRate(L) = S0·exp(-λL) + k
    def sink_exp(x, S0, lam, k):
        return S0 * np.exp(-lam * x) + k
    popt_sink, _ = curve_fit(sink_exp, L, SINK, p0=[0.012, 0.3, 0.004], maxfev=10000)
    S0_fit, lam_fit, k_fit = popt_sink
    print(f"\n=== C. Sink Rate 衰减场: SinkRate(L) = S₀·exp(-λL) + k ===")
    print(f"  S₀ = {S0_fit:.6f}, λ = {lam_fit:.4f}, k = {k_fit:.6f}")
    print(f"  衰减系数 λ = {lam_fit:.4f} (特征深度 1/λ = {1/lam_fit:.2f} 层)")

    # ====== 绘图 ======
    plt.rcParams['font.sans-serif'] = ['Songti SC']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    L_fit = np.linspace(0.8, 6.5, 200)

    # 1. PPL 幂律拟合
    axes[0, 0].plot(L, PPL, 'go', markersize=10, label='实测 PPL', zorder=5)
    axes[0, 0].plot(L_fit, ppl_power_law(L_fit, *popt_ppl), 'g--', linewidth=2,
                    label=f'拟合: A·L$^{{-α}}$+C\nα={alpha_fit:.2f}, C={C_fit:.2f}')
    axes[0, 0].axhline(y=C_fit, color='gray', linestyle=':', alpha=0.7, label=f'不可还原熵 C={C_fit:.2f}')
    axes[0, 0].set_xlabel('num_blocks (L)')
    axes[0, 0].set_ylabel('PPL')
    axes[0, 0].set_title('A. PPL 幂律拟合')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].set_xticks(scaling['num_blocks'])

    # 2. Accuracy vs ln(1/PPL) 线性关系
    axes[0, 1].plot(inv_ppl, ACC, 'ro', markersize=10, label='实测', zorder=5)
    x_line = np.linspace(min(inv_ppl) - 0.1, max(inv_ppl) + 0.1, 100)
    axes[0, 1].plot(x_line, beta_fit * x_line + K_fit, 'r--', linewidth=2,
                    label=f'拟合: β·ln(1/PPL)+K\nβ={beta_fit:.3f}')
    # 标注每个点对应的 num_blocks
    for i, n in enumerate(scaling['num_blocks']):
        axes[0, 1].annotate(f'L={n}', (inv_ppl[i], ACC[i]), textcoords="offset points",
                            xytext=(8, -5), fontsize=9, color='darkred')
    axes[0, 1].set_xlabel('ln(1/PPL)')
    axes[0, 1].set_ylabel('填空准确率')
    axes[0, 1].set_title('B. PPL→准确率 转换效率')
    axes[0, 1].legend(fontsize=9)

    # 3. PPL + Accuracy 合并 (双 y 轴)
    ax3a = axes[1, 0]
    ax3b = ax3a.twinx()
    l1, = ax3a.plot(L, PPL, 'g-o', linewidth=2, markersize=8, label='PPL')
    l2, = ax3b.plot(L, ACC, 'r-o', linewidth=2, markersize=8, label='填空准确率')
    ax3a.set_xlabel('num_blocks')
    ax3a.set_ylabel('PPL', color='g')
    ax3b.set_ylabel('填空准确率', color='r')
    ax3a.set_title('Scaling Law: PPL & 填空准确率')
    ax3a.set_xticks(scaling['num_blocks'])
    ax3a.legend(handles=[l1, l2], loc='center right')

    # 4. Sink Rate 指数衰减拟合
    axes[1, 1].plot(L, SINK, 'bo', markersize=10, label='实测 Sink Rate', zorder=5)
    axes[1, 1].plot(L_fit, sink_exp(L_fit, *popt_sink), 'b--', linewidth=2,
                    label=f'拟合: S₀·e$^{{-λL}}$+k\nλ={lam_fit:.3f}, 1/λ={1/lam_fit:.1f}层')
    axes[1, 1].axhline(y=k_fit, color='gray', linestyle=':', alpha=0.7, label=f'渐近基线 k={k_fit:.4f}')
    axes[1, 1].set_xlabel('num_blocks (L)')
    axes[1, 1].set_ylabel('Sink Rate')
    axes[1, 1].set_title('C. Sink Rate 衰减场')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].set_xticks(scaling['num_blocks'])

    plt.tight_layout()
    plt.savefig('downstream_evaluation.png', dpi=150)
    print('\nPlot saved to downstream_evaluation.png')
    plt.show()

    # ====== Token Cluster Animation ======
    print("\n=== Token Cluster Evolution (层 = 时间) ===")
    from sklearn.decomposition import PCA
    from matplotlib.animation import FuncAnimation

    # 使用 validation 文本
    cluster_texts = val_texts[:30]
    experiment.model.eval()
    experiment.model.reset_cache()

    with torch.no_grad():
        all_ids = experiment.tokenizer.encode(cluster_texts).to(experiment.device)
        # 先获取 embedding 层输出
        embed_out = experiment.model.embedding(all_ids).cpu().numpy()  # [batch, seq, d_model]
        # 前向传播，hidden_probe 自动捕获每层输出
        _ = experiment.model(all_ids)
        layer_outputs = [t.numpy() for t in experiment.model.hidden_probe.captured_data]

    n_layers = len(layer_outputs) + 1  # +1 for embedding
    print(f"Captured {n_layers} stages (embedding + {len(layer_outputs)} transformer layers)")

    # 获取非 padding 的 token 位置
    mask = (all_ids != 0).cpu().numpy()
    valid_tokens = [(b, s) for b in range(all_ids.shape[0]) for s in range(all_ids.shape[1]) if mask[b, s]]
    print(f"Total valid tokens: {len(valid_tokens)}")

    # 随机采样 2000 个 token
    if len(valid_tokens) > 2000:
        random.shuffle(valid_tokens)
        sample = valid_tokens[:2000]
    else:
        sample = valid_tokens

    # 提取 token ID（用于着色）
    token_ids = np.array([all_ids[b, s].item() for b, s in sample])
    # 将 token 映射到字符类别
    def char_category(tid):
        ch = experiment.tokenizer.id_to_char.get(tid, '')
        if ch.isupper(): return 0
        if ch.islower(): return 1
        if ch.isdigit(): return 2
        if ch == ' ': return 3
        return 4  # 标点/特殊
    categories = np.array([char_category(tid) for tid in token_ids])

    # 提取每个 stage 的隐藏状态
    all_stages = [embed_out] + layer_outputs
    per_layer_states = []
    for stage in all_stages:
        states = np.array([stage[b, s] for b, s in sample])
        per_layer_states.append(states)

    # L2 归一化：只保留方向，消除残差连接导致的模长增长
    per_layer_normed = []
    for s in per_layer_states:
        norms = np.linalg.norm(s, axis=1, keepdims=True).clip(min=1e-8)
        per_layer_normed.append(s / norms)

    # PCA: 在归一化后的所有 stage 数据上联合拟合
    all_concat = np.concatenate(per_layer_normed, axis=0)
    pca = PCA(n_components=2)
    pca.fit(all_concat)
    per_layer_2d = [pca.transform(s) for s in per_layer_normed]
    print(f"PCA explained variance: {pca.explained_variance_ratio_}")

    # 统一的坐标轴范围
    all_x = np.concatenate([p[:, 0] for p in per_layer_2d])
    all_y = np.concatenate([p[:, 1] for p in per_layer_2d])
    pad_x = 0.1 * (all_x.max() - all_x.min())
    pad_y = 0.1 * (all_y.max() - all_y.min())

    # 颜色映射
    cat_colors = {0: '#e41a1c', 1: '#377eb8', 2: '#4daf4a', 3: '#984ea3', 4: '#ff7f00'}
    cat_labels = {0: '大写字母', 1: '小写字母', 2: '数字', 3: '空格', 4: '标点'}
    point_colors = [cat_colors[c] for c in categories]

    # 创建动图
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.rcParams['font.sans-serif'] = ['Songti SC']

    def update(frame):
        ax.clear()
        for cat_id in sorted(cat_colors.keys()):
            idx = categories == cat_id
            ax.scatter(per_layer_2d[frame][idx, 0], per_layer_2d[frame][idx, 1],
                       c=cat_colors[cat_id], s=6, alpha=0.6, label=cat_labels[cat_id])
        name = 'Embedding' if frame == 0 else f'Transformer Block {frame}'
        ax.set_title(f'Layer {frame} — {name}', fontsize=14)
        ax.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
        ax.set_ylim(all_y.min() - pad_y, all_y.max() + pad_y)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.legend(loc='upper right', fontsize=9, markerscale=3)
        return []

    ani = FuncAnimation(fig, update, frames=n_layers, interval=800, blit=False)
    gif_path = 'token_cluster_evolution.gif'
    ani.save(gif_path, writer='pillow', fps=1.5)
    print(f'Animation saved to {gif_path}')

    # ====== 分析 PC1/PC2 的语义含义 ======
    print("\n=== PCA 轴语义分析（最终层 Layer 6）===")
    final_2d = per_layer_2d[-1]  # 取最后一层的 2D 投影

    # 分析 PC1：取出极端 token
    pc1_sorted = np.argsort(final_2d[:, 0])
    print("\nPC1 (横轴) 极端 token:")
    print("  PC1 最小（左侧）:", [experiment.tokenizer.id_to_char.get(token_ids[i].item(), '?') for i in pc1_sorted[:15]])
    print("  PC1 最大（右侧）:", [experiment.tokenizer.id_to_char.get(token_ids[i].item(), '?') for i in pc1_sorted[-15:]])

    # 分析 PC2
    pc2_sorted = np.argsort(final_2d[:, 1])
    print("\nPC2 (纵轴) 极端 token:")
    print("  PC2 最小（底部）:", [experiment.tokenizer.id_to_char.get(token_ids[i].item(), '?') for i in pc2_sorted[:15]])
    print("  PC2 最大（顶部）:", [experiment.tokenizer.id_to_char.get(token_ids[i].item(), '?') for i in pc2_sorted[-15:]])

    # 各类别在 PC1/PC2 上的均值
    print("\n各类别在 PC1/PC2 上的质心:")
    for cat_id in sorted(cat_colors.keys()):
        idx = categories == cat_id
        c1 = final_2d[idx, 0].mean()
        c2 = final_2d[idx, 1].mean()
        n = idx.sum()
        print(f"  {cat_labels[cat_id]} (n={n}): PC1={c1:.2f}, PC2={c2:.2f}")
