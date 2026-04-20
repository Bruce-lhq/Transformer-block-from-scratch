import random
import numpy as np
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
matplotlib.rcParams['font.sans-serif'] = ['Songti SC']


def run_animation(experiment, val_texts):
    """Token Cluster 动画：把 token 当粒子，层数当时间，可视化聚类演化"""
    print("\n=== Token Cluster Evolution (层 = 时间) ===")

    # 使用 validation 文本
    cluster_texts = val_texts[:30]
    experiment.model.eval()
    experiment.model.reset_cache()

    with torch.no_grad():
        all_ids = experiment.tokenizer.encode(cluster_texts).to(experiment.device)
        embed_out = experiment.model.embedding(all_ids).cpu().numpy()
        _ = experiment.model(all_ids)
        layer_outputs = [t.numpy() for t in experiment.model.hidden_probe.captured_data]

    n_layers = len(layer_outputs) + 1  # +1 for embedding
    print(f"Captured {n_layers} stages (embedding + {len(layer_outputs)} transformer layers)")

    # 获取非 padding 的 token 位置
    mask = (all_ids != 0).cpu().numpy()
    # 每个序列位置 0 是 <pad>，即模型真正的 Sink Token
    sink_positions = set((b, 0) for b in range(all_ids.shape[0]))
    valid_tokens = [(b, s) for b in range(all_ids.shape[0]) for s in range(all_ids.shape[1]) if mask[b, s]]
    # 把 sink token 也加入采样池
    all_sample_pool = list(sink_positions) + valid_tokens
    print(f"Total valid tokens: {len(valid_tokens)}, Sink tokens: {len(sink_positions)}")

    # 随机采样 2000 个 token（优先保留所有 sink token）
    sink_list = list(sink_positions)
    if len(all_sample_pool) > 2000:
        remaining_quota = 2000 - len(sink_list)
        random.shuffle(valid_tokens)
        sample = sink_list + valid_tokens[:max(0, remaining_quota)]
    else:
        sample = all_sample_pool

    # 提取 token ID（用于着色）
    token_ids = np.array([all_ids[b, s].item() for b, s in sample])

    # 将 token 映射到字符类别（类别 5: Sink Token = 位置 0 的 <pad>）
    def char_category(tid, pos):
        if pos in sink_positions:
            return 5  # Sink Token（优先级最高）
        ch = experiment.tokenizer.id_to_char.get(tid, '')
        if ch.isupper():
            return 0
        if ch.islower():
            return 1
        if ch.isdigit():
            return 2
        if ch == ' ':
            return 3
        return 4  # 标点/特殊

    categories = np.array([char_category(tid, (b, s)) for tid, (b, s) in zip(token_ids, sample)])

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

    # 颜色映射（新增句首 Sink）
    cat_colors = {
        0: '#e41a1c',  # 红 - 大写字母
        1: '#377eb8',  # 蓝 - 小写字母
        2: '#4daf4a',  # 绿 - 数字
        3: '#984ea3',  # 紫 - 空格
        4: '#ff7f00',  # 橙 - 标点
        5: '#a65628',  # 棕 - Sink Token（用星形标记）
    }
    cat_labels = {
        0: '大写字母', 1: '小写字母', 2: '数字',
        3: '空格', 4: '标点', 5: 'Sink Token',
    }
    cat_markers = {
        0: 'o', 1: 'o', 2: 'o', 3: 'o', 4: 'o',
        5: '*',  # 句首 Sink 用星形
    }
    cat_sizes = {
        0: 6, 1: 6, 2: 6, 3: 6, 4: 6,
        5: 60,  # 句首 Sink 更大
    }

    # 创建动图
    fig, ax = plt.subplots(figsize=(10, 8))

    def update(frame):
        ax.clear()
        for cat_id in sorted(cat_colors.keys()):
            idx = categories == cat_id
            if idx.sum() == 0:
                continue
            ax.scatter(
                per_layer_2d[frame][idx, 0], per_layer_2d[frame][idx, 1],
                c=cat_colors[cat_id], s=cat_sizes[cat_id], alpha=0.6,
                marker=cat_markers[cat_id], label=cat_labels[cat_id],
            )
        name = 'Embedding' if frame == 0 else f'Transformer Block {frame}'
        ax.set_title(f'Layer {frame} — {name}', fontsize=14)
        ax.set_xlim(all_x.min() - pad_x, all_x.max() + pad_x)
        ax.set_ylim(all_y.min() - pad_y, all_y.max() + pad_y)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.legend(loc='upper right', fontsize=9, markerscale=2)
        return []

    ani = FuncAnimation(fig, update, frames=n_layers, interval=800, blit=False)
    gif_path = 'token_cluster_evolution.gif'
    ani.save(gif_path, writer='pillow', fps=1.5)
    print(f'Animation saved to {gif_path}')

    # ====== 分析 PC1/PC2 的语义含义 ======
    print("\n=== PCA 轴语义分析（最终层 Layer 6）===")
    final_2d = per_layer_2d[-1]

    pc1_sorted = np.argsort(final_2d[:, 0])
    print("\nPC1 (横轴) 极端 token:")
    print("  PC1 最小（左侧）:", [experiment.tokenizer.id_to_char.get(token_ids[i].item(), '?') for i in pc1_sorted[:15]])
    print("  PC1 最大（右侧）:", [experiment.tokenizer.id_to_char.get(token_ids[i].item(), '?') for i in pc1_sorted[-15:]])

    pc2_sorted = np.argsort(final_2d[:, 1])
    print("\nPC2 (纵轴) 极端 token:")
    print("  PC2 最小（底部）:", [experiment.tokenizer.id_to_char.get(token_ids[i].item(), '?') for i in pc2_sorted[:15]])
    print("  PC2 最大（顶部）:", [experiment.tokenizer.id_to_char.get(token_ids[i].item(), '?') for i in pc2_sorted[-15:]])

    print("\n各类别在 PC1/PC2 上的质心:")
    for cat_id in sorted(cat_labels.keys()):
        idx = categories == cat_id
        if idx.sum() == 0:
            continue
        c1 = final_2d[idx, 0].mean()
        c2 = final_2d[idx, 1].mean()
        n = idx.sum()
        print(f"  {cat_labels[cat_id]} (n={n}): PC1={c1:.2f}, PC2={c2:.2f}")
