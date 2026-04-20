import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Songti SC']


def run_analysis(scaling):
    """接收 scaling 数据，进行三组拟合分析并绘图保存"""
    L = np.array(scaling['num_blocks'], dtype=float)
    PPL = np.array(scaling['ppl'])
    ACC = np.array(scaling['cloze_acc'])
    SINK = np.array(scaling['sink_rate'])

    # ============================================================
    # A. PPL 幂律拟合: PPL(L) = A * L^(-alpha) + C
    # ============================================================
    def ppl_power_law(x, A, alpha, C):
        return A * np.power(x, -alpha) + C
    popt_ppl, _ = curve_fit(ppl_power_law, L, PPL, p0=[50, 1.5, 5], maxfev=10000)
    A_fit, alpha_fit, C_fit = popt_ppl
    print(f"\n=== A. PPL 幂律拟合: PPL(L) = A·L^(-α) + C ===")
    print(f"  A = {A_fit:.4f}, α = {alpha_fit:.4f}, C = {C_fit:.4f}")
    print(f"  不可还原熵 C = {C_fit:.2f} (模型深度→∞时的 PPL 下界)")

    # ============================================================
    # B. Accuracy vs ln(1/PPL): Acc = β·ln(1/PPL) + K
    # ============================================================
    inv_ppl = np.log(1.0 / PPL)
    coeffs_b = np.polyfit(inv_ppl, ACC, 1)
    beta_fit, K_fit = coeffs_b
    print(f"\n=== B. PPL-Accuracy 关系: Acc = β·ln(1/PPL) + K ===")
    print(f"  β = {beta_fit:.4f}, K = {K_fit:.4f}")
    print(f"  转换效率: PPL 每下降 1 单位 → 准确率提升约 {beta_fit / np.mean(PPL):.4f}")
    for i in range(1, len(PPL)):
        d_acc = ACC[i] - ACC[i-1]
        d_inv_ppl = inv_ppl[i] - inv_ppl[i-1]
        local_beta = d_acc / d_inv_ppl if d_inv_ppl != 0 else 0
        print(f"  L={L[i-1]:.0f}→{L[i]:.0f}: 局部β = {local_beta:.4f}" +
              (" ← 饱和" if local_beta < beta_fit * 0.5 else ""))

    # ============================================================
    # C. Sink Rate 指数衰减: SinkRate(L) = S0·exp(-λL) + k
    # ============================================================
    def sink_exp(x, S0, lam, k):
        return S0 * np.exp(-lam * x) + k
    popt_sink, _ = curve_fit(sink_exp, L, SINK, p0=[0.012, 0.3, 0.004], maxfev=10000)
    S0_fit, lam_fit, k_fit = popt_sink
    print(f"\n=== C. Sink Rate 衰减场: SinkRate(L) = S₀·exp(-λL) + k ===")
    print(f"  S₀ = {S0_fit:.6f}, λ = {lam_fit:.4f}, k = {k_fit:.6f}")
    print(f"  衰减系数 λ = {lam_fit:.4f} (特征深度 1/λ = {1/lam_fit:.2f} 层)")

    # ============================================================
    # 绘图: 2×2 Scaling Law 图
    # ============================================================
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
