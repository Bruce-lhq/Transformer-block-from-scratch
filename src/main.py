from attention_sink_module import AttentionSinkExperiment
import warnings
import os
import random
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

    # ====== 调用分析模块 ======
    from analysis import run_analysis
    run_analysis(scaling)

    # ====== 调用动画模块 ======
    from animation import run_animation
    run_animation(experiment, val_texts)
