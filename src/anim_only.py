from animation import run_animation
from attention_sink_module import AttentionSinkExperiment
import os, random, warnings
os.environ["HF_DATASETS_OFFLINE"] = "1"
from datasets import load_dataset
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn.utils")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")

raw_datasets = load_dataset("wikitext", "wikitext-103-v1")
idx = random.sample(range(len(raw_datasets['train'])), 1000)
texts = [raw_datasets['train']['text'][i] for i in idx if len(raw_datasets['train']['text'][i].strip()) > 10]
corpus = "".join(set("".join(texts))) + "<pad>"

if __name__ == "__main__":
    ckpt = "attention_sink_checkpoint.pth" if os.path.exists("attention_sink_checkpoint.pth") else None
    experiment = AttentionSinkExperiment(num_blocks=6, corpus=corpus, load_from=ckpt, num_heads=8, d_model=512, sink_size=1, window_size=30)
    val_texts = [t for t in raw_datasets['validation']['text'] if len(t.strip()) > 10]
    val_texts = random.sample(val_texts, min(30, len(val_texts)))
    run_animation(experiment, val_texts)
