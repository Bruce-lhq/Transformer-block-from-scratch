from attention_sink_module import AttentionSinkExperiment
import warnings
import os
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn.utils") # 忽略 seaborn 的中文字体警告，避免干扰输出
if __name__ == "__main__":
    # 训练文本：包含重复模式的中文句子，用于凸显 attention sink 现象
    texts_for_train = [
        "我我我喜欢学习人工智能。",
        "这个这个这个是一个测试句子。",
        "开始开始开始今天天气很好。",
        "结束结束结束我们回家吧。",
        "啊啊啊这是一个惊讶的表达。",
        "重复重复重复是注意力汇的关键。",
        "你好，你好，你好，很高兴认识你！",
        "今天今天今天是个特殊的日子。",
        "学习学习学习让我进步。",
        "明天明天明天会更好。",
        "人工智能正在改变世界。",
        "注意力机制是Transformer模型的核心。",
        "深度学习已经颠覆了许多领域。",
        "自然语言处理让机器理解人类语言。",
        "计算机视觉使机器能够看见。",
        "机器学习算法从数据中学习模式。",
        "神经网络模仿人脑的工作方式。",
        "大数据时代带来了新的挑战和机遇。",
        "云计算提供了强大的计算能力。",
        "物联网连接了物理世界和数字世界。",
        "区块链技术保证了数据的安全性。",
        "量子计算将开启新的计算范式。",
        "自动驾驶汽车正在成为现实。",
        "虚拟现实创造了沉浸式的体验。",
        "增强现实将数字信息叠加到现实世界。",
        "机器人技术正在改变制造业。",
        "基因编辑技术有望治愈遗传疾病。",
        "可再生能源是未来的发展方向。",
        "气候变化是全球性的挑战。",
        "可持续发展是人类共同的目标。",
        "教育是推动社会进步的力量。",
        "文化多样性丰富了人类文明。",
        "艺术表达是人类情感的表现。",
        "体育运动促进身心健康。",
        "健康饮食有助于维持良好状态。",
        "心理健康与身体健康同样重要。",
        "友谊是人生宝贵的财富。",
        "家庭是温暖的港湾。",
        "爱情是人类永恒的主题。",
        "梦想驱动人们不断前进。",
        "努力工作是成功的基础。",
        "坚持不懈才能克服困难。",
        "创新思维推动社会进步。",
        "团队合作能够实现更大目标。",
        "沟通技巧在人际交往中至关重要。",
        "时间管理提高工作效率。",
        "批判性思维帮助做出明智决策。",
        "终身学习适应快速变化的世界。",
        "感恩之心让生活更加美好。",
        "乐观态度面对生活中的挑战。"
    ]
    # 语料库包含常用汉字、英文字母、数字和标点符号
    corpus = "的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从无情己面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已亲其进此话常与活正感1234567890!@#$%^&*()_+-=~`qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM[]{}|;:'\",.<>/? \n，。！？；：“”‘’、……——·《》【】"
    corpus += ''.join(set(''.join(texts_for_train)))  # 确保训练文本中的所有字符都包含在语料库中
    # 如果之前已经训练过模型并保存了检查点，就从检查点继续训练，否则从头开始训练
    if os.path.exists("attention_sink_checkpoint.pth"):
        load_from = "attention_sink_checkpoint.pth"
    else:
        load_from = None
    
    # 创建实验对象，训练模型，并可视化注意力图和生成文本
    experiment = AttentionSinkExperiment(num_blocks=6, corpus=corpus, load_from=load_from,learning_rate=1e-4, sink_size=4, window_size=10, log_dir="runs/attention_sink_experiment")
    experiment.train(texts_for_train, save_path="attention_sink_checkpoint.pth", epochs=100, log_interval=5, epoch_interval=1, batch_size=16)
    test_text = "你好，你好，你好，很高兴认识你！"
    experiment.visualize_attention(test_text, layer_idx=-1, head_idx='mean')
    print("Generated text:", experiment.generate(test_text, max_new_tokens=100))