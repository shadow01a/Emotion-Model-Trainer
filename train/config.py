import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Config:
    """训练配置类，从环境变量加载配置"""

    # 数据配置
    DATA_DIR = os.getenv("DATA_DIR", "./data")
    DATA_PATH = os.getenv("DATA_PATH", "./data/emotion_data_manual.csv")
    TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH", "./data/train.csv")
    EVAL_DATA_PATH = os.getenv("EVAL_DATA_PATH", "./data/eval.csv")

    # 模型配置
    MODEL_NAME = os.getenv("MODEL_NAME", "bert-base-chinese")
    MAX_LENGTH = int(os.getenv("MAX_LENGTH", 128))
    NUM_TRAIN_EPOCHS = int(os.getenv("NUM_TRAIN_EPOCHS", 8))
    PER_DEVICE_TRAIN_BATCH_SIZE = int(os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE", 16))
    PER_DEVICE_EVAL_BATCH_SIZE = int(os.getenv("PER_DEVICE_EVAL_BATCH_SIZE", 32))
    LEARNING_RATE = float(os.getenv("LEARNING_RATE", 2e-5))
    WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", 0.01))
    WARMUP_RATIO = float(os.getenv("WARMUP_RATIO", 0.1))
    SEED = int(os.getenv("SEED", 42))

    # DoRA配置 (基于LoRA的改进版本)
    DORA_R = int(os.getenv("DORA_R", 32))
    DORA_ALPHA = int(os.getenv("DORA_ALPHA", 64))
    DORA_DROPOUT = float(os.getenv("DORA_DROPOUT", 0.9))
    DORA_TARGET_MODULES = os.getenv("DORA_TARGET_MODULES", "query,key,value,dense").split(",")
    DORA_USE_MAGNITUDE = os.getenv("DORA_USE_MAGNITUDE", "true").lower() == "true"  # 是否使用幅度分解

    # RSLora配置 (Rank-Stabilized LoRA)
    USE_RSLORA = os.getenv("USE_RSLORA", "true").lower() == "true"  # 是否使用RSLora
    RSLORA_R = int(os.getenv("RSLORA_R", 16))  # RSLora秩参数
    RSLORA_ALPHA = int(os.getenv("RSLORA_ALPHA", 32))  # RSLora缩放因子
    RSLORA_DROPOUT = float(os.getenv("RSLORA_DROPOUT", 0.1))  # RSLora Dropout概率
    RSLORA_TARGET_MODULES = os.getenv("RSLORA_TARGET_MODULES", "query,key,value,dense").split(",")

    # 设备配置
    DEVICE = os.getenv("DEVICE", "cpu")

    # 输出路径
    OUTPUT_DIR_BASE = os.getenv("OUTPUT_DIR_BASE", "./results_19emo")
    FINAL_MODEL_DIR = os.getenv("FINAL_MODEL_DIR", "./emotion_model_19emo")
    LOGGING_DIR = os.getenv("LOGGING_DIR", "./logs_19emo")

    # 19类情绪标签
    TARGET_EMOTIONS = os.getenv("TARGET_EMOTIONS", 
                               "高兴,平静,厌恶,害羞,害怕,生气,认真,紧张,慌张,疑惑,兴奋,无奈,担心,惊讶,哭泣,心动,难为情,自信,调皮").split(",")
    NUM_LABELS = len(TARGET_EMOTIONS)

    @classmethod
    def get_cpu_train_batch_size(cls):
        """获取CPU训练的batch size"""
        return cls.PER_DEVICE_TRAIN_BATCH_SIZE // 2

    @classmethod
    def get_cpu_eval_batch_size(cls):
        """获取CPU评估的batch size"""
        return cls.PER_DEVICE_EVAL_BATCH_SIZE // 2