
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from .config import Config

def load_data(data_path=None, train_data_path=None, eval_data_path=None):
    """加载并预处理数据，强制使用定义的情绪标签"""
    if train_data_path is None:
        train_data_path = Config.TRAIN_DATA_PATH
    if eval_data_path is None:
        eval_data_path = Config.EVAL_DATA_PATH

    # 加载训练数据
    try:
        train_data = pd.read_csv(train_data_path)
    except FileNotFoundError:
        print(f"错误：训练数据文件 '{train_data_path}' 未找到。")
        print("请先运行 data_splitter.py 分割数据")
        exit()

    # 加载评测数据
    try:
        eval_data = pd.read_csv(eval_data_path)
    except FileNotFoundError:
        print(f"错误：评测数据文件 '{eval_data_path}' 未找到。")
        print("请先运行 data_splitter.py 分割数据")
        exit()

    # 筛选有效标签，并转换为字符串以防万一
    train_data["label"] = train_data["label"].astype(str)
    eval_data["label"] = eval_data["label"].astype(str)
    
    # 只保留目标情绪
    train_data = train_data[train_data["label"].isin(Config.TARGET_EMOTIONS)].copy()
    eval_data = eval_data[eval_data["label"].isin(Config.TARGET_EMOTIONS)].copy()
    
    train_data["text"] = train_data["text"].astype(str)
    eval_data["text"] = eval_data["text"].astype(str)

    if train_data.empty and eval_data.empty:
        print(f"错误：在训练和评测数据中都没有找到属于 TARGET_EMOTIONS 的数据。")
        exit()
    elif train_data.empty:
        print(f"警告：训练数据为空，只有评测数据")
    elif eval_data.empty:
        print(f"警告：评测数据为空，只有训练数据")

    # 数据统计
    print("\n=== 数据统计 ===")
    print(f"目标情绪类别数量: {Config.NUM_LABELS}")
    print(f"训练集样本数: {len(train_data)}")
    print(f"评测集样本数: {len(eval_data)}")
    if len(train_data) > 0:
        print("训练集类别分布:\n", train_data["label"].value_counts().sort_index())
    if len(eval_data) > 0:
        print("评测集类别分布:\n", eval_data["label"].value_counts().sort_index())

    # 使用固定顺序的标签编码器
    label_encoder = LabelEncoder()
    label_encoder.fit(Config.TARGET_EMOTIONS)  # 强制按定义顺序编码

    # 准备数据
    train_texts = train_data["text"].tolist() if len(train_data) > 0 else []
    train_labels = train_data["label"].tolist() if len(train_data) > 0 else []
    eval_texts = eval_data["text"].tolist() if len(eval_data) > 0 else []
    eval_labels = eval_data["label"].tolist() if len(eval_data) > 0 else []

    # 编码标签
    train_labels_encoded = label_encoder.transform(train_labels).tolist() if train_labels else []
    eval_labels_encoded = label_encoder.transform(eval_labels).tolist() if eval_labels else []

    print(f"\n最终数据: 训练集={len(train_texts)}, 评测集={len(eval_texts)}")
    
    # 创建标签映射字典
    label_mapping = {label: idx for idx, label in enumerate(label_encoder.classes_)}
    print("标签映射:", label_mapping)

    return train_texts, eval_texts, train_labels_encoded, eval_labels_encoded, label_encoder
