import pandas as pd
import os
from train.config import Config

def split_data():
    """分割数据为训练集和评测集，每个情绪类别抽取10条作为评测集"""
    
    # 读取原始数据
    data_path = Config.DATA_PATH
    print(f"读取数据文件: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"错误：数据文件 '{data_path}' 不存在")
        return
    
    # 加载数据
    df = pd.read_csv(data_path)
    print(f"原始数据总行数: {len(df)}")
    
    # 筛选目标情绪
    df["label"] = df["label"].astype(str)
    df_filtered = df[df["label"].isin(Config.TARGET_EMOTIONS)].copy()
    print(f"筛选后数据行数: {len(df_filtered)}")
    
    # 按情绪分组
    eval_dfs = []
    train_dfs = []
    
    for emotion in Config.TARGET_EMOTIONS:
        emotion_data = df_filtered[df_filtered["label"] == emotion].copy()
        print(f"情绪 '{emotion}': {len(emotion_data)} 条数据")
        
        if len(emotion_data) >= 10:
            # 随机抽取10条作为评测集
            eval_sample = emotion_data.sample(n=10, random_state=Config.SEED)
            train_sample = emotion_data.drop(eval_sample.index)
        else:
            # 如果少于10条，全部作为评测集，训练集为空
            eval_sample = emotion_data
            train_sample = pd.DataFrame(columns=emotion_data.columns)
            print(f"警告：情绪 '{emotion}' 只有 {len(emotion_data)} 条数据，少于10条")
        
        eval_dfs.append(eval_sample)
        if not train_sample.empty:
            train_dfs.append(train_sample)
    
    # 合并评测集和训练集
    eval_df = pd.concat(eval_dfs, ignore_index=True)
    train_df = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame(columns=df.columns)
    
    # 打乱顺序
    eval_df = eval_df.sample(frac=1, random_state=Config.SEED).reset_index(drop=True)
    train_df = train_df.sample(frac=1, random_state=Config.SEED).reset_index(drop=True)
    
    # 保存文件
    data_dir = Config.DATA_DIR
    train_path = os.path.join(data_dir, "train.csv")
    eval_path = os.path.join(data_dir, "eval.csv")
    
    train_df.to_csv(train_path, index=False, encoding='utf-8-sig')
    eval_df.to_csv(eval_path, index=False, encoding='utf-8-sig')
    
    print(f"\n分割完成！")
    print(f"训练集保存到: {train_path} ({len(train_df)} 条)")
    print(f"评测集保存到: {eval_path} ({len(eval_df)} 条)")
    
    # 显示评测集统计
    print("\n=== 评测集统计 ===")
    print(eval_df["label"].value_counts().sort_index())
    
    # 显示训练集统计
    print("\n=== 训练集统计 ===")
    if len(train_df) > 0:
        print(train_df["label"].value_counts().sort_index())
    else:
        print("训练集为空")

if __name__ == "__main__":
    split_data()