
import numpy as np
import torch
from train import load_data, train_and_evaluate

# 固定随机种子保证可复现
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def main():
    """主函数：加载数据、训练和评估模型"""
    print("=== 情绪分类系统 ===")

    # 选择运行模式
    mode = input("请选择模式 (1: 训练, 2: 导出ONNX, 3: 推理): ").strip()

    if mode == "1":
        print("=== 19类情绪分类模型训练 ===")
        # 1. 加载数据（使用预分割的train.csv和eval.csv）
        train_texts, test_texts, train_labels, test_labels, label_encoder = load_data()

        # 2. 训练和评估模型
        train_and_evaluate(train_texts, test_texts, train_labels, test_labels, label_encoder)

    elif mode == "2":
        print("=== 导出ONNX模型 ===")
        from train.onnx_exporter import convert_to_onnx
        from train.config import Config
        convert_to_onnx(Config.FINAL_MODEL_DIR, Config.FINAL_MODEL_DIR + "/model.onnx", max_length=128)

    elif mode == "3":
        print("=== 情绪推理 ===")
        from inference.emotion_predictor import EmotionPredictor
        predictor = EmotionPredictor()
        while True:
            text = input("请输入要分析的文本（输入'quit'退出）: ").strip()
            if text.lower() == 'quit':
                break
            if text:
                result = predictor.predict(text)
                print(f"预测情绪: {result['emotion']}, 置信度: {result['confidence']:.4f}")
    else:
        print("无效选择")

if __name__ == "__main__":
    main()
