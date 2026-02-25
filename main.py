import numpy as np

# 固定随机种子保证可复现
SEED = 42
np.random.seed(SEED)

def main():
    """主函数：加载数据、训练和评估模型"""
    print("=== 情绪分类系统 ===")

    # 选择运行模式
    mode = input("请选择模式 (1: 训练, 2: 导出ONNX, 3: 推理, 4: 评测): ").strip()

    if mode == "1":
        # 延迟导入
        import torch
        from train import load_data, train_and_evaluate
        torch.manual_seed(SEED)
        
        print("=== 19类情绪分类模型训练 ===")
        # 1. 加载数据（使用预分割的train.csv和eval.csv）
        train_texts, test_texts, train_labels, test_labels, label_encoder = load_data()

        # 2. 训练和评估模型
        train_and_evaluate(train_texts, test_texts, train_labels, test_labels, label_encoder)

    elif mode == "2":
        print("=== 导出ONNX模型 ===")
        from train.onnx_exporter import convert_to_onnx
        from train.config import Config
        convert_to_onnx(Config.FINAL_MODEL_DIR, Config.FINAL_MODEL_DIR + "/model.onnx", max_length=Config.MAX_LENGTH)

    elif mode == "3":
        print("=== 情绪推理 ===")
        from train.config import Config
        
        # 让用户选择推理方式
        inference_mode = input("请选择推理方式 (1: ONNX推理, 2: 原始文件推理): ").strip()
        
        if inference_mode == "1":
            print("=== 使用ONNX模型进行推理 ===")
            from inference.emotion_predictor import EmotionPredictor
            predictor = EmotionPredictor(Config.FINAL_MODEL_DIR, Config.MAX_LENGTH)
        elif inference_mode == "2":
            print("=== 使用原始模型文件进行推理 ===")
            from inference.safetensors_predictor import SafetensorsPredictor
            predictor = SafetensorsPredictor(Config.FINAL_MODEL_DIR, Config.MAX_LENGTH)
        else:
            print("无效选择，使用默认ONNX推理")
            from inference.emotion_predictor import EmotionPredictor
            predictor = EmotionPredictor(Config.FINAL_MODEL_DIR, Config.MAX_LENGTH)
            
        while True:
            text = input("请输入要分析的文本（输入'quit'退出）: ").strip()
            if text.lower() == 'quit':
                break
            if text:
                result = predictor.predict(text, return_top_k=3)  # 返回前3个结果以显示置信度
                print(f"预测结果:")
                for i, (emotion, confidence) in enumerate(result):
                    print(f"  {i+1}. {emotion}: {confidence:.4f}")
    elif mode == "4":
        print("=== 模型评测 ===")
        from train.evaluate_model import evaluate
        from train.config import Config

        # 选择评测方式
        eval_mode = input("请选择评测方式 (1: ONNX 评测, 2: 原始模型评测): ").strip()
        use_onnx = (eval_mode == "1")

        if eval_mode not in ["1", "2"]:
            print("无效选择，使用默认 ONNX 评测")
            use_onnx = True

        try:
            # 执行评测
            results = evaluate(
                model_dir=Config.FINAL_MODEL_DIR,
                eval_data_path=Config.EVAL_DATA_PATH,
                use_onnx=use_onnx,
                max_length=Config.MAX_LENGTH
            )

            print(f"\n=== 评测完成 ===")
            print(f"准确率: {results['accuracy']:.4f}")
            print(f"宏平均 F1: {results['macro_f1']:.4f}")
            print(f"加权平均 F1: {results['weighted_f1']:.4f}")
            print(f"评测样本数: {results['valid_num_samples']} (原始: {results['original_num_samples']})")

        except Exception as e:
            print(f"评测过程中出现错误: {e}")
            import traceback
            traceback.print_exc()

    else:
        print("无效选择")

if __name__ == "__main__":
    main()