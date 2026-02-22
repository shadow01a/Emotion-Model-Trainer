import torch
import numpy as np
import os
import json
from typing import Dict, Any, cast
from sklearn.metrics import classification_report
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# 动态导入peft，避免导入错误
try:
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training
    )
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    print("警告: 未安装peft库，请运行 'pip install peft'")

from .config import Config
from .dataset import EmotionDataset

def train_and_evaluate(train_texts, test_texts, train_labels, test_labels, label_encoder):
    """训练和评估情绪分类模型 - 使用DoRA微调"""
    
    if not HAS_PEFT:
        raise ImportError("请先安装peft库: pip install peft")
        
    # 1. 初始化模型和分词器
    tokenizer = BertTokenizer.from_pretrained(Config.MODEL_NAME, use_fast=True)
    model = BertForSequenceClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=Config.NUM_LABELS,
        id2label={i: label for i, label in enumerate(label_encoder.classes_)},
        label2id={label: i for i, label in enumerate(label_encoder.classes_)}
    )

    # 2. 准备模型进行DoRA微调
    print("\n准备模型进行DoRA微调...")
    model = prepare_model_for_kbit_training(model)

    # 3. 配置DoRA参数 (基于LoRA的改进版本)
    dora_config = LoraConfig(
        r=Config.DORA_R,                    # DoRA秩
        lora_alpha=Config.DORA_ALPHA,       # DoRA缩放因子
        target_modules=Config.DORA_TARGET_MODULES,  # 目标模块
        lora_dropout=Config.DORA_DROPOUT,   # Dropout概率
        bias="none",                        # 不训练bias
        task_type="SEQ_CLS",                # 序列分类任务
        use_dora=Config.DORA_USE_MAGNITUDE  # 启用DoRA的幅度分解特性
    )

    # 4. 应用DoRA到模型
    model = get_peft_model(model, dora_config)
    
    # 打印模型信息
    model.print_trainable_parameters()
    
    # 5. 数据编码
    print("\nTokenizing 数据...")
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding="max_length",
        max_length=Config.MAX_LENGTH,
    )
    test_encodings = tokenizer(
        test_texts,
        truncation=True,
        padding="max_length",
        max_length=Config.MAX_LENGTH,
    )

    # 6. 创建数据集
    train_dataset = EmotionDataset(train_encodings, train_labels)
    test_dataset = EmotionDataset(test_encodings, test_labels)

    # 7. 训练配置 - CPU训练模式
    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR_BASE,
        num_train_epochs=Config.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=Config.get_cpu_train_batch_size(),  # 使用CPU优化的batch size
        per_device_eval_batch_size=Config.get_cpu_eval_batch_size(),    # 使用CPU优化的batch size
        learning_rate=Config.LEARNING_RATE,  # DoRA可以使用标准学习率
        weight_decay=Config.WEIGHT_DECAY,
        warmup_ratio=Config.WARMUP_RATIO,
        eval_strategy="epoch",  # 修正参数名
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        logging_dir=Config.LOGGING_DIR,
        logging_steps=50,
        seed=Config.SEED,
        fp16=True,  # 强制禁用混合精度训练
        report_to="none",
        use_cpu=True,  # 强制使用CPU
        dataloader_pin_memory=False,  # CPU训练不需要pin memory
        remove_unused_columns=False,  # 保留所有列以避免数据处理问题
    )

    # 8. 自定义评估指标
    def compute_metrics(pred) -> Dict[str, float]:
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        report = classification_report(
            labels, preds,
            target_names=label_encoder.classes_,
            output_dict=True,
            zero_division=0
        )
        # 使用cast来避免类型检查错误
        accuracy = cast(float, report["accuracy"])
        weighted_avg = cast(dict, report["weighted avg"])
        f1_score = cast(float, weighted_avg["f1-score"])
        precision = cast(float, weighted_avg["precision"])
        recall = cast(float, weighted_avg["recall"])
        
        return {
            "accuracy": accuracy,
            "f1_weighted": f1_score,
            "precision_weighted": precision,
            "recall_weighted": recall,
        }

    # 9. 训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    print(f"\n开始DoRA训练...")
    print(f"训练设备: {Config.DEVICE}")
    print(f"DoRA参数 - r: {Config.DORA_R}, alpha: {Config.DORA_ALPHA}, dropout: {Config.DORA_DROPOUT}")
    print(f"使用幅度分解: {Config.DORA_USE_MAGNITUDE}")
    print(f"训练batch size: {Config.get_cpu_train_batch_size()}")
    print(f"评估batch size: {Config.get_cpu_eval_batch_size()}")
    trainer.train()

    # 10. 最终评估
    print("\n=== 测试集最终性能 (使用最佳模型) ===")
    eval_results = trainer.evaluate(test_dataset)
    print(f"评估结果: {eval_results}")

    # 获取详细的分类报告
    print("\n详细分类报告:")
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    print(classification_report(
        test_labels,
        y_pred,
        target_names=label_encoder.classes_,
        digits=4
    ))

    # 11. 保存模型和配置
    os.makedirs(Config.FINAL_MODEL_DIR, exist_ok=True)
    print(f"\n保存最佳模型到 {Config.FINAL_MODEL_DIR}...")
    trainer.save_model(Config.FINAL_MODEL_DIR)
    tokenizer.save_pretrained(Config.FINAL_MODEL_DIR)

    # 保存标签映射
    label_mapping_path = os.path.join(Config.FINAL_MODEL_DIR, "label_mapping.json")
    print(f"保存标签映射到 {label_mapping_path}...")
    with open(label_mapping_path, "w", encoding="utf-8") as f:
        json.dump({
            "id2label": {str(i): label for i, label in enumerate(label_encoder.classes_)},
            "label2id": {label: i for i, label in enumerate(label_encoder.classes_)}
        }, f, ensure_ascii=False, indent=2)

    # 保存DoRA配置
    dora_config_path = os.path.join(Config.FINAL_MODEL_DIR, "dora_config.json")
    print(f"保存DoRA配置到 {dora_config_path}...")
    with open(dora_config_path, "w", encoding="utf-8") as f:
        json.dump({
            "r": Config.DORA_R,
            "lora_alpha": Config.DORA_ALPHA,
            "lora_dropout": Config.DORA_DROPOUT,
            "target_modules": Config.DORA_TARGET_MODULES,
            "use_dora": Config.DORA_USE_MAGNITUDE
        }, f, ensure_ascii=False, indent=2)

    print(f"\n模型和配置已保存到 {Config.FINAL_MODEL_DIR}")

    from .onnx_exporter import convert_to_onnx
    convert_to_onnx(Config.FINAL_MODEL_DIR, Config.FINAL_MODEL_DIR + "model.onnx", max_length=Config.MAX_LENGTH)