import torch
import numpy as np
import os
import json
from typing import Dict, Any
from sklearn.metrics import classification_report
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from .config import Config
from .dataset import EmotionDataset

def train_and_evaluate(train_texts, test_texts, train_labels, test_labels, label_encoder):
    """训练和评估情绪分类模型 - 支持PiSSA、DoRA、RSLora三种方法的组合微调"""
        
    # 1. 初始化模型和分词器
    tokenizer = BertTokenizer.from_pretrained(Config.MODEL_NAME, use_fast=True)
    model = BertForSequenceClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=Config.NUM_LABELS,
        id2label={i: label for i, label in enumerate(label_encoder.classes_)},
        label2id={label: i for i, label in enumerate(label_encoder.classes_)}
    )

    # 2. 准备模型进行微调
    model = prepare_model_for_kbit_training(model)

    # 3. 配置组合微调参数
    use_pissa = Config.USE_PISSA
    
    # 构建配置名称用于显示
    config_parts = []
    if use_pissa:
        config_parts.append("PiSSA")
    if Config.DORA_USE_MAGNITUDE:
        config_parts.append("DoRA")
    if Config.USE_RSLORA:
        config_parts.append("RSLora")
    
    if not config_parts:
        config_parts.append("LoRA")
    
    config_name = "+".join(config_parts)
    
    # 创建LoraConfig参数字典
    lora_config_kwargs = {
        'r': Config.LORA_R,
        'lora_alpha': Config.LORA_ALPHA,
        'target_modules': Config.LORA_TARGET_MODULES,
        'lora_dropout': Config.LORA_DROPOUT,
        'bias': "none",
        'task_type': "SEQ_CLS",
        'use_dora': Config.DORA_USE_MAGNITUDE,
        'use_rslora': Config.USE_RSLORA
    }
    
    # 添加PiSSA相关配置（如果启用）
    if use_pissa:
        if Config.PISSA_INIT_METHOD == "pissa":
            lora_config_kwargs['init_lora_weights'] = "pissa"
        else:
            lora_config_kwargs['init_lora_weights'] = "pissa_niter_4"
    
    # 创建LoraConfig
    finetune_config = LoraConfig(**lora_config_kwargs)

    # 4. 应用组合微调到模型
    model = get_peft_model(model, finetune_config)
    
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

    # 计算warmup_steps替代warmup_ratio
    total_train_samples = len(train_dataset)
    total_train_steps = (total_train_samples // Config.PER_DEVICE_TRAIN_BATCH_SIZE) * Config.NUM_TRAIN_EPOCHS
    warmup_steps = int(Config.WARMUP_RATIO * total_train_steps)

    # 7. 训练配置
    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR_BASE,
        num_train_epochs=Config.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=Config.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=Config.PER_DEVICE_EVAL_BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        warmup_steps=warmup_steps,  # 替换warmup_ratio
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        logging_steps=50,
        seed=Config.SEED,
        fp16=torch.cuda.is_available(),
        report_to="none",
        use_cpu=Config.DEVICE.lower() == "cpu",
        dataloader_pin_memory=Config.DEVICE.lower() != "cpu",
        remove_unused_columns=False,
    )

    # 8. 自定义评估指标
    def compute_metrics(pred) -> Dict[str, float]:
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        report_dict = classification_report(
            labels, preds,
            target_names=label_encoder.classes_,
            output_dict=True,
            zero_division=0
        )
        # 直接访问字典键
        accuracy = float(report_dict["accuracy"])
        weighted_avg = report_dict["weighted avg"]
        f1_score = float(weighted_avg["f1-score"])
        precision = float(weighted_avg["precision"])
        recall = float(weighted_avg["recall"])
        
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

    print(f"\n开始{config_name}组合训练...")
    print(f"训练设备: {Config.DEVICE}")
    print(f"LoRA参数 - r: {Config.LORA_R}, alpha: {Config.LORA_ALPHA}, dropout: {Config.LORA_DROPOUT}")
    if use_pissa:
        print(f"PiSSA初始化方法: {Config.PISSA_INIT_METHOD}")
    print(f"DoRA幅度分解: {Config.DORA_USE_MAGNITUDE}")
    print(f"RSLora稳定化: {Config.USE_RSLORA}")
    print(f"训练batch size: {Config.PER_DEVICE_TRAIN_BATCH_SIZE}")
    print(f"评估batch size: {Config.PER_DEVICE_EVAL_BATCH_SIZE}")
    print(f"Warmup steps: {warmup_steps}")  # 添加warmup_steps信息
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
    
    # 对于PiSSA，需要特殊处理来转换为正常模型
    if Config.USE_PISSA:
        # 保存PiSSA适配器
        pissa_adapter_dir = os.path.join(Config.FINAL_MODEL_DIR, "pissa_adapter")
        os.makedirs(pissa_adapter_dir, exist_ok=True)
        model.save_pretrained(pissa_adapter_dir)
        
        # 转换PiSSA适配器为正常模型（合并权重）
        try:
            model.save_pretrained(
                Config.FINAL_MODEL_DIR,
                path_initial_model_for_weight_conversion=pissa_adapter_dir
            )
            print(f"PiSSA适配器已转换并合并到正常模型中")
        except Exception as e:
            # 如果转换失败，使用标准保存方式
            print(f"PiSSA权重转换失败: {e}，使用标准保存方式")
            trainer.save_model(Config.FINAL_MODEL_DIR)
    else:
        # 标准保存方式
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

    # 保存组合微调配置
    finetune_config_path = os.path.join(Config.FINAL_MODEL_DIR, "combined_finetune_config.json")
    config_data = {
        "lora_r": Config.LORA_R,
        "lora_alpha": Config.LORA_ALPHA,
        "lora_dropout": Config.LORA_DROPOUT,
        "lora_target_modules": Config.LORA_TARGET_MODULES,
        "use_pissa": use_pissa,
        "pissa_init_method": Config.PISSA_INIT_METHOD if use_pissa else None,
        "use_dora": Config.DORA_USE_MAGNITUDE,
        "use_rslora": Config.USE_RSLORA
    }
    
    print(f"保存组合微调配置到 {finetune_config_path}...")
    with open(finetune_config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)

    print(f"\n模型和配置已保存到 {Config.FINAL_MODEL_DIR}")

    from .onnx_exporter import convert_to_onnx
    convert_to_onnx(Config.FINAL_MODEL_DIR, Config.FINAL_MODEL_DIR + "/model.onnx", max_length=Config.MAX_LENGTH)