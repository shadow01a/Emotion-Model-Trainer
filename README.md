# 情绪分类模型训练项目

本项目用于训练一个能够识别19+种不同情绪的文本分类模型，使用DoRA(Weight-Decomposed Low-Rank Adaptation)技术进行高效微调。

## 项目特点

- 🚀 使用DoRA技术进行参数高效微调（LoRA的改进版本）
- 💻 支持CPU训练模式
- 🎯 19+种细粒度情绪识别
- 📦 支持ONNX模型导出
- ⚙️ 配置化管理训练参数

## 项目结构

```
Emotion-Model-Trainer/
├── data/                    # 数据目录
│   └── emotion_data_manual.csv
├── pre-process/             # 数据预处理
│   └── csvCleaner.py
├── train/                   # 训练模块
│   ├── __init__.py
│   ├── config.py           # 配置模块
│   ├── data_loader.py      # 数据加载模块
│   ├── dataset.py          # 数据集类
│   └── trainer.py          # 训练模块
├── inference/              # 推理模块
│   ├── __init__.py
│   ├── config.py           # 推理设置模块
│   └── emotion_predictor.py# 推理类模块
├── .env                    # 环境变量配置
├── main.py                 # 主程序入口
├── requirements.txt        # 依赖列表
└── README.md              # 项目说明
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置

项目使用.env文件进行配置，包含以下主要配置项：

### 基础配置
- `DATA_PATH`: 训练数据路径
- `MODEL_NAME`: 预训练模型名称
- `MAX_LENGTH`: 文本最大长度
- `NUM_TRAIN_EPOCHS`: 训练轮数

### DoRA配置
- `DORA_R=32`: DoRA秩参数
- `DORA_ALPHA=64`: DoRA缩放因子
- `DORA_DROPOUT=0.9`: Dropout概率
- `DORA_TARGET_MODULES`: 目标模块列表(query,key,value,dense)
- `DORA_USE_MAGNITUDE=true`: 是否使用幅度分解（DoRA核心特性）

### 训练配置
- `DEVICE=cpu`: 训练设备
- `LEARNING_RATE`: 学习率
- `PER_DEVICE_TRAIN_BATCH_SIZE`: 训练批次大小
- `PER_DEVICE_EVAL_BATCH_SIZE`: 评估批次大小

### 输出配置
- `OUTPUT_DIR_BASE`: 训练输出目录
- `FINAL_MODEL_DIR`: 最终模型保存目录

## 使用方法

运行训练程序：

```bash
python main.py
```

## 情绪类别

模型可以识别以下19种情绪：
可以在env和data中的csv新增识别情绪

1. 高兴
2. 厌恶
3. 害羞
4. 害怕
5. 生气
6. 认真
7. 紧张
8. 慌张
9. 疑惑
10. 兴奋
11. 无奈
12. 担心
13. 惊讶
14. 哭泣
15. 心动
16. 难为情
17. 自信
18. 调皮
19. 平静

## DoRA微调优势

相比传统的LoRA微调，DoRA具有以下优势：

- ✅ **参数效率**: 继承LoRA的低参数特性
- ✅ **性能提升**: 通过权重分解获得更好的表达能力
- ✅ **内存友好**: 显著减少GPU/CPU内存占用
- ✅ **训练快速**: 更快的训练速度
- ✅ **幅度控制**: 独立控制权重的幅度和方向
- ✅ **易于部署**: 支持合并到原模型中

## DoRA vs LoRA

| 特性 | LoRA | DoRA |
|------|------|------|
| 参数效率 | ✅ 高效 | ✅ 高效 |
| 权重表示 | 直接低秩适应 | 分解为幅度×方向 |
| 表达能力 | 基础 | 更强 |
| 实现复杂度 | 简单 | 中等 |
| 性能表现 | 良好 | 更优 |

## 模型输出

训练完成后会生成以下文件：
- `adapter_model.bin`: DoRA适配器权重
- `adapter_config.json`: DoRA配置文件
- `config.json`: 模型配置
- `pytorch_model.bin`: 合并后的完整模型权重
- `tokenizer_config.json`: 分词器配置
- `vocab.txt`: 词汇表
- `label_mapping.json`: 标签映射文件
- `dora_config.json`: DoRA参数配置
- `model.onnx`: ONNX格式模型

## 注意事项

1. 确保数据文件格式正确
2. 根据硬件条件调整batch size
3. DoRA参数可根据具体任务进行调优
4. 训练过程中会显示可训练参数统计信息
5. `DORA_USE_MAGNITUDE=true`启用DoRA的核心特性