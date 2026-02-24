import os
import json
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification


class SafetensorsPredictor:
    """使用.safetensors文件进行情绪预测的推理器"""

    def __init__(self, model_dir, max_length=128):
        """
        初始化情绪预测器

        Args:
            model_dir (str): 模型目录路径，包含model.safetensors（或pytorch_model.bin）和label_mapping.json
            max_length (int): 输入文本的最大长度
        """
        self.model_dir = model_dir
        self.max_length = max_length

        # 加载标签映射
        label_mapping_path = os.path.join(model_dir, "label_mapping.json")
        if not os.path.exists(label_mapping_path):
            raise FileNotFoundError(f"标签映射文件不存在: {label_mapping_path}")

        with open(label_mapping_path, "r", encoding="utf-8") as f:
            label_mapping = json.load(f)
            self.id2label = label_mapping["id2label"]
            self.label2id = label_mapping["label2id"]
        
        num_labels = len(self.id2label)

        # 加载分词器
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)

        # 尝试加载模型 - 优先尝试直接加载完整模型
        self.model = self._load_model(model_dir, num_labels)
        
        # 设置为评估模式
        self.model.eval()
        
        # 移动到CPU（因为项目支持CPU训练）
        self.model.to('cpu')

        print(f"[SafetensorsPredictor] 模型加载成功: {model_dir}")
        print(f"[SafetensorsPredictor] 支持的情绪类别: {len(self.id2label)}")

    def _load_model(self, model_dir, num_labels):
        """加载模型，优先尝试完整模型，然后尝试适配器"""
        # 首先检查是否有safetensors文件
        safetensors_path = os.path.join(model_dir, "model.safetensors")
        pytorch_path = os.path.join(model_dir, "pytorch_model.bin")
        
        # 检查是否是完整模型（包含classifier权重）
        if os.path.exists(safetensors_path):
            print(f"[SafetensorsPredictor] 找到safetensors文件: {safetensors_path}")
            try:
                from safetensors.torch import load_file
                state_dict = load_file(safetensors_path)
                
                # 检查是否包含分类器权重
                if 'classifier.weight' in state_dict or 'classifier.bias' in state_dict:
                    print(f"[SafetensorsPredictor] 检测到完整模型权重，直接加载...")
                    model = BertForSequenceClassification.from_pretrained(
                        model_dir,
                        num_labels=num_labels,
                        id2label=self.id2label,
                        label2id=self.label2id,
                        state_dict=state_dict
                    )
                    return model
                else:
                    print(f"[SafetensorsPredictor] safetensors文件不包含完整分类器权重，尝试其他方式...")
            except Exception as e:
                print(f"[SafetensorsPredictor] 加载safetensors文件失败: {e}")
        
        # 检查PyTorch模型文件
        if os.path.exists(pytorch_path):
            print(f"[SafetensorsPredictor] 找到PyTorch模型文件: {pytorch_path}")
            try:
                # 直接加载PyTorch模型
                model = BertForSequenceClassification.from_pretrained(
                    model_dir,
                    num_labels=num_labels,
                    id2label=self.id2label,
                    label2id=self.label2id
                )
                return model
            except Exception as e:
                print(f"[SafetensorsPredictor] 加载PyTorch模型失败: {e}")
        
        # 如果以上都失败，尝试从基础模型加载并应用适配器
        print(f"[SafetensorsPredictor] 尝试从基础模型加载并应用适配器...")
        try:
            # 先加载基础模型
            base_model = BertForSequenceClassification.from_pretrained(
                model_dir,
                num_labels=num_labels,
                id2label=self.id2label,
                label2id=self.label2id
            )
            
            # 检查是否有适配器配置
            adapter_config_path = os.path.join(model_dir, "adapter_config.json")
            if os.path.exists(adapter_config_path):
                print(f"[SafetensorsPredictor] 检测到PEFT适配器配置，尝试加载适配器...")
                from peft import PeftModel
                # 使用PeftModel加载，但避免重复加载
                model = PeftModel.from_pretrained(base_model, model_dir, is_trainable=False)
                # 合并适配器权重以获得完整模型
                model = model.merge_and_unload()
                return model
            else:
                return base_model
                
        except Exception as e:
            print(f"[SafetensorsPredictor] 从基础模型加载失败: {e}")
            # 最后尝试直接从目录加载
            print(f"[SafetensorsPredictor] 尝试直接从目录加载模型...")
            model = BertForSequenceClassification.from_pretrained(
                model_dir,
                num_labels=num_labels,
                id2label=self.id2label,
                label2id=self.label2id
            )
            return model

    def predict(self, text, return_top_k=1):
        """
        预测文本的情绪

        Args:
            text (str): 输入文本
            return_top_k (int): 返回前k个最可能的情绪类别

        Returns:
            如果return_top_k=1: 返回预测的情绪标签
            如果return_top_k>1: 返回前k个情绪标签及其概率的列表，格式为[(label, probability), ...]
        """
        # 文本预处理
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # 获取模型输出
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0]  # shape: (num_labels,)

        # 计算概率
        probabilities = self._softmax(logits.numpy())

        # 获取top-k结果
        top_k_indices = np.argsort(probabilities)[::-1][:return_top_k]

        if return_top_k == 1:
            return self.id2label[str(top_k_indices[0])]
        else:
            return [
                (self.id2label[str(idx)], float(probabilities[idx]))
                for idx in top_k_indices
            ]

    def predict_batch(self, texts, return_top_k=1):
        """
        批量预测文本的情绪

        Args:
            texts (list[str]): 输入文本列表
            return_top_k (int): 返回前k个最可能的情绪类别

        Returns:
            如果return_top_k=1: 返回预测的情绪标签列表
            如果return_top_k>1: 返回前k个情绪标签及其概率的列表的列表
        """
        # 文本预处理
        inputs = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # 获取模型输出
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # shape: (batch_size, num_labels)

        # 计算概率
        probabilities = self._softmax(logits.numpy(), axis=1)

        # 获取top-k结果
        top_k_indices = np.argsort(probabilities, axis=1)[:, ::-1][:, :return_top_k]

        if return_top_k == 1:
            return [self.id2label[str(idx[0])] for idx in top_k_indices]
        else:
            return [
                [
                    (self.id2label[str(idx)], float(probabilities[i][idx]))
                    for idx in top_k_indices[i]
                ]
                for i in range(len(texts))
            ]

    @staticmethod
    def _softmax(x, axis=None):
        """计算softmax概率"""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)