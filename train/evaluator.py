import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import classification_report
from transformers import Trainer


class Evaluator:
    """模型评估器 - 负责测试集性能评估和分类报告生成"""

    def __init__(self, trainer: Trainer, label_encoder):
        self.trainer = trainer
        self.label_encoder = label_encoder

    def evaluate(self, test_dataset, test_labels) -> Dict[str, Any]:
        """评估测试集性能并打印详细分类报告"""
        print("\n=== 测试集最终性能 (使用最佳模型) ===")

        # 获取评估结果
        eval_results = self.trainer.evaluate(test_dataset)
        print(f"评估结果：{eval_results}")

        # 获取详细的分类报告
        print("\n详细分类报告:")
        predictions = self.trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)

        report = classification_report(
            test_labels,
            y_pred,
            target_names=self.label_encoder.classes_,
            digits=4
        )
        print(report)

        return eval_results

    def predict(self, dataset) -> np.ndarray:
        """进行预测并返回预测结果"""
        predictions = self.trainer.predict(dataset)
        return np.argmax(predictions.predictions, axis=1)

    def get_classification_report(self, test_labels, y_pred, digits: int = 4) -> str:
        """生成分类报告字符串"""
        return classification_report(
            test_labels,
            y_pred,
            target_names=self.label_encoder.classes_,
            digits=digits
        )
