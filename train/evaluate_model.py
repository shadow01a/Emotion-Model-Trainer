import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

from .config import Config


def load_eval_data(eval_data_path: Optional[str] = None) -> Tuple[List[str], List[int], LabelEncoder]:
    """
    加载评测数据集

    Args:
        eval_data_path: 评测数据集路径，如果为 None 则使用 Config.EVAL_DATA_PATH

    Returns:
        tuple: (eval_texts, eval_labels_encoded, label_encoder)
    """
    if eval_data_path is None:
        eval_data_path = Config.EVAL_DATA_PATH

    # 加载评测数据
    try:
        eval_data = pd.read_csv(eval_data_path)
    except FileNotFoundError:
        print(f"错误：评测数据文件 '{eval_data_path}' 未找到。")
        raise

    # 筛选有效标签，并转换为字符串以防万一
    eval_data["label"] = eval_data["label"].astype(str)

    # 只保留目标情绪
    eval_data = eval_data[eval_data["label"].isin(Config.TARGET_EMOTIONS)].copy()
    eval_data["text"] = eval_data["text"].astype(str)

    if eval_data.empty:
        print(f"错误：在评测数据中没有找到属于 TARGET_EMOTIONS 的数据。")
        raise ValueError("评测数据为空")

    # 数据统计
    print("\n=== 评测数据统计 ===")
    print(f"目标情绪类别数量: {Config.NUM_LABELS}")
    print(f"评测集样本数: {len(eval_data)}")
    print("评测集类别分布:\n", eval_data["label"].value_counts().sort_index())

    # 使用固定顺序的标签编码器
    label_encoder = LabelEncoder()
    label_encoder.fit(Config.TARGET_EMOTIONS)  # 强制按定义顺序编码

    # 准备数据
    eval_texts = eval_data["text"].tolist()
    eval_labels = eval_data["label"].tolist()

    # 编码标签
    eval_labels_encoded = label_encoder.transform(eval_labels).tolist()

    print(f"最终评测数据: {len(eval_texts)} 个样本")

    # 创建标签映射字典
    label_mapping = {label: idx for idx, label in enumerate(label_encoder.classes_)}
    print("标签映射:", label_mapping)

    return eval_texts, eval_labels_encoded, label_encoder


def evaluate(
    model_dir: str,
    eval_data_path: Optional[str] = None,
    use_onnx: bool = True,
    max_length: int = 128,
    batch_size: int = 32
) -> Dict[str, Any]:
    """
    评测模型性能

    Args:
        model_dir: 模型目录
        eval_data_path: 评测数据集路径，如果为 None 则使用 Config.EVAL_DATA_PATH
        use_onnx: 是否使用 ONNX 模型（True: ONNX, False: 原始模型）
        max_length: 最大文本长度
        batch_size: 批量预测大小

    Returns:
        评测结果字典（包含准确率、F1 等指标）
    """
    print(f"\n=== 开始模型评测 ===")
    print(f"模型目录: {model_dir}")
    print(f"评测数据集: {eval_data_path or Config.EVAL_DATA_PATH}")
    print(f"评测方式: {'ONNX' if use_onnx else '原始模型'}")
    print(f"最大长度: {max_length}")

    # 1. 加载评测数据
    eval_texts, eval_labels, label_encoder = load_eval_data(eval_data_path)

    # 2. 加载预测器
    if use_onnx:
        print("加载 ONNX 预测器...")
        from inference.emotion_predictor import EmotionPredictor
        predictor = EmotionPredictor(model_dir, max_length)
    else:
        print("加载原始模型预测器...")
        from inference.safetensors_predictor import SafetensorsPredictor
        predictor = SafetensorsPredictor(model_dir, max_length)

    # 3. 批量预测
    print(f"开始批量预测，样本数: {len(eval_texts)}，批次大小: {batch_size}")
    all_predictions = []

    for i in range(0, len(eval_texts), batch_size):
        batch_texts = eval_texts[i:i + batch_size]
        batch_predictions = predictor.predict_batch(batch_texts, return_top_k=1)
        all_predictions.extend(batch_predictions)

        if i % (batch_size * 10) == 0:
            print(f"  已处理 {min(i + batch_size, len(eval_texts))}/{len(eval_texts)} 个样本")

    # 4. 将预测标签转换为编码，并统一真实标签编码
    # 预测器返回的是标签字符串，需要转换为数字编码
    # 优先使用预测器的 label2id 映射

    if not hasattr(predictor, 'label2id'):
        # 尝试获取 label2id
        if hasattr(predictor, 'id2label'):
            # 从 id2label 创建 label2id
            id2label = predictor.id2label
            label2id = {}
            for label_id_str, label in id2label.items():
                label2id[label] = int(label_id_str)
            predictor.label2id = label2id  # 临时添加属性
        else:
            print("错误: 预测器没有标签映射信息，无法进行评测")
            raise ValueError("预测器缺少标签映射信息")

    label2id = predictor.label2id
    print(f"使用预测器的标签映射进行转换，标签数量: {len(label2id)}")

    # 转换预测标签为数字ID
    y_pred = []
    invalid_indices = []

    # 统计预测类别分布
    pred_distribution = {}

    # 首先确保label2id中的键都是字符串（可能已经是了）
    # 但为了安全，我们将所有预测标签转换为字符串进行比较
    for idx, pred in enumerate(all_predictions):
        # 将所有预测标签转换为字符串
        pred_str = str(pred).strip()

        # 调试：输出前几个预测结果
        if idx < 5:
            print(f"预测[{idx}]: 原始='{pred}' (类型: {type(pred)}), 转换后='{pred_str}'")

        if pred_str in label2id:
            pred_id = label2id[pred_str]
            y_pred.append(pred_id)
            # 统计分布
            pred_distribution[pred_str] = pred_distribution.get(pred_str, 0) + 1
        else:
            # 如果找不到映射，记录为无效
            y_pred.append(-1)
            invalid_indices.append(idx)
            if len(invalid_indices) <= 3:  # 只输出前几个无效标签
                print(f"无效预测[{idx}]: '{pred_str}' 不在label2id中")
                print(f"  可用的label2id键: {list(label2id.keys())[:5]}...")

    invalid_count = len(invalid_indices)
    if invalid_count > 0:
        print(f"警告: {invalid_count} 个预测标签无法映射到ID")
        if invalid_count > 0:
            print(f"示例无效预测: '{all_predictions[invalid_indices[0]]}' -> '{str(all_predictions[invalid_indices[0]]).strip()}'")
            print(f"label2id检查: '兴奋' in label2id: {'兴奋' in label2id}")
            print(f"label2id检查: '{str(all_predictions[invalid_indices[0]]).strip()}' in label2id: {str(all_predictions[invalid_indices[0]]).strip() in label2id}")

    # 输出预测分布
    print(f"\n预测类别分布 (总数: {len(all_predictions)}):")
    for label, count in sorted(pred_distribution.items(), key=lambda x: x[1], reverse=True):
        percentage = count / len(all_predictions) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")

    # 将真实标签（label_encoder编码）转换为预测器的编码
    # 首先将数字ID转换为标签字符串
    true_label_strings = label_encoder.inverse_transform(eval_labels)

    # 然后使用预测器的label2id转换为数字ID
    y_true = []
    invalid_true_indices = []
    for idx, label_obj in enumerate(true_label_strings):
        # 将标签转换为字符串（处理numpy字符串对象）
        label_str = str(label_obj).strip()

        # 调试：输出前几个真实标签
        if idx < 5:
            print(f"真实标签[{idx}]: 原始={repr(label_obj)} (类型: {type(label_obj)}), 转换后='{label_str}'")

        if label_str in label2id:
            y_true.append(label2id[label_str])
        else:
            # 真实标签应该在映射中，如果不在则报错
            print(f"错误: 真实标签 '{label_str}' 不在预测器的标签映射中")
            y_true.append(-1)
            invalid_true_indices.append(idx)
            if len(invalid_true_indices) <= 3:
                print(f"  转换后的标签: '{label_str}'")
                print(f"  label2id检查: '兴奋' in label2id: {'兴奋' in label2id}")
                print(f"  label2id检查: '{label_str}' in label2id: {label_str in label2id}")

    if invalid_true_indices:
        print(f"警告: {len(invalid_true_indices)} 个真实标签无法映射到预测器的标签映射")

    # 过滤无效的预测（y_pred == -1 或 y_true == -1）
    valid_indices = [i for i in range(len(y_true)) if y_true[i] != -1 and y_pred[i] != -1]
    if len(valid_indices) < len(y_true):
        print(f"过滤无效样本: {len(y_true) - len(valid_indices)} 个样本被移除")
        y_true = [y_true[i] for i in valid_indices]
        y_pred = [y_pred[i] for i in valid_indices]

    if len(y_true) == 0:
        print("错误: 没有有效的样本进行评测")
        raise ValueError("所有样本都被过滤，无法进行评测")

    print(f"有效评测样本数: {len(y_true)}")

    # 5. 计算评估指标
    print("\n=== 评测结果 ===")

    # 准备分类报告的标签名称（按预测器的标签顺序）
    # 从 label2id 创建 id2label 映射
    id2label = {id: label for label, id in label2id.items()}
    # 按数字ID排序
    sorted_ids = sorted(id2label.keys())
    target_names = [id2label[id] for id in sorted_ids]

    # 准确率
    accuracy = accuracy_score(y_true, y_pred)
    print(f"准确率 (Accuracy): {accuracy:.4f}")

    # F1 分数
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"宏平均 F1 (Macro F1): {macro_f1:.4f}")
    print(f"加权平均 F1 (Weighted F1): {weighted_f1:.4f}")

    # 分类报告
    print("\n详细分类报告:")
    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=4
    )
    print(report)

    # 6. 收集结果
    results = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'original_num_samples': len(eval_texts),
        'valid_num_samples': len(y_true),
        'num_classes': Config.NUM_LABELS,
        'model_type': 'onnx' if use_onnx else 'original',
        'classification_report': report
    }

    # 7. 保存结果到文件
    output_dir = os.path.join(model_dir, 'evaluation_results')
    os.makedirs(output_dir, exist_ok=True)

    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    result_file = os.path.join(output_dir, f'evaluation_{timestamp}.json')

    # 转换为可序列化的格式
    serializable_results = {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'num_samples': len(y_true),  # 有效样本数
        'original_num_samples': len(eval_texts),
        'valid_num_samples': len(y_true),
        'num_classes': Config.NUM_LABELS,
        'model_type': 'onnx' if use_onnx else 'original',
        'model_dir': model_dir,
        'eval_data_path': eval_data_path or Config.EVAL_DATA_PATH,
        'timestamp': timestamp,
        'classification_report': report
    }

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)

    print(f"\n评测结果已保存到: {result_file}")

    return results


def evaluate_with_custom_predictor(
    predictor,
    eval_texts: List[str],
    eval_labels: List[int],
    label_encoder: LabelEncoder
) -> Dict[str, Any]:
    """
    使用自定义预测器进行评测（高级用法）

    Args:
        predictor: 预测器对象，必须实现 predict_batch 方法
        eval_texts: 评测文本列表
        eval_labels: 评测标签列表（已编码）
        label_encoder: 标签编码器

    Returns:
        评测结果字典
    """
    print(f"开始评测，样本数: {len(eval_texts)}")

    # 批量预测
    batch_size = 32
    all_predictions = []

    for i in range(0, len(eval_texts), batch_size):
        batch_texts = eval_texts[i:i + batch_size]
        batch_predictions = predictor.predict_batch(batch_texts, return_top_k=1)
        all_predictions.extend(batch_predictions)

    # 转换预测标签为数字ID
    y_pred = all_predictions
    y_true = eval_labels

    # 计算指标
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')

    # 分类报告
    target_names = label_encoder.classes_
    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=4
    )

    results = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'num_samples': len(eval_texts),
        'classification_report': report
    }

    return results


if __name__ == "__main__":
    # 命令行测试
    import argparse

    parser = argparse.ArgumentParser(description='评测模型性能')
    parser.add_argument('--model_dir', type=str, default=Config.FINAL_MODEL_DIR,
                       help='模型目录路径')
    parser.add_argument('--eval_data_path', type=str, default=None,
                       help='评测数据集路径')
    parser.add_argument('--use_onnx', action='store_true', default=True,
                       help='使用ONNX模型（默认）')
    parser.add_argument('--use_original', action='store_true', default=False,
                       help='使用原始模型')
    parser.add_argument('--max_length', type=int, default=Config.MAX_LENGTH,
                       help='最大文本长度')

    args = parser.parse_args()

    use_onnx = args.use_onnx and not args.use_original

    results = evaluate(
        model_dir=args.model_dir,
        eval_data_path=args.eval_data_path,
        use_onnx=use_onnx,
        max_length=args.max_length
    )