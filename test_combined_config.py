#!/usr/bin/env python3
"""测试DoRA+RSLora组合微调配置是否正确加载"""

from train.config import Config

def test_combined_config():
    """测试组合微调相关配置"""
    print("=== DoRA+RSLora组合微调配置测试 ===")
    
    print("\n--- DoRA配置 ---")
    print(f"DORA_R: {Config.DORA_R}")
    print(f"DORA_ALPHA: {Config.DORA_ALPHA}")
    print(f"DORA_DROPOUT: {Config.DORA_DROPOUT}")
    print(f"DORA_TARGET_MODULES: {Config.DORA_TARGET_MODULES}")
    print(f"DORA_USE_MAGNITUDE: {Config.DORA_USE_MAGNITUDE}")
    
    print("\n--- RSLora配置 ---")
    print(f"USE_RSLORA: {Config.USE_RSLORA}")
    print(f"RSLORA_R: {Config.RSLORA_R}")
    print(f"RSLORA_ALPHA: {Config.RSLORA_ALPHA}")
    print(f"RSLORA_DROPOUT: {Config.RSLORA_DROPOUT}")
    print(f"RSLORA_TARGET_MODULES: {Config.RSLORA_TARGET_MODULES}")
    
    print("\n--- 其他配置 ---")
    print(f"MODEL_NAME: {Config.MODEL_NAME}")
    print(f"MAX_LENGTH: {Config.MAX_LENGTH}")
    print(f"NUM_TRAIN_EPOCHS: {Config.NUM_TRAIN_EPOCHS}")
    print(f"LEARNING_RATE: {Config.LEARNING_RATE}")
    print(f"DEVICE: {Config.DEVICE}")
    
    print("\n--- CPU Batch Size ---")
    print(f"CPU Train Batch Size: {Config.get_cpu_train_batch_size()}")
    print(f"CPU Eval Batch Size: {Config.get_cpu_eval_batch_size()}")
    
    print("\n--- 组合微调优势 ---")
    if Config.DORA_USE_MAGNITUDE and Config.USE_RSLORA:
        print("✅ 同时启用DoRA幅度分解和RSLora秩稳定化")
        print("✅ 获得最佳的参数效率和性能表现")
        print("✅ 训练过程更加稳定可靠")
    elif Config.DORA_USE_MAGNITUDE:
        print("⚠️ 仅启用DoRA幅度分解")
        print("💡 建议启用USE_RSLORA=true获得完整组合优势")
    elif Config.USE_RSLORA:
        print("⚠️ 仅启用RSLora秩稳定化")
        print("💡 建议启用DORA_USE_MAGNITUDE=true获得完整组合优势")
    else:
        print("❌ 未启用任何高级微调技术")
        print("💡 建议在.env中设置USE_RSLORA=true和DORA_USE_MAGNITUDE=true")

if __name__ == "__main__":
    test_combined_config()
