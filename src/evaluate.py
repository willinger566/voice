"""
模型评估脚本 (修复版)
适配新的数据预处理流程（标准化 + 自动维度检测）
"""

import torch
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from train import EmotionClassifier, EmotionDataset
from torch.utils.data import DataLoader
import joblib  
def plot_confusion_matrix(cm, class_names, save_path):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def load_and_process_test_data(features_path, labels_path, scaler_path):
    """加载并预处理测试数据"""
    print(f"加载测试数据: {features_path}")
    test_feats = np.load(features_path)
    test_labels = np.load(labels_path)
    
    # 1. 处理维度 
    if len(test_feats.shape) == 3:
        print("执行 Mean Pooling...")
        test_feats = np.mean(test_feats, axis=1)
        
    # 2. 加载训练好的 Scaler 并进行标准化
    print(f"加载标准化参数: {scaler_path}")
    if not Path(scaler_path).exists():
        raise FileNotFoundError(f"找不到 {scaler_path}，请先运行训练脚本！")
        
    scaler = joblib.load(scaler_path)
    test_feats = scaler.transform(test_feats)
    print("测试集标准化完成")
    
    return test_feats, test_labels

def evaluate_model(model, test_loader, device, class_names):
    model.eval()
    all_preds, all_labels = [], []

    print("正在进行推理...")
    with torch.no_grad():
        for feats, labs in test_loader:
            feats = feats.to(device)
            logits = model(feats)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labs.numpy())

    all_preds = np.asarray(all_preds)
    all_labels = np.asarray(all_labels)

    # 获取测试集中实际出现的类别
    present_labels = np.unique(all_labels)
    present_class_names = [class_names[i] for i in present_labels]
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', labels=present_labels, zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', labels=present_labels, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', labels=present_labels, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=present_labels)
    
    report = classification_report(
        all_labels,
        all_preds,
        labels=present_labels,
        target_names=present_class_names,
        digits=4,
        output_dict=True,
        zero_division=0
    )

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'predictions': all_preds.tolist(),
        'labels': all_labels.tolist(),
        'present_labels': present_labels.tolist(),
        'present_class_names': present_class_names
    }

def main():
    # 配置
    MODEL_DIR = Path('models/finetuned')
    TEST_FEATURES = 'data/features/CASIA/test/features.npy'
    TEST_LABELS = 'data/features/CASIA/test/labels.npy'
    OUTPUT_DIR = Path('results/evaluation')
    SCALER_PATH = MODEL_DIR / 'scaler.pkl'
    
    CLASS_NAMES = ['neutral', 'happy', 'sad', 'angry', 'fear', 'surprise']
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 加载模型配置
    config_path = MODEL_DIR / 'model_config.json'
    if not config_path.exists():
        raise FileNotFoundError("找不到模型配置文件，请检查路径")
        
    with open(config_path, 'r') as f:
        model_config = json.load(f)
    
    # 2. 加载并预处理测试数据
    test_feats, test_labels = load_and_process_test_data(TEST_FEATURES, TEST_LABELS, SCALER_PATH)
    
    # 3. 创建 Dataset 和 Loader
 
    test_dataset = EmotionDataset(test_feats, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 4. 创建模型
    input_dim = model_config.get('input_dim', 768) 
    print(f"模型输入维度: {input_dim}")
    
    model = EmotionClassifier(
        input_dim=input_dim,
        hidden_dim=model_config['hidden_dim'],
        num_classes=model_config['num_classes'],
        dropout=model_config['dropout']
    )
    
    # 5. 加载权重
    model.load_state_dict(torch.load(MODEL_DIR / 'best_model.pth', map_location=device))
    model = model.to(device)
    print("模型加载完成!")
    
    # 6. 评估
    results = evaluate_model(model, test_loader, device, CLASS_NAMES)
    
    # 打印结果
    print("\n" + "=" * 60)
    print(f"准确率 (Accuracy): {results['accuracy']:.4f}")
    print(f"F1分数 (F1-Score): {results['f1_score']:.4f}")
    print("=" * 60)
    
    print("\n分类报告:")
    print(classification_report(
        results['labels'],
        results['predictions'],
        labels=results['present_labels'],
        target_names=results['present_class_names'],
        digits=4,
        zero_division=0
    ))
    
    # 保存图表
    cm = np.array(results['confusion_matrix'])
    plot_confusion_matrix(cm, results['present_class_names'], OUTPUT_DIR / 'confusion_matrix.png')
    
    # 保存结果 JSON
    with open(OUTPUT_DIR / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"\n评估完成，结果已保存至: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()