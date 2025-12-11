"""
推理脚本 - 支持单个音频文件和批量推理
"""

import torch
import numpy as np
from pathlib import Path
import json
from funasr import AutoModel
from train import EmotionClassifier
import argparse


class EmotionRecognizer:
    """情感识别器 - 端到端推理"""
    
    def __init__(self, emotion2vec_model, classifier_path, config_path, device='cpu'):
        """
        参数:
            emotion2vec_model: emotion2vec模型名称或路径
            classifier_path: 分类器模型路径
            config_path: 模型配置路径
            device: 计算设备
        """
        self.device = device
        
        # 加载emotion2vec模型
        print("加载emotion2vec模型...")
        self.feature_extractor = AutoModel(
            model=emotion2vec_model,
            hub="ms",
            device=self.device
        )
        
        # 加载分类器配置
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # 加载分类器模型
        print("加载分类器模型...")
        self.classifier = EmotionClassifier(
            input_dim=self.config['input_dim'],
            hidden_dim=self.config['hidden_dim'],
            num_classes=self.config['num_classes'],
            dropout=self.config['dropout']
        )
        
        self.classifier.load_state_dict(
            torch.load(classifier_path, map_location=device)
        )
        self.classifier = self.classifier.to(device)
        self.classifier.eval()
        
        # 类别名称
        self.class_names = ['neutral', 'happy', 'sad', 'angry', 'fear', 'surprise']
        
        print("模型加载完成!")
    
    def predict(self, audio_path):
        """预测单个音频的情感"""
        # 1. 提取特征 ➜ 返回 list[dict]
        results = self.feature_extractor.generate(
            audio_path,
            granularity="utterance",
            extract_embedding=True
        )
        if not isinstance(results, list) or len(results) == 0:
            raise RuntimeError("generate 返回空列表")
        # 2. 取第一条 utterance 的 768 维向量
        feats = results[0].get("feats")
        if feats is None:
            raise RuntimeError("未找到 feats 字段")

        # 3. 统一成 numpy
        feats = np.asarray(feats)          

        feats_tensor = torch.from_numpy(feats).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.classifier(feats_tensor)
            probs  = torch.softmax(logits, dim=1)[0]
            pred_id = torch.argmax(probs).item()

        return {
            'predicted_emotion': self.class_names[pred_id],
            'predicted_id': int(pred_id),
            'confidence': float(probs[pred_id]),
            'all_probabilities': {
                name: float(probs[i]) for i, name in enumerate(self.class_names)
            }
        }
    
    def batch_predict(self, audio_paths):
        """批量预测"""
        results = []
        for audio_path in audio_paths:
            try:
                result = self.predict(audio_path)
                result['audio_path'] = str(audio_path)
                results.append(result)
            except Exception as e:
                print(f"预测失败 {audio_path}: {e}")
        
        return results


def export_to_onnx(model, config, output_path):
    """导出模型为ONNX格式"""
    model.eval()
    
    # 创建示例输入
    dummy_input = torch.randn(1, config['input_dim'])
    
    # 导出
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"ONNX模型已保存到: {output_path}")


def main(args):
    # 创建识别器
    recognizer = EmotionRecognizer(
        emotion2vec_model=args.emotion2vec_model,
        classifier_path=args.classifier_path,
        config_path=args.config_path,
        device=args.device
    )
    
    if args.mode == 'single':
        # 单个文件推理
        result = recognizer.predict(args.audio_path)
        
        print("\n" + "=" * 60)
        print("推理结果")
        print("=" * 60)
        print(f"音频文件: {args.audio_path}")
        print(f"预测情感: {result['predicted_emotion']}")
        print(f"置信度: {result['confidence']:.4f}")
        print("\n所有类别概率:")
        for emotion, prob in result['all_probabilities'].items():
            print(f"  {emotion}: {prob:.4f}")
        print("=" * 60)
        
        # 保存结果
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n结果已保存到: {args.output}")
    
    elif args.mode == 'batch':
        # 批量推理
        audio_dir = Path(args.audio_dir)
        audio_files = list(audio_dir.glob('*.wav'))
        
        print(f"\n找到 {len(audio_files)} 个音频文件")
        results = recognizer.batch_predict(audio_files)
        
        # 保存结果
        output_file = args.output or 'batch_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"批量推理完成! 结果已保存到: {output_file}")
    
    elif args.mode == 'export':
        # 导出ONNX模型
        with open(args.config_path, 'r') as f:
            config = json.load(f)
        
        model = EmotionClassifier(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            num_classes=config['num_classes'],
            dropout=config['dropout']
        )
        
        model.load_state_dict(
            torch.load(args.classifier_path, map_location='cpu')
        )
        
        export_to_onnx(model, config, args.output or 'model.onnx')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='情感识别推理')
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['single', 'batch', 'export'],
                       help='运行模式')
    parser.add_argument('--emotion2vec_model', type=str,
                       default='iic/emotion2vec_base')
    parser.add_argument('--classifier_path', type=str,
                       default='models/finetuned/best_model.pth')
    parser.add_argument('--config_path', type=str,
                       default='models/finetuned/model_config.json')
    parser.add_argument('--audio_path', type=str,
                       help='单个音频文件路径')
    parser.add_argument('--audio_dir', type=str,
                       help='音频目录（批量模式）')
    parser.add_argument('--output', type=str,
                       help='输出文件路径')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'])
    
    args = parser.parse_args()
    main(args)