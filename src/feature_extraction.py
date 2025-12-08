
"""
使用emotion2vec提取特征
支持批量处理和多GPU加速
"""

import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from funasr import AutoModel
import pandas as pd
import json


class Emotion2VecFeatureExtractor:
    """emotion2vec特征提取器"""
    
    def __init__(self, model_name="iic/emotion2vec_base", 
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 hub="ms"):
        """
        参数:
            model_name: 模型名称
            device: 计算设备
            hub: 模型仓库 ("ms" for ModelScope, "hf" for HuggingFace)
        """
        self.device = device
        print(f"使用设备: {self.device}")
        print(f"加载模型: {model_name}")
        
        # 加载emotion2vec模型
        self.model = AutoModel(
            model=model_name,
            hub=hub,
            device=self.device
        )
        
        print("模型加载完成!")
    
    def extract_features(self, audio_path, granularity="utterance"):
        """
        单条音频 -> np.ndarray (768,) 或 (T, 768)
        """
        try:
            # generate 返回 list[dict]
            results = self.model.generate(audio_path,
                                          granularity=granularity,
                                          extract_embedding=True)
            if not isinstance(results, list) or len(results) == 0:
                raise RuntimeError("generate 返回空列表")

            # 取第一条 utterance 的特征
            feats = results[0].get("feats")
            if feats is None:
                raise RuntimeError("字典中无 feats 字段")

            # 统一转成 numpy
            feats = np.asarray(feats)
            return feats

        except Exception as e:
            print(f"[ERROR] 特征提取失败 {audio_path}: {e}")
            return None
        
    def extract_dataset_features(self, data_dir, output_dir, 
                                 granularity="utterance", batch_size=32):
        """
        批量提取数据集特征
        
        参数:
            data_dir: 数据目录
            output_dir: 输出目录
            granularity: 特征粒度
            batch_size: 批处理大小（暂未实现真正的批处理）
        """
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 读取元数据
        metadata_file = data_dir / 'metadata.csv'
        if not metadata_file.exists():
            raise FileNotFoundError(f"未找到元数据文件: {metadata_file}")
        
        df = pd.read_csv(metadata_file)
        
        print(f"\n提取特征: {data_dir.name}")
        print(f"总样本数: {len(df)}")
        print(f"特征粒度: {granularity}")
        
        # 提取特征
        features_list = []
        labels_list = []
        filenames_list = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            audio_path = row['path']
            
            # 提取特征
            features = self.extract_features(audio_path, granularity)
            
            if features is not None:
                features_list.append(features)
                labels_list.append(row['emotion_id'])
                filenames_list.append(row['filename'])
        
        # 保存特征
        features_array = np.array(features_list)
        labels_array = np.array(labels_list)
        
        np.save(output_dir / 'features.npy', features_array)
        np.save(output_dir / 'labels.npy', labels_array)
        
        # 保存文件名映射
        with open(output_dir / 'filenames.txt', 'w') as f:
            for filename in filenames_list:
                f.write(f"{filename}\n")
        
        # 保存特征信息
        feature_info = {
            'num_samples': len(features_list),
            'feature_dim': features_array.shape[1] if len(features_array.shape) == 2 else features_array.shape,
            'granularity': granularity,
            'model': self.model.model_path if hasattr(self.model, 'model_path') else 'unknown'
        }
        
        with open(output_dir / 'feature_info.json', 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        print(f"\n特征保存完成:")
        print(f"  特征shape: {features_array.shape}")
        print(f"  标签shape: {labels_array.shape}")
        print(f"  输出目录: {output_dir}")
        
        return features_array, labels_array


def extract_all_splits(data_root, output_root, model_name="iic/emotion2vec_plus_large"):
    """提取所有数据集分割的特征"""
    
    extractor = Emotion2VecFeatureExtractor(model_name=model_name)
    
    splits = ['train', 'val', 'test']
    
    for split in splits:
        data_dir = Path(data_root) / split
        output_dir = Path(output_root) / split
        
        if data_dir.exists():
            extractor.extract_dataset_features(data_dir, output_dir)
        else:
            print(f"跳过 {split}: 目录不存在")


if __name__ == '__main__':
    # 配置路径
    DATA_ROOT = 'data/processed/CASIA'
    OUTPUT_ROOT = 'data/features/CASIA'
    
    # 提取所有数据集的特征
    extract_all_splits(DATA_ROOT, OUTPUT_ROOT)