"""
CASIA数据集下载和预处理脚本
功能：下载数据、组织文件结构、生成训练/验证/测试集
"""

import os
import shutil
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json

class CASIADataPreprocessor:
    """CASIA数据集预处理器"""
    
    def __init__(self, raw_data_dir, output_dir, target_sr=16000):
        """
        参数:
            raw_data_dir: 原始数据目录
            output_dir: 输出目录
            target_sr: 目标采样率
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.target_sr = target_sr
        
        # CASIA数据集情感标签映射
        self.emotion_map = {
            'neutral': 0,
            'happy': 1,
            'sad': 2,
            'angry': 3,
            'fear': 4,
            'surprise': 5
        }
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'train').mkdir(exist_ok=True)
        (self.output_dir / 'val').mkdir(exist_ok=True)
        (self.output_dir / 'test').mkdir(exist_ok=True)
    
    def scan_dataset(self):
        """扫描数据集，构建文件列表"""
        print("正在扫描数据集...")
        
        audio_files = []
        
        # CASIA数据集通常按照 speaker/emotion/filename.wav 的结构组织
        for speaker_dir in self.raw_data_dir.iterdir():
            if not speaker_dir.is_dir():
                continue
                
            speaker_id = speaker_dir.name
            
            for emotion_dir in speaker_dir.iterdir():
                if not emotion_dir.is_dir():
                    continue
                
                emotion = emotion_dir.name.lower()
                if emotion not in self.emotion_map:
                    continue
                
                for audio_file in emotion_dir.glob('*.wav'):
                    audio_files.append({
                        'path': str(audio_file),
                        'speaker': speaker_id,
                        'emotion': emotion,
                        'emotion_id': self.emotion_map[emotion],
                        'filename': audio_file.name
                    })
        
        print(f"找到 {len(audio_files)} 个音频文件")
        return pd.DataFrame(audio_files)
    
    def preprocess_audio(self, audio_path, output_path):
        """
        预处理单个音频文件
        - 重采样到目标采样率
        - 归一化音量
        - 去除静音段
        """
        try:
            # 加载音频
            y, sr = librosa.load(audio_path, sr=self.target_sr)
            
            # 归一化
            y = librosa.util.normalize(y)
            
            # 去除静音段（可选）
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            
            # 保存处理后的音频
            sf.write(output_path, y_trimmed, self.target_sr)
            
            return True
        except Exception as e:
            print(f"处理音频失败 {audio_path}: {str(e)}")
            return False
    
    def split_dataset(self, df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """划分训练集、验证集和测试集"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        # 按speaker分层划分，确保每个speaker的样本都出现在各个集合中
        train_list, temp_list = [], []
        
        for speaker in df['speaker'].unique():
            speaker_df = df[df['speaker'] == speaker]
            
            # 训练集
            train_df, temp_df = train_test_split(
                speaker_df, 
                test_size=(val_ratio + test_ratio),
                random_state=42,
                stratify=speaker_df['emotion']
            )
            
            # 验证集和测试集
            val_df, test_df = train_test_split(
                temp_df,
                test_size=test_ratio/(val_ratio + test_ratio),
                random_state=42,
                stratify=temp_df['emotion']
            )
            
            train_list.append(train_df)
            train_list.append(val_df)
            train_list.append(test_df)
        
        train_data = pd.concat([df for i, df in enumerate(train_list) if i % 3 == 0])
        val_data = pd.concat([df for i, df in enumerate(train_list) if i % 3 == 1])
        test_data = pd.concat([df for i, df in enumerate(train_list) if i % 3 == 2])
        
        return train_data, val_data, test_data
    
    def process_split(self, df, split_name):
        """处理单个数据集分割"""
        print(f"\n处理 {split_name} 集合...")
        
        split_dir = self.output_dir / split_name
        metadata = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            # 构建输出文件名
            output_filename = f"{row['speaker']}_{row['emotion']}_{row['filename']}"
            output_path = split_dir / output_filename
            
            # 预处理音频
            if self.preprocess_audio(row['path'], output_path):
                metadata.append({
                    'filename': output_filename,
                    'path': str(output_path),
                    'speaker': row['speaker'],
                    'emotion': row['emotion'],
                    'emotion_id': row['emotion_id']
                })
        
        # 保存元数据
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(split_dir / 'metadata.csv', index=False)
        
        # 生成wav.scp文件（Kaldi风格）
        with open(split_dir / 'wav.scp', 'w') as f:
            for _, row in metadata_df.iterrows():
                f.write(f"{row['filename']}\t{row['path']}\n")
        
        # 生成标签文件
        with open(split_dir / 'labels.txt', 'w') as f:
            for _, row in metadata_df.iterrows():
                f.write(f"{row['filename']}\t{row['emotion_id']}\n")
        
        print(f"{split_name} 集合处理完成: {len(metadata)} 个样本")
        return metadata_df
    
    def run(self):
        """执行完整的预处理流程"""
        print("=" * 60)
        print("开始CASIA数据集预处理")
        print("=" * 60)
        
        # 1. 扫描数据集
        df = self.scan_dataset()
        
        # 2. 划分数据集
        train_df, val_df, test_df = self.split_dataset(df)
        
        print(f"\n数据集划分:")
        print(f"  训练集: {len(train_df)} 样本")
        print(f"  验证集: {len(val_df)} 样本")
        print(f"  测试集: {len(test_df)} 样本")
        
        # 3. 处理各个分割
        train_meta = self.process_split(train_df, 'train')
        val_meta = self.process_split(val_df, 'val')
        test_meta = self.process_split(test_df, 'test')
        
        # 4. 保存数据集统计信息
        stats = {
            'total_samples': len(df),
            'train_samples': len(train_meta),
            'val_samples': len(val_meta),
            'test_samples': len(test_meta),
            'num_speakers': df['speaker'].nunique(),
            'num_emotions': len(self.emotion_map),
            'emotion_distribution': df['emotion'].value_counts().to_dict(),
            'target_sample_rate': self.target_sr
        }
        
        with open(self.output_dir / 'dataset_stats.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 60)
        print("数据预处理完成！")
        print(f"输出目录: {self.output_dir}")
        print("=" * 60)


if __name__ == '__main__':
    # 配置路径
    RAW_DATA_DIR = 'D:\\s3\\s3\\voice\\data\\CASIA'
    OUTPUT_DIR = 'data/processed/CASIA'
    
    # 创建预处理器并运行
    preprocessor = CASIADataPreprocessor(RAW_DATA_DIR, OUTPUT_DIR)
    preprocessor.run()