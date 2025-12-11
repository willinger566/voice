"""
emotion2vec微调训练脚本 
使用CASIA数据集进行情感分类器训练
包含特征标准化(StandardScaler)、BatchNorm优化、自动维度检测
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import argparse
from sklearn.preprocessing import StandardScaler
import joblib

class EmotionDataset(Dataset):
    """情感识别数据集"""
    
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = torch.FloatTensor(self.features[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        return feature, label


class EmotionClassifier(nn.Module):
    """情感分类器模型"""
    
    def __init__(self, input_dim, hidden_dim=256, num_classes=6, dropout=0.5):
        super(EmotionClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            # 第一层
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # 第二层
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # 输出层
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)


class EmotionTrainer:
    """训练器"""
    
    def __init__(self, model, train_loader, val_loader, 
                 device, lr=0.001, num_epochs=50):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
   
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )
        
        self.best_val_acc = 0.0
        self.best_model_state = None
        
        self.history = {
            'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []
        }
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc='训练')
        for features, labels in pbar:
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.train_loader), accuracy_score(all_labels, all_preds)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in tqdm(self.val_loader, desc='验证'):
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return (total_loss / len(self.val_loader), 
                accuracy_score(all_labels, all_preds), 
                f1_score(all_labels, all_preds, average='weighted'))
    
    def train(self):
        print("\n" + "=" * 60)
        print("开始训练")
        print("=" * 60)
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-" * 60)
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, val_f1 = self.validate()
            
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            print(f"当前学习率: {current_lr:.6f}")
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                print(f"保存最佳模型 (验证准确率: {val_acc:.4f})")
        
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        print("\n" + "=" * 60)
        print(f"训练完成! 最佳验证准确率: {self.best_val_acc:.4f}")
        print("=" * 60)
        return self.history

def load_and_preprocess_data(args):
    print("\n加载并预处理数据...")
    
    train_feats = np.load(args.train_features)
    train_labels = np.load(args.train_labels)
    val_feats = np.load(args.val_features)
    val_labels = np.load(args.val_labels)
    
    print(f"原始训练集维度: {train_feats.shape}")
    
    # 时序特征处理
    if len(train_feats.shape) == 3:
        print("检测到时序特征，执行 Mean Pooling...")
        train_feats = np.mean(train_feats, axis=1)
    if len(val_feats.shape) == 3:
        val_feats = np.mean(val_feats, axis=1)
        
    # 标准化
    print("执行特征标准化 (StandardScaler)...")
    scaler = StandardScaler()
    train_feats = scaler.fit_transform(train_feats)
    val_feats = scaler.transform(val_feats)
    
    # 保存scaler
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, Path(args.output_dir) / 'scaler.pkl')
    
    return train_feats, train_labels, val_feats, val_labels

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. 加载数据
    train_feats, train_labels, val_feats, val_labels = load_and_preprocess_data(args)
    
    actual_input_dim = train_feats.shape[1]
    print(f"\n>>> 自动检测到的输入特征维度: {actual_input_dim}")
    if actual_input_dim != args.input_dim:
        print(f">>> 注意: 命令行参数 input_dim ({args.input_dim}) 与实际数据 ({actual_input_dim}) 不符，将使用实际数据维度。")
    
    # 2. 创建 Dataset
    train_dataset = EmotionDataset(train_feats, train_labels)
    val_dataset = EmotionDataset(val_feats, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0) # windows下建议num_workers设为0
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 3. 创建模型 
    print("\n创建模型...")
    model = EmotionClassifier(
        input_dim=actual_input_dim,  
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        dropout=args.dropout
    )
    
    # 4. 训练
    trainer = EmotionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        num_epochs=args.epochs
    )
    
    history = trainer.train()
    
    # 5. 保存
    torch.save(model.state_dict(), Path(args.output_dir) / 'best_model.pth')
    with open(Path(args.output_dir) / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
 
    model_config = {
        'input_dim': actual_input_dim, 
        'hidden_dim': args.hidden_dim,
        'num_classes': args.num_classes,
        'dropout': args.dropout,
        'best_val_acc': trainer.best_val_acc
    }
    
    with open(Path(args.output_dir) / 'model_config.json', 'w') as f:
        json.dump(model_config, f, indent=2)
    
    print(f"\n所有模型文件已保存到: {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # 数据参数
    parser.add_argument('--train_features', type=str, default='data/features/CASIA/train/features.npy')
    parser.add_argument('--train_labels', type=str, default='data/features/CASIA/train/labels.npy')
    parser.add_argument('--val_features', type=str, default='data/features/CASIA/val/features.npy')
    parser.add_argument('--val_labels', type=str, default='data/features/CASIA/val/labels.npy')
    
    # 模型参数
    parser.add_argument('--input_dim', type=int, default=768, help="默认值，会被实际数据覆盖")
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.5)
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--output_dir', type=str, default='models/finetuned')
    
    args = parser.parse_args()
    main(args)