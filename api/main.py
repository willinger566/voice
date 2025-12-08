"""
FastAPI Web服务
提供情感识别API接口
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import torch
import numpy as np
from pathlib import Path
import tempfile
import os
from typing import Optional
from pydantic import BaseModel
import json

# 导入推理模块
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from inference import EmotionRecognizer


# 创建FastAPI应用
app = FastAPI(
    title="中文语音情感识别API",
    description="基于emotion2vec的中文语音情感识别服务",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量：模型
recognizer = None


class PredictionResponse(BaseModel):
    """预测响应模型"""
    success: bool
    predicted_emotion: str
    predicted_id: int
    confidence: float
    all_probabilities: dict
    message: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    global recognizer
    
    print("正在加载模型...")
    
    # 配置路径
    EMOTION2VEC_MODEL = os.getenv("EMOTION2VEC_MODEL", "iic/emotion2vec_plus_large")
    CLASSIFIER_PATH = os.getenv("CLASSIFIER_PATH", "models/finetuned/best_model.pth")
    CONFIG_PATH = os.getenv("CONFIG_PATH", "models/finetuned/model_config.json")
    DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        recognizer = EmotionRecognizer(
            emotion2vec_model=EMOTION2VEC_MODEL,
            classifier_path=CLASSIFIER_PATH,
            config_path=CONFIG_PATH,
            device=DEVICE
        )
        print(f"模型加载完成! 使用设备: {DEVICE}")
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        raise


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "中文语音情感识别API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "info": "/info"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": recognizer is not None
    }


@app.get("/info")
async def get_info():
    """获取模型信息"""
    if recognizer is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    return {
        "model_name": "emotion2vec + EmotionClassifier",
        "num_classes": len(recognizer.class_names),
        "class_names": recognizer.class_names,
        "device": str(recognizer.device),
        "input_format": "16kHz mono WAV",
        "supported_emotions": {
            "discrete": recognizer.class_names,
            "dimensional": ["valence", "arousal", "dominance"]
        }
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_emotion(file: UploadFile = File(...)):
    """
    预测音频文件的情感
    
    参数:
        file: 上传的音频文件 (WAV格式, 16kHz推荐)
    
    返回:
        预测结果
    """
    if recognizer is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    # 检查文件格式
    if not file.filename.endswith(('.wav', '.mp3', '.flac')):
        raise HTTPException(
            status_code=400,
            detail="不支持的文件格式。请上传WAV/MP3/FLAC文件"
        )
    
    try:
        # 保存上传的文件到临时目录
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # 进行预测
        result = recognizer.predict(tmp_path)
        
        # 删除临时文件
        os.unlink(tmp_path)
        
        # 返回结果
        return PredictionResponse(
            success=True,
            predicted_emotion=result['predicted_emotion'],
            predicted_id=result['predicted_id'],
            confidence=result['confidence'],
            all_probabilities=result['all_probabilities'],
            message="预测成功"
        )
    
    except Exception as e:
        # 清理临时文件
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass
        
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@app.post("/predict_stream")
async def predict_emotion_stream(file: UploadFile = File(...)):
    """
    流式预测（实时处理）
    适用于实时语音输入
    """
    # TODO: 实现流式处理
    return {"message": "流式处理功能开发中"}


if __name__ == "__main__":
    # 运行服务
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )