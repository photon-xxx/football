"""
基于SigLIP深度视觉特征的球队分类器

使用预训练的SigLIP视觉模型提取特征，UMAP降维，KMeans聚类来区分两支球队。
相比基于颜色的方法，这种方法更鲁棒，能捕捉纹理、形状等视觉模式。
"""

from typing import Generator, Iterable, List, TypeVar
import numpy as np

# 修复numpy版本兼容性问题 (numpy >= 1.24移除了np.object)
# 这需要在导入transformers之前修复，因为transformers的依赖onnx使用了np.object
if not hasattr(np, "object"):
    np.object = object

import supervision as sv
import torch
from umap import UMAP  # 正确的导入方式
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel

V = TypeVar("V")

# 本地模型路径（必需，模型必须已手动下载）
# 默认路径，可以在初始化时通过 local_model_path 参数覆盖
LOCAL_SIGLIP_MODEL_PATH = None  # 例如: "Storage/models/siglip/siglip-base-patch16-224"


def create_batches(
    sequence: Iterable[V], batch_size: int
) -> Generator[List[V], None, None]:
    """
    将序列分批处理
    
    Args:
        sequence: 输入序列
        batch_size: 批次大小
    
    Yields:
        批次列表
    """
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


class TeamClassifier:
    """
    基于SigLIP视觉模型的球队分类器
    
    使用预训练的SigLIP模型提取深度特征，UMAP降维，KMeans聚类来区分两支球队。
    """
    def __init__(self, device: str = 'cpu', batch_size: int = 32, local_model_path: str = None):
        """
        初始化TeamClassifier
        
        Args:
            device: 运行设备 ('cpu' 或 'cuda')
            batch_size: 批量处理图像的大小
            local_model_path: 本地模型路径（必需），如果为None则使用LOCAL_SIGLIP_MODEL_PATH
        """
        self.device = device
        self.batch_size = batch_size
        
        # 确定使用的模型路径（只支持本地路径）
        import os
        if local_model_path:
            model_path = local_model_path
        elif LOCAL_SIGLIP_MODEL_PATH and os.path.exists(LOCAL_SIGLIP_MODEL_PATH):
            model_path = LOCAL_SIGLIP_MODEL_PATH
        else:
            raise ValueError(
                f"未找到本地模型路径！\n"
                f"请设置 local_model_path 参数或修改 LOCAL_SIGLIP_MODEL_PATH 配置。\n"
                f"当前 LOCAL_SIGLIP_MODEL_PATH: {LOCAL_SIGLIP_MODEL_PATH}\n"
                f"模型路径必须包含以下文件：config.json, model.safetensors (或 pytorch_model.bin), preprocessor_config.json 等"
            )
        
        # 验证模型路径是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"模型路径不存在: {model_path}\n"
                f"请确保模型已下载到该路径"
            )
        
        # 验证必需文件是否存在
        required_files = ['config.json', 'preprocessor_config.json']
        model_files = ['model.safetensors', 'pytorch_model.bin']
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(os.path.join(model_path, file)):
                missing_files.append(file)
        
        # 检查模型权重文件（至少需要一个）
        has_model_file = any(os.path.exists(os.path.join(model_path, f)) for f in model_files)
        if not has_model_file:
            missing_files.append('model.safetensors 或 pytorch_model.bin')
        
        if missing_files:
            raise FileNotFoundError(
                f"模型路径缺少必需文件: {missing_files}\n"
                f"模型路径: {model_path}\n"
                f"请确保所有必需文件都已下载"
            )
        
        print(f"正在从本地路径加载SigLIP模型: {model_path}")
        
        try:
            # 只使用本地文件，不尝试从网络下载
            self.features_model = SiglipVisionModel.from_pretrained(
                model_path,
                local_files_only=True
            ).to(device)
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                local_files_only=True
            )
        except Exception as e:
            print(f"❌ 从本地路径加载模型失败: {e}")
            print(f"   模型路径: {model_path}")
            print(f"   请检查路径是否正确，确保所有必需文件都已下载")
            print(f"   必需文件: config.json, preprocessor_config.json, model.safetensors (或 pytorch_model.bin)")
            raise
        
        self.reducer = UMAP(n_components=3, random_state=42)
        self.cluster_model = KMeans(n_clusters=2, random_state=42, n_init=10)
        self.is_fitted = False
        print("✓ SigLIP模型加载完成")

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        从图像裁剪中提取特征
        
        Args:
            crops: 图像裁剪列表 (BGR格式的numpy数组)
        
        Returns:
            提取的特征数组
        """
        # 将OpenCV BGR图像转换为PIL RGB图像
        crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = create_batches(crops, self.batch_size)
        data = []
        
        with torch.no_grad():
            for batch in tqdm(batches, desc='提取视觉特征'):
                inputs = self.processor(
                    images=batch, return_tensors="pt").to(self.device)
                outputs = self.features_model(**inputs)
                # 对空间维度求平均，得到图像级特征
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)

        return np.concatenate(data)

    def fit(self, crops: List[np.ndarray]) -> None:
        """
        在图像裁剪上训练分类器
        
        Args:
            crops: 图像裁剪列表，用于训练
        """
        if len(crops) == 0:
            raise ValueError("训练数据不能为空")
        
        print(f"正在提取 {len(crops)} 个球员的特征...")
        data = self.extract_features(crops)
        
        print("正在使用UMAP降维...")
        projections = self.reducer.fit_transform(data)
        
        print("正在训练KMeans聚类器...")
        self.cluster_model.fit(projections)
        
        self.is_fitted = True
        print("分类器训练完成")

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        预测图像裁剪的队伍标签
        
        Args:
            crops: 图像裁剪列表
        
        Returns:
            预测的队伍标签数组 (0或1)
        """
        if not self.is_fitted:
            raise ValueError("分类器尚未训练，请先调用fit()方法")
        
        if len(crops) == 0:
            return np.array([])

        data = self.extract_features(crops)
        projections = self.reducer.transform(data)
        return self.cluster_model.predict(projections)

