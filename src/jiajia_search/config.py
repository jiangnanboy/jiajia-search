"""
JiaJia-Search 动态配置类
通过代码动态设置模型路径、参数
"""

class Config:
    _instance = None

    # 单例模式：全局唯一配置
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_defaults()
        return cls._instance

    # 初始化默认参数（仅占位，用户必须覆盖）
    def _init_defaults(self):
        # ========== 向量 Embedding 模型（用户必须配置）==========
        self.VECTOR_MODEL_PATH = None
        self.VECTOR_ONNX_FILE = None
        self.VECTOR_DIM = 384

        # ========== 重排模型（用户必须配置）==========
        self.RERANK_MODEL_PATH = None
        self.RERANK_ONNX_FILE = None

        # ========== RRF 融合参数（可选配置）==========
        self.RRF_K = 60
        self.TOP_K_RECALL = 30
        self.DEFAULT_QUERY_WEIGHT = 2.0
        self.DEFAULT_BONUS_RANK1 = 0.05
        self.DEFAULT_BONUS_RANK2_3 = 0.02

        # ========== 位置感知融合权重（可选配置）==========
        self.BLEND_WEIGHTS = {
            "top1-3": {"retrieval": 0.75, "reranker": 0.25},
            "top4-10": {"retrieval": 0.60, "reranker": 0.40},
            "top11+": {"retrieval": 0.40, "reranker": 0.60}
        }

    # ========== 【核心】用户配置方法：一行设置所有参数 ==========
    def setup(
        self,
        # 必选：向量模型
        vector_model_path: str,
        vector_onnx_file: str,
        # 必选：重排模型
        rerank_model_path: str,
        rerank_onnx_file: str,
        # 可选：向量维度
        vector_dim: int = 384,
        # 可选：RRF参数
        rrf_k: int = 60,
        top_k_recall: int = 30,
        default_query_weight: float = 2.0
    ):
        """
        全局配置 JiaJia-Search
        【必选】设置本地ONNX模型路径
        【可选】自定义检索/融合参数
        """
        # 向量模型
        self.VECTOR_MODEL_PATH = vector_model_path
        self.VECTOR_ONNX_FILE = vector_onnx_file
        self.VECTOR_DIM = vector_dim

        # 重排模型
        self.RERANK_MODEL_PATH = rerank_model_path
        self.RERANK_ONNX_FILE = rerank_onnx_file

        # RRF参数
        self.RRF_K = rrf_k
        self.TOP_K_RECALL = top_k_recall
        self.DEFAULT_QUERY_WEIGHT = default_query_weight

        print("✅ JiaJia-Search 配置已全局生效！")

# 创建全局配置实例（用户直接用这个）
config = Config()