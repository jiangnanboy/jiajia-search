import time

from src.jiajia_search import config

# 配置你的本地ONNX模型路径（一次配置，到处使用）
config.setup(
    # 必选：向量模型
    vector_model_path=r"E:\git_project\factory-qa\models\multilingual-e5-small-onnx",
    vector_onnx_file="model.onnx",
    # 必选：重排模型
    rerank_model_path=r"E:\git_project\factory-qa\models\bge-reranker-v2-m3-ONNX",
    rerank_onnx_file="model_int8.onnx",
    # 可选：自定义参数
    vector_dim=384
)


from src.jiajia_search import generate_embedding

# 文本转向量（直接存Milvus/FAISS）
text = "苹果手机多少钱"
start_time = time.time()
emb = generate_embedding(text)
print("耗时：")
print(time.time() - start_time)
print("Embedding向量：", emb.shape)  # (384,)


from src.jiajia_search import jiajia_search_pipeline

query = "苹果手机多少钱"
documents = ["iPhone 15售价多少", "苹果手机官方定价", "华为手机报价"]

# 一键搜索
start_time = time.time()
final_ranked = jiajia_search_pipeline(query, documents)
print(f'final-rank:{final_ranked}')
print("耗时：")
print(time.time() - start_time)



query = "How much is an iPhone?"

documents = [

    "iPhone 15 Price",

    "Apple Phone Official Price",

    "Latest Huawei Phone Quote",

    "Xiaomi Phone Price Inquiry",

    "How much is a used iPhone?"

]

final_rank=jiajia_search_pipeline(query, documents)

# 用户自定义参数
config.setup(
    # === 必选模型路径 ===
    vector_model_path="你的向量模型路径",
    vector_onnx_file="模型文件名",
    rerank_model_path="你的重排模型路径",
    rerank_onnx_file="模型文件名",

    # === 可选高级参数 ===
    vector_dim=384,  # 向量维度
    rrf_k=60,  # RRF融合系数
    top_k_recall=30,  # 重排候选数
    default_query_weight=2.0  # 查询权重
)

