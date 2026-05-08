# 🌟 JiaJia-Search

**Multilingual Lightweight & High-Performance Hybrid Search Engine | Built for RAG**

***

## 🌏 语言切换 | Language Switch

👉 **Click to quickly switch**：

[中文文档](README_CN.md) ｜ [English Documentation](README.md)

***

# 📗 English Version

## 📖 Introduction

JiaJia-Search is a **Multilingual highly generalized, production-ready hybrid search framework** focusing on full-link optimization of **retrieval + fusion + reranking** for RAG (Retrieval-Augmented Generation) scenarios.

It supports **independent embedding generation** (directly storable in vector databases), **parallel acceleration of BM25 full-text search & vector search**, **RRF multi-way ranking fusion**, **CrossEncoder high-precision reranking**, and **position-aware weighted fusion**. All APIs are independently callable to meet diverse search requirements.

Core Advantages: **No third-party service dependencies, full ONNX CPU model acceleration, global singleton model loading, dynamic configuration without code intrusion, perfect balance of extreme performance and generalization**.

## ✨ Core Features



1. **Independent Embedding Generation**

   Support single/batch text vectorization, output numpy arrays directly, seamlessly connect to Milvus/FAISS/Chroma and other vector databases.

2. **Dual-Parallel Retrieval**

   BM25 full-text search + vector semantic search run **simultaneously**, significantly improving retrieval performance.

3. **RRF Intelligent Ranking Fusion**

   Support query weighting and top-rank bonus mechanism to solve multi-retrieval result fusion.

4. **CrossEncoder High-Precision Reranking**

   Adapt to ONNX models such as bge-reranker-v2-m3, support Sigmoid normalization + relevance level classification.

5. **Position-Aware Weighted Fusion**

   Dynamically allocate weights according to retrieval ranks, balancing exact match and semantic relevance.

6. **Global Singleton Model Loading**

   Vector model & rerank model loaded **only once**, resident in memory for reuse, eliminating repeated loading performance loss.

7. **Dynamic Configuration System**

   No code modification required, configure model paths and hyperparameters with one line of code.

8. **Fully Independent API Calls**

   All modules are decoupled; embedding/retrieval/reranking/fusion can be used separately for maximum generalization.

## 🚀 Quick Installation



```
  Install from PyPI

pip install jiajia-search

  Upgrade to latest version

pip install --upgrade jiajia-search
```

## ⚙️ Core Configuration

**Local ONNX model paths must be configured for first use**; the framework supports dynamic configuration without code changes.



```
from jiajia_search import config

  Global configuration (takes effect for the entire lifecycle)

config.setup(

    # Vector Embedding Model (Required)

    vector_model_path=r"Yourmultilingual-e5-small-onnxPath",

    vector_onnx_file="model.onnx",

    # CrossEncoder Rerank Model (Required)

    rerank_model_path=r"Yourbge-reranker-v2-m3-ONNXPath",

    rerank_onnx_file="model_int8.onnx",

    # Optional Parameters

    vector_dim=384,

    rrf_k=60,

    top_k_recall=30,

    default_query_weight=2.0

)
```

## 📌 Quick Start

### 1. Independent Embedding Generation (For Vector DB)



```
from jiajia_search import generate_embedding, generate_embeddings_batch

  Single text embedding

text = "How much is an iPhone?"

embedding = generate_embedding(text)

print(f"Vector Shape: {embedding.shape}")  # (384,)

  Batch text embeddings

texts = ["iPhone 15 Price", "Huawei Phone Quote", "Xiaomi Phone Price"]

embeddings = generate_embeddings_batch(texts)
```

### 2. Standalone BM25 Retrieval



```
from jiajia_search import bm25_fts_retrieval

query = "How much is an iPhone?"

documents = ["iPhone 15 Official Price", "Huawei Mate 70 Quote", "Xiaomi 14 Price"]

ranked_results = bm25_fts_retrieval(query, documents)
```

### 3. Standalone Vector Retrieval



```
from jiajia_search import vector_index_retrieval

ranked_results = vector_index_retrieval(query, documents)
```

### 4. Standalone CrossEncoder Reranking



```
from jiajia_search import rerank_with_normalize

norm_scores, rel_levels = rerank_with_normalize(query, documents)

print("Normalized Scores:", norm_scores)

print("Relevance Levels:", rel_levels)
```

### 5. Full Search & Rerank Pipeline



```
from jiajia_search import jiajia_search_pipeline

query = "How much is an iPhone?"

documents = [

    "iPhone 15 Price",

    "Apple Phone Official Price",

    "Latest Huawei Phone Quote",

    "Xiaomi Phone Price Inquiry",

    "How much is a used iPhone?"

]

  Run full pipeline

final_rank=jiajia_search_pipeline(query, documents)

====================================================================================================
🏆 JiaJia-Search Final Sort Results
====================================================================================================
Rank    Final Score    Re-ranking Score    Correlation Level             Document
----------------------------------------------------------------------------------------------------
Top1   0.3323    0.9858      Highly relevant         How much is a used iPhone
Top2   0.2879    0.9010      Highly relevant         "Apple Phone Official Price
Top3   0.2611    0.7971      Moderately relevant     iPhone 15 Price
Top4   0.0385    0.0047      Low relevance           Latest Huawei Phone Quote
Top5   0.0372    0.0013      Low relevance           Xiaomi Phone Price Inquiry
=================================================================================

```

## 🧠 Relevance Score Standard



| Score Range | Relevance Level     | Meaning             |
| --------- | ------------------- | ------------------- |
| 0.8 ~ 1.0 | Highly relevant     | Highly relevant     |
| 0.5 ~ 0.8 | Moderately relevant | Moderately relevant |
| 0.2 ~ 0.5 | Somewhat relevant   | Somewhat relevant   |
| 0.0 ~ 0.2 | Low relevance       | Low relevance       |

## 🏗️ Technical Architecture

### Architecture Flowchart (Core Process)



```
┌─────────────────────────────────────────────────────────────────────────────┐

│                       JiaJia-Search Hybrid Retrieval Pipeline               │

└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────┐

                              │   User Query    │

                              └────────┬────────┘

                                       │

                        ┌──────────────┼──────────────┐

                        ▼                             ▼

               ┌────────────────┐            ┌────────────────┐

               │ BM25 Full-Text │            │ Vector Semantic│

               │ Search (Jieba) │            │ Search (E5-ONNX)│

               └───────┬────────┘            └───────┬────────┘

                       │                             │

                       │  Parallel Execution        │

                       └──────────────┬──────────────┘

                                      │

                                      ▼

                          ┌───────────────────────┐

                          │ RRF Fusion + Rank Bonus│

                          │ RRF_K=60 + Top30 Kept  │

                          │ Top1: +0.05 / Top2-3: +0.02 │

                          └───────────┬───────────┘

                                      │

                                      ▼

                          ┌───────────────────────┐

                          │ CrossEncoder Reranking│

                          │ (bge-reranker-v2-ONNX)│

                          │ Sigmoid Normalization + Relevance Level │

                          └───────────┬───────────┘

                                      │

                                      ▼

                          ┌───────────────────────┐

                          │ Position-Aware Blending│

                          │ Top1-3: 75% RRF + 25% Rerank │

                          │ Top4-10: 60% RRF + 40% Rerank │

                          │ Top11+: 40% RRF + 60% Rerank  │

                          └───────────┬───────────┘

                                      │

                                      ▼

                          ┌───────────────────────┐

                          │ Final Ranked Results  │

                          │ (Relevance Level + Confidence Score) │

                          └───────────────────────┘
```

### Score Normalization & Fusion Strategy

#### Retrieval Backend Score Conversion



| Backend                | Raw Score Type    | Conversion Method           | Output Range |
| ---------------------- | ----------------- | --------------------------- | --------- |
| BM25 Full-Text Search  | BM25 Raw Score    | Absolute value (abs(score)) | 0 ~ 25+   |
| Vector Search          | Cosine Similarity | Direct use (native 0\~1)    | 0.0 ~ 1.0 |
| CrossEncoder Reranking | Logits Score      | Sigmoid Normalization       | 0.0 ~ 1.0 |

#### Core Fusion Strategy

JiaJia-Search adopts an industrial-grade workflow of "Dual-Parallel Retrieval + Intelligent Fusion + Precise Reranking" with the following core logic:



1. **Dual-Parallel Retrieval**: BM25 full-text search (keyword exact match) and vector semantic search (contextual semantic match) run simultaneously, improving performance by nearly 100%;

2. **RRF Multi-Way Fusion**: Combine dual-path results using Reciprocal Rank Fusion (RRF) with the formula `Σ(1/(RRF_K + rank + 1))` (RRF_K=60), and add bonuses for top-ranked documents (Top1+0.05, Top2-3+0.02);

3. **Candidate Selection**: Keep Top30 documents after fusion for reranking, balancing recall rate and computational cost;

4. **CrossEncoder Reranking**: Use a lightweight ONNX reranking model (bge-reranker-v2-m3) for fine-grained sorting of candidates, outputting 0\~1 normalized scores and relevance levels;

5. **Position-Aware Blending**: Dynamically assign weights based on RRF fusion ranks:

* Top1-3: Prioritize precise matching from retrieval (75% RRF score + 25% rerank score);

* Top4-10: Balance retrieval and reranking results (60% RRF score + 40% rerank score);

* Top11+: Rely on reranking model to correct semantic relevance (40% RRF score + 60% rerank score);

1. **Result Output**: Finally sort by weighted scores, with relevance levels (Highly/Moderately/Somewhat/Low relevant) for quick business-layer filtering.

## 📦 Supported Models

The framework runs on ONNX format, please download the following open-source models manually:



1. Multilingual Vector Embedding Model: [multilingual-e5-small-onnx](https://www.modelscope.cn/models/intfloat/multilingual-e5-small)

2. Multilingual Reranking Model: [bge-reranker-v2-m3-ONNX](https://www.modelscope.cn/models/onnx-community/bge-reranker-v2-m3-ONNX)

## ⚠️ Notes

1. Models must be in **ONNX format** (PyTorch native models are not supported);

2. Global singleton loading: models are loaded once and reused in memory;

3. Model paths must be configured via `config.setup()`;

4. All dependencies are installed automatically.

## Fixes and Updates

1. Fixed issues related to Linux installation and usage

## 📄 License

**MIT License**

Free for personal and commercial use.

## 🤝 Contribution & Feedback



* GitHub Repository: [jiajia-search](https://github.com/jiangnanboy/jiajia-search)

* Issue Feedback: GitHub Issues

* Feature Requests: Pull Requests & Issues are welcome



***
