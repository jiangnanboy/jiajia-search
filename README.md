# 🌟 JiaJia-Search

**多语言轻量级高性能混合检索引擎 | 专为 RAG 全链路检索 & 重排设计**

***

## 🌏 语言切换 | Language Switch

👉 **点击快速切换**：

[中文文档](README_CN.md) ｜ [English Documentation](README_EN.md)


***

# 📘 中文版本

## 📖 项目介绍

JiaJia-Search 是一款**多语言、泛化性极强、生产级可用**的混合检索框架，聚焦于 RAG（检索增强生成）场景的**召回 + 融合 + 重排**全链路优化。

框架支持**独立向量生成**（可直接存入向量数据库）、**BM25 全文检索与向量检索并行加速**、**RRF 多路排序融合**、**CrossEncoder 高精度重排**以及**位置感知加权融合**，所有 API 均可独立调用，满足多样化检索需求。

核心优势：**无第三方服务依赖、全 ONNX CPU模型加速、全局单例模型加载、动态配置无源码侵入、极致性能与泛化性兼顾**。

## ✨ 核心特性



1. **独立 Embedding 生成**

   支持单文本 / 批量文本生成向量，直接输出 numpy 数组，无缝对接 Milvus/FAISS/Chroma 等向量数据库。

2. **双路并行检索**

   BM25 全文检索 + 向量语义检索**同时执行**，检索性能显著提升。

3. **RRF 智能排序融合**

   支持查询加权、Top 排名奖励机制，解决多路检索结果融合问题。

4. **CrossEncoder 高精度重排**

   适配 bge-reranker-v2-m3 等 ONNX 模型，支持分数 Sigmoid 归一化 + 相关性等级判定。

5. **位置感知加权融合**

   按检索排名动态分配权重，兼顾精准匹配与语义相关性。

6. **全局单例模型加载**

   向量模型、重排模型**仅加载 1 次**，常驻内存复用，杜绝重复加载性能损耗。

7. **动态配置体系**

   无需修改源码，一行代码配置模型路径与超参，开箱即用。

8. **全 API 独立调用**

   所有模块解耦，可单独使用 Embedding / 检索 / 重排 / 融合能力，泛化性拉满。

## 🚀 快速安装



```
  官方 PyPI 安装

pip install jiajia-search

  升级最新版本

pip install --upgrade jiajia-search
```

## ⚙️ 核心配置

**首次使用必须配置本地 ONNX 模型路径**，框架支持动态配置，无需修改源码。



```
from jiajia_search import config

  全局配置（一次配置，全生命周期生效）

config.setup(

    # 向量 Embedding 模型（必填）

    vector_model_path=r"你的multilingual-e5-small-onnx模型路径",

    vector_onnx_file="model.onnx",

    # 重排 CrossEncoder 模型（必填）

    rerank_model_path=r"你的bge-reranker-v2-m3-ONNX模型路径",

    rerank_onnx_file="model_int8.onnx",

    # 可选参数

    vector_dim=384,

    rrf_k=60,

    top_k_recall=30,

    default_query_weight=2.0

)
```

## 📌 快速使用

### 1. 独立生成 Embedding（存入向量库）



```
from jiajia_search import generate_embedding, generate_embeddings_batch

  单文本生成 Embedding

text = "苹果手机多少钱"

embedding = generate_embedding(text)

print(f"向量维度: {embedding.shape}")  # (384,)

  批量文本生成 Embedding

texts = ["iPhone 15 售价", "华为手机报价", "小米手机价格"]

embeddings = generate_embeddings_batch(texts)
```

### 2. 独立 BM25 全文检索


```
from jiajia_search import bm25_fts_retrieval

query = "苹果手机多少钱"

documents = ["iPhone 15 官方定价", "华为 Mate 70 报价", "小米 14 价格"]

ranked_results = bm25_fts_retrieval(query, documents)
```

### 3. 独立向量语义检索


```
from jiajia_search import vector_index_retrieval

ranked_results = vector_index_retrieval(query, documents)
```

### 4. 独立 CrossEncoder 重排



```
from jiajia_search import rerank_with_normalize

norm_scores, rel_levels = rerank_with_normalize(query, documents)

print("归一化分数:", norm_scores)

print("相关性等级:", rel_levels)
```

### 5. 完整检索重排流水线（一键调用）



```
from jiajia_search import jiajia_search_pipeline

query = "苹果手机多少钱"

documents = [

    "iPhone 15 售价多少",

    "苹果手机官方定价",

    "华为手机最新报价",

    "小米手机价格查询",

    "二手苹果手机多少钱"

]

  启动全流程检索

final_rank=jiajia_search_pipeline(query, documents)

====================================================================================================
🏆 JiaJia-Search 最终排序结果
====================================================================================================
排名   最终得分      重排分         相关性等级                    文档
----------------------------------------------------------------------------------------------------
Top1   0.3323    0.9858      Highly relevant         二手苹果手机多少钱
Top2   0.2879    0.9010      Highly relevant         苹果手机官方定价
Top3   0.2611    0.7971      Moderately relevant     iPhone 15 售价多少
Top4   0.0385    0.0047      Low relevance           华为手机最新报价
Top5   0.0372    0.0013      Low relevance           小米手机价格查询
=================================================================================

```

## 🧠 相关性分数标准



| 分数范围      | 相关性等级               | 中文释义 |
| --------- | ------------------- | ---- |
| 0.8 ~ 1.0 | Highly relevant     | 高度相关 |
| 0.5 ~ 0.8 | Moderately relevant | 中等相关 |
| 0.2 ~ 0.5 | Somewhat relevant   | 一般相关 |
| 0.0 ~ 0.2 | Low relevance       | 低相关  |

## 🏗️ 技术架构

### 架构流程图（核心流程）



```
┌─────────────────────────────────────────────────────────────────────────────┐

│                       JiaJia-Search Hybrid Retrieval Pipeline               │

└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────┐

                              │   用户查询 (User Query)  │

                              └────────┬────────┘

                                       │

                        ┌──────────────┼──────────────┐

                        ▼                             ▼

               ┌────────────────┐            ┌────────────────┐

               │   BM25 全文检索  │            │   向量语义检索   │

               │ (Jieba 分词 + FTS) │            │ (E5-ONNX 向量匹配) │

               └───────┬────────┘            └───────┬────────┘

                       │                             │

                       │ 双路并行执行 (Parallel Run)  │

                       └──────────────┬──────────────┘

                                      │

                                      ▼

                          ┌───────────────────────┐

                          │   RRF 融合 + 排名奖励   │

                          │  RRF _K=60 + Top30 保留  │

                          │  Top1: +0.05 / Top2-3: +0.02 │

                          └───────────┬───────────┘

                                      │

                                      ▼

                          ┌───────────────────────┐

                          │ CrossEncoder 重排      │

                          │ (bge-reranker-v2-ONNX) │

                          │ Sigmoid 归一化 + 相关性等级 │

                          └───────────┬───────────┘

                                      │

                                      ▼

                          ┌───────────────────────┐

                          │ 位置感知加权融合       │

                          │ Top1-3: 75% RRF + 25% 重排分 │

                          │ Top4-10: 60% RRF + 40% 重排分 │

                          │ Top11+: 40% RRF + 60% 重排分  │

                          └───────────┬───────────┘

                                      │

                                      ▼

                          ┌───────────────────────┐

                          │ 最终排序结果          │

                          │ (含相关性等级/置信分数) │

                          └───────────────────────┘
```

### 分数归一化与融合策略

#### 检索后端分数转换（Score Normalization）



| 检索后端            | 原始分数类型    | 转换方式             | 输出范围      |
| --------------- | --------- | ---------------- | --------- |
| BM25 全文检索       | BM25 原始分数 | 取绝对值（abs (score)） | 0 ~ 25+   |
| 向量检索            | 余弦相似度     | 直接使用（原生 0~1 分布）  | 0.0 ~ 1.0 |
| CrossEncoder 重排 | Logits 分数 | Sigmoid 函数归一化    | 0.0 ~ 1.0 |

#### 融合核心策略（Fusion Strategy）

JiaJia-Search 采用「双路并行召回 + 智能融合 + 精准重排」的工业级流程，核心逻辑如下：



1. **双路并行召回**：BM25 全文检索（关键词精准匹配）与向量语义检索（上下文语义匹配）同时执行，性能提升近 100%；

2. **RRF 多路融合**：使用 Reciprocal Rank Fusion（RRF）算法融合双路结果，公式为 `Σ(1/(RRF_K + rank + 1))`（RRF _K=60），并对 Top 排名文档追加奖励（Top1+0.05，Top2-3+0.02）；

3. **候选集筛选**：融合后保留 Top30 文档进入重排阶段，平衡召回率与计算成本；

4. **CrossEncoder 重排**：使用轻量级 ONNX 重排模型（bge-reranker-v2-m3）对候选集精细化排序，输出 0\~1 归一化分数与相关性等级；

5. **位置感知加权**：根据 RRF 融合后的排名动态分配权重：

* 前排文档（Top1-3）：优先保留检索阶段的精准匹配结果（75% RRF 分数 + 25% 重排分数）；

* 中排文档（Top4-10）：均衡检索与重排结果（60% RRF 分数 + 40% 重排分数）；

* 后排文档（Top11+）：依赖重排模型修正语义相关性（40% RRF 分数 + 60% 重排分数）；

1. **结果输出**：最终按加权分数排序，附带相关性等级（Highly/Moderately/Somewhat/Low relevant），支持业务层快速筛选。

## 📦 支持模型

框架基于 ONNX 格式运行，需自行下载以下开源模型：



1. 多语言向量 Embedding 模型：[multilingual-e5-small-onnx](https://www.modelscope.cn/models/intfloat/multilingual-e5-small)

2. 多语言重排模型：[bge-reranker-v2-m3-ONNX](https://www.modelscope.cn/models/onnx-community/bge-reranker-v2-m3-ONNX)

## ⚠️ 注意事项



1. 模型必须为 **ONNX 格式**，不支持 PyTorch 原生模型；

2. 模型采用**全局单例加载**，首次调用加载，后续常驻内存；

3. 必须通过 `config.setup()` 配置模型路径；

4. 所有依赖自动安装，无需手动配置环境。

## 📄 许可证

**MIT License**

自由使用、修改、分发，适用于个人与商业场景。

## 🤝 贡献与反馈



* GitHub 仓库：[jiajia-search](https://github.com/jiangnanboy/jiajia-search)

* 问题反馈：Issues 板块

* 功能建议：欢迎提交 PR 与 Issue

