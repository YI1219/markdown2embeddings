from sentence_transformers import SentenceTransformer
import numpy as np


def get_embeddings(model, texts):
    # 批量生成 embeddings
    embeddings = model.encode(texts, batch_size=32, convert_to_numpy=True)  # batch_size 根据 GPU 内存调整
    return embeddings

def get_qwen3_emb_model():
    return SentenceTransformer("Qwen/Qwen3-Embedding-4B")