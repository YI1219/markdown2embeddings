import sqlite3
import chromadb
from typing import List, Dict, Optional
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
import os
from pathlib import Path

from md_processor import ArxivPandocMarkdownSplitter
from utils.time_log import TimeLogger

class VectorDatabaseManager:
    def __init__(
        self,
        sqlite_db_path: str = "RAG.db",
        embedding_model_name: str = "Qwen3-Embedding-0.6B",
    ):
        """
        初始化数据库管理器
        
        参数:
            sqlite_db_path: SQLite数据库文件路径
            embedding_model_name: 嵌入模型名称
        """
        self.sqlite_db_path = sqlite_db_path
        self.embedding_model_name = embedding_model_name
        self.sqlite_conn = None

        # 初始化组件
        self._init_logger()
        self._init_sqlite()
        self._init_embedding_model()
        self._init_chroma()

    def _init_logger(self):
        """初始化日志系统"""
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def _init_sqlite(self):
        if hasattr(self, 'conn') and self.conn:
            self.sqlite_conn.close()
        """初始化SQLite数据库"""
        self.sqlite_conn = sqlite3.connect(
            self.sqlite_db_path,
            timeout=5,
            check_same_thread=False
        )
        self._create_sqlite_tables()

    def _create_sqlite_tables(self):
        """创建SQLite表结构"""
        cursor = self.sqlite_conn.cursor()
        # 文档表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS doc_table (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            file_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # metadata 表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS metadata_table (
            doc_id TEXT PRIMARY KEY,
            file_id TEXT,
            publish_time TEXT,
            title TEXT,
            author TEXT,
            dates TEXT,
            keywords TEXT,
            acknowledgements TEXT,
            raw_info TEXT,
            block_path TEXT
        );
        """)
        
        # 向量表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS embedding_table (
            doc_id TEXT PRIMARY KEY,
            embedding BLOB NOT NULL,
            FOREIGN KEY (doc_id) REFERENCES documents(id)
        );
        """)
        
        self.sqlite_conn.commit()
        self.logger.info("SQLite 表初始化完成")

    def _init_embedding_model(self):
        """加载嵌入模型"""
        self.logger.info(f"正在加载嵌入模型: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.logger.info("嵌入模型加载完成")

    def _init_chroma(self):
        """初始化Chroma客户端"""
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(name="arxiv_rag")
        self.logger.info("Chroma 向量数据库初始化完成")

    def add_document(self, doc):
        """
        添加文档到数据库
        """
        try:
            # 生成嵌入向量
            embedding = self.embedding_model.encode(doc.page_content, convert_to_numpy=True)
            
            # 存储到SQLite
            cursor = self.sqlite_conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO doc_table (id, text, file_path) VALUES (?, ?, ?)",
                (doc.metadata['id'], doc.page_content, doc.metadata['file_path']),
            )

            cursor.execute(
                """
                INSERT OR REPLACE INTO metadata_table 
                (doc_id, file_id, publish_time, title, author, 
                dates, keywords, acknowledgements, raw_info, block_path) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (doc.metadata['id'], doc.metadata['file_id'], doc.metadata['publish_time'], 
                doc.metadata['title'], doc.metadata['author'], doc.metadata['dates'], 
                doc.metadata['keywords'], doc.metadata['acknowledgements'], 
                doc.metadata['raw_info'], str(doc.metadata['block_path'])),
            )
            
            # 存储向量 (numpy数组转为bytes)
            cursor.execute(
                "INSERT OR REPLACE INTO embedding_table (doc_id, embedding) VALUES (?, ?)",
                (doc.metadata['id'], embedding.tobytes()),
            )

            self.sqlite_conn.commit()
            self.logger.info(f"文档 {doc.metadata['id']} 已保存到SQLite")
            
        except Exception as e:
            self.sqlite_conn.rollback()
            self.logger.error(f"添加文档失败: {str(e)}")
            raise

    def load_all_to_chroma(self):
        """将SQLite中的所有文档加载到Chroma"""
        try:
            cursor = self.sqlite_conn.cursor()
            
            # 获取所有文档和向量
            cursor.execute("""
            SELECT d.id, d.text, v.embedding 
            FROM doc_table d
            JOIN embedding_table v ON d.id = v.doc_id
            """)
            
            records = cursor.fetchall()
            if not records:
                self.logger.warning("SQLite中没有可加载的文档")
                return
            
            # 准备批量数据
            ids, texts, embeddings= [], [], []
            
            for doc_id, text, emb in records:
                ids.append(doc_id)
                texts.append(text)
                embeddings.append(np.frombuffer(emb, dtype=np.float32).tolist())
            
            batch_size = 1024

            for i in range(0, len(ids), batch_size):
                batched_ids, batched_texts, batched_emb = ids[i: i+batch_size], texts[i: i+batch_size], embeddings[i: i+batch_size]
                # 批量添加到Chroma
                self.collection.add(
                    ids=batched_ids,
                    documents=batched_texts,
                    embeddings=batched_emb,
                )
            
            self.logger.info(f"成功从SQLite加载 {len(ids)} 条文档到Chroma")
            
        except Exception as e:
            self.logger.error(f"加载数据到Chroma失败: {str(e)}")
            raise

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        语义搜索
        
        参数:
            query: 查询文本
            top_k: 返回结果数量
            
        返回:
            包含相似文档的列表，按相似度排序
        """
        try:
            # 生成查询向量
            query_embedding = self.embedding_model.encode(
                query, 
                convert_to_numpy=True
            ).tolist()
            
            # 执行查询
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
            )
            
            # 格式化结果
            formatted_results = []
            for i, (doc_id, text, distance) in enumerate(zip(
                results["ids"][0],
                results["documents"][0],
                results["distances"][0],
            )):
                formatted_results.append({
                    "rank": i + 1,
                    "id": doc_id,
                    "text": text,
                    "score": 1 - distance,  # 转换为相似度分数
                    "distance": distance,
                })
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"搜索失败: {str(e)}")
            raise

    def close(self):
        """关闭数据库连接"""
        if self.sqlite_conn:
            self.sqlite_conn.close()
            self.sqlite_conn = None
            self.logger.info("数据库连接已关闭")


def find_md_files(root_dir: str) -> List[str]:
    """
    递归查找所有Markdown文件路径
    :param root_dir: 根目录路径
    :return: 所有.md文件的绝对路径列表
    """
    md_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".md"):
                full_path = os.path.join(dirpath, filename)
                md_files.append(full_path)
    return md_files


def get_db_manager(
    sqlite_db_path: str = "/home/reallm/yuyang/db/arxiv_rag.db", 
    embedding_model_name: str = "/home/reallm/yuyang/embedding_models/Qwen3-Embedding-0.6B"
):
    manager = VectorDatabaseManager(
        sqlite_db_path=sqlite_db_path,
        embedding_model_name=embedding_model_name
    )
    manager.load_all_to_chroma()
    return manager


if __name__ == "__main__":
    # # 检索用这个
    # manager = get_db_manager()
    # manager.search("food", top_k=8)
    # import pdb; pdb.set_trace()

    model_name = "/home/reallm/yuyang/embedding_models/Qwen3-Embedding-0.6B"

    # 新版操作
    # 初始化数据库
    print("Initializing Database...")
    manager = VectorDatabaseManager(
        sqlite_db_path="/home/reallm/yuyang/db/arxiv_rag.db",
        embedding_model_name=model_name
    )

    # 使用对应模型编码器
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 设置块最大 512 tokens，上下文最长 200 tokens
    splitter = ArxivPandocMarkdownSplitter.from_huggingface_tokenizer(
        tokenizer,
        max_chunk_length = 512,
        context_length = 200,
        separators = [" ", ""],
    )

    logger = TimeLogger()

    # 遍历 markdown 文件夹
    print("Retrievaling Folder...")
    root_dir = "/RLS002/Public/arxiv/process_script/arxiv2markdown/output"
    with os.scandir(root_dir) as it:
        for entry in it:
            if entry.is_dir():
                arxiv_id = entry.path.rsplit("/", 1)[-1]
                file_path = os.path.join(entry.path, arxiv_id+".md")
                if os.path.exists(file_path):
                    print(f"Processing MD File {file_path}...")
                    abs_chunks, body_chunks = splitter.process(file_path)
                    print(f"Chunk {file_path} OK")

                    # 3. 添加文档到数据库
                    print("Adding Document...")
                    for doc in (abs_chunks + body_chunks):
                        manager.add_document(doc)
                    print("All done")
                    logger.log()

                else:
                    with open(failed_record, 'a') as f:
                        f.write(entry.path + "\tFilename Wrong" + '\n')
                    logger.reset_time()
    manager.close()

    # 老版操作
    # # 初始化向量数据库
    # print("Initializing Database...")
    # manager = VectorDatabaseManager(
    #     sqlite_db_path="/home/reallm/yuyang/db/arxiv_rag.db",
    #     embedding_model_name="/home/reallm/yuyang/embedding_models/Qwen3-Embedding-0.6B"
    # )
    
    # # 遍历 markdown 文件夹
    # print("Retrievaling Folder...")
    # root_dir = "/RLS002/Public/arxiv/process_script/arxiv2markdown/output"
    # failed_record = "/RLS002/Public/arxiv/process_script/markdown2embeddings/temp_files/failed_markdown.line"
    # with os.scandir(root_dir) as it:
    #     for entry in it:
    #         if entry.is_dir():
    #             arxiv_id = entry.path.rsplit("/", 1)[-1]
    #             file_path = os.path.join(entry.path, arxiv_id+".md")
    #             if os.path.exists(file_path):
    #                 print(f"Processing MD File {file_path}...")
    #                 results = process_md(file_path, 2048)
    #                 if results[0]:
    #                     _, merged_chunks, metadatas, ids = results
    #                 else:
    #                     with open(failed_record, 'a') as f:
    #                         f.write(file_path + "\t" + results[1] + '\n')
    #                     continue
    #                 print(f"Chunk {file_path} OK")

    #                 # 3. 添加文档到数据库
    #                 print("Adding Document...")
    #                 for chunk, metadata, _id in zip(merged_chunks, metadatas, ids):
    #                     manager.add_document(chunk, _id, metadata)
    #                 print("All done")

    #             else:
    #                 with open(failed_record, 'a') as f:
    #                     f.write(entry.path + "\tFilename Wrong" + '\n')
    # manager.close()