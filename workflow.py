import json
import os
import tempfile
from typing import List
try:
    import fitz
except ImportError:
    from PyMuPDF import fitz

import httpx
import lancedb
import openai
from lancedb.pydantic import LanceModel, Vector as LanceVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich import print
from rich.console import Console
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
import numpy as np

import streamlit as st
import logging
import traceback
import re
import jieba
import jieba.posseg as pseg

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 在文件顶部添加全局变量
global_model = None

# 定义数据模型
class ChunkEmbeddingTable(LanceModel):
    vector: LanceVector(768)
    chunk: str
    chunk_id: str

class ContextualChunkEmbeddingTable(LanceModel):
    vector: LanceVector(768)
    chunk: str
    chunk_with_context: str
    chunk_id: str

class ChunkContext(BaseModel):
    chunks: List[str]
    chunk_contexts: List[str]
    chunk_ids: List[str]

# OpenAI 客户端配置
client = openai.OpenAI(
    api_key="sk-1pUmQlsIkgla3CuvKTgCrzDZ3r0pBxO608YJvIHCN18lvOrn",
    base_url="https://api.chatanywhere.tech/v1"
)

def generate_chunk_contexts(doc: str, file_name: str) -> ChunkContext:
    """生成文档块的上下文"""
    patient_name = os.path.splitext(file_name)[0]
    logging.info(f"从文件名中提取的患者姓名: {patient_name}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=75,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_text(doc)
    
    chunks_list = []
    chunks_context_list = []
    chunk_ids = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"{patient_name}_{i}"
        chunk_with_name = f"患者姓名：{patient_name}\n\nchunk_id: {chunk_id}\n\n{chunk}"
        print(f"Processing chunk {i} of {len(chunks)} with size {len(chunk)}")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer precisely."},
                {"role": "user", "content": f"请为以下文本块提供简洁的上下文描述，帮助理解其在整个文档中的位置和重要性：\n\n{chunk_with_name}"}
            ]
        )

        print(f'oai prompt read from cache {response.usage}')

        chunks_list.append(chunk_with_name)
        chunks_context_list.append(response.choices[0].message.content)
        chunk_ids.append(chunk_id)

    return ChunkContext(
        chunks=chunks_list,
        chunk_contexts=chunks_context_list,
        chunk_ids=chunk_ids,
    )

def process_embeddings(chunk_context: ChunkContext):
    """处理文本嵌入"""
    model = get_model()
    context_embeddings = []
    chunk_embeddings = []
    chunks = []
    chunk_with_contexts = []

    for chunk, context, chunk_id in zip(chunk_context.chunks, chunk_context.chunk_contexts, chunk_context.chunk_ids):
        context_embedding = model.encode(chunk + '-\n' + context)
        chunk_embedding = model.encode(chunk)

        context_embeddings.append(context_embedding.tolist())
        chunk_embeddings.append(chunk_embedding.tolist())
        chunks.append(chunk)
        chunk_with_contexts.append(context)

    return {
        'context_embeddings': context_embeddings,
        'chunk_embeddings': chunk_embeddings,
        'chunks': chunks,
        'contexts': chunk_with_contexts,
        'chunk_ids': chunk_context.chunk_ids
    }

def process_document(doc, file_name):
    """处理文档的主函数"""
    try:
        logging.info(f"开始处理文档: {file_name}")
        # 1. 生成文档块和上下文
        chunk_context = generate_chunk_contexts(doc, file_name)
        
        # 2. 处理嵌入
        embeddings_data = process_embeddings(chunk_context)
        
        # 3. 写入数据库
        success = write_to_lancedb(embeddings_data)
        
        if success:
            logging.info(f"文档 {file_name} 处理完成")
        else:
            logging.error(f"文档 {file_name} 处理失败")
            
    except Exception as e:
        logging.error(f"处理文档 {file_name} 时出错: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def write_to_lancedb(chunk_data):
    """将数据写入 LanceDB"""
    try:
        db_client = lancedb.connect("vectordb.lance")
        contextual_chunk_table = db_client.create_table(
            "contextual-chunk-embeddings", 
            schema=ContextualChunkEmbeddingTable, 
            exist_ok=True
        )
        chunk_table = db_client.create_table(
            "chunk-embeddings", 
            schema=ChunkEmbeddingTable, 
            exist_ok=True
        )

        # 写入数据
        for context_embedding, chunk_embedding, chunk, context, chunk_id in zip(
            chunk_data['context_embeddings'],
            chunk_data['chunk_embeddings'],
            chunk_data['chunks'],
            chunk_data['contexts'],
            chunk_data['chunk_ids']
        ):
            contextual_chunk_table.add([
                ContextualChunkEmbeddingTable(
                    vector=context_embedding,
                    chunk=chunk,
                    chunk_with_context=context,
                    chunk_id=chunk_id
                )
            ])

            chunk_table.add([
                ChunkEmbeddingTable(
                    vector=chunk_embedding,
                    chunk=chunk,
                    chunk_id=chunk_id
                )
            ])
        
        return True
    except Exception as e:
        logging.error(f"写入 LanceDB 时出错: {str(e)}")
        logging.error(traceback.format_exc())
        return False

def parse_pdf(file_path):
    """解析PDF文件并返回文本内容"""
    try:
        logging.info(f"开始解析PDF文件: {file_path}")
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        logging.info(f"PDF文件解析完成共 {len(text)} 个字符")
        return text
    except Exception as e:
        logging.error(f"解析PDF文件时出错: {str(e)}")
        logging.error(traceback.format_exc())
        raise

def extract_patient_name_from_question(question):
    """从问题中提取患者姓名"""
    # 匹配多种可能的问题模式
    patterns = [
        r'([^\s]+)得了什么病',
        r'([^\s]+)做了哪些检查',
        r'([^\s]+)的检查结果',
        r'([^\s]+)的病历',
        r'([^\s]+)的情况',
        r'关于([^\s]+)的',
        r'([^\s]+)的'  # 最通用的模式放在最后
    ]
    
    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            return match.group(1)
    
    # 如果没有匹配到任何模式，尝试提取问题中的 "某某" 格式的名字
    name_match = re.search(r'([^\s]+某某)', question)
    if name_match:
        return name_match.group(1)
    
    return None

def get_model():
    """获取或初始化 SentenceTransformer 模型"""
    global global_model
    if global_model is None:
        logging.info("正在加载 SentenceTransformer 模型...")
        global_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        logging.info("SentenceTransformer 模型加载完成")
    return global_model

def extract_cited_chunks(response):
    """从响应中提取引用的chunks"""
    citations = []
    lines = response.split('\n')
    in_citations = False
    current_chunk = {"chunk_id": None, "content": []}
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith("引用："):
            in_citations = True
            continue
            
        if in_citations:
            # 如果是新的 chunk_id
            if line.startswith("chunk_id:"):
                # 保存之前的 chunk（如果有的话）
                if current_chunk["chunk_id"] and current_chunk["content"]:
                    citations.append({
                        "chunk_id": current_chunk["chunk_id"],
                        "content": "\n".join(current_chunk["content"])
                    })
                # 开始新的 chunk
                current_chunk = {
                    "chunk_id": line.replace("chunk_id:", "").strip(),
                    "content": []
                }
            # 如果是内容部分
            elif current_chunk["chunk_id"]:
                current_chunk["content"].append(line)
            # 如果没有 chunk_id 但有内容，可能是直接的引用内容
            elif line:
                citations.append({
                    "chunk_id": "引用内容",
                    "content": line
                })
    
    # 保存最后一个 chunk
    if current_chunk["chunk_id"] and current_chunk["content"]:
        citations.append({
            "chunk_id": current_chunk["chunk_id"],
            "content": "\n".join(current_chunk["content"])
        })
    
    return citations

def answer_question(question):
    """回答问题的主函数"""
    try:
        logging.info(f"开始回答问题: {question}")
        
        # 从问题中提取患者姓名
        target_patient = extract_patient_name_from_question(question)
        if target_patient:
            logging.info(f"从问题中提取的患者姓名: {target_patient}")
        else:
            logging.info("未从问题中提取到患者姓名")

        model = get_model()
        l_client = lancedb.connect("vectordb.lance")
        question_embd = model.encode(question)

        # 获取所有匹配的文本块
        chunks = l_client.open_table("chunk-embeddings").search(question_embd).limit(10).to_list()
        
        # 如果指定了患者姓名，只保留相关患者的文本块
        if target_patient:
            filtered_chunks = []
            for chunk in chunks:
                if target_patient in chunk["chunk_id"]:
                    filtered_chunks.append(chunk)
            chunks = filtered_chunks[:5]
        else:
            chunks = chunks[:5]

        if not chunks:
            return {
                "patient_name": f"患者姓名：{target_patient if target_patient else '未知'}",
                "answer": "答案：未找到该患者的相关信息",
                "explanation": "解释：数据库中没有这位患者的病历记录",
                "confidence": "信心：0",
                "citations": []
            }

        logging.info(f"找到的相关文本块数量: {len(chunks)}")
        for chunk in chunks:
            logging.info(f"Chunk {chunk['chunk_id']} 前100字符:\n{chunk['chunk'][:100]}...\n")

        # 构建提示词
        d = []
        for chunk in chunks:
            content = chunk["chunk"]
            if len(content) > 500:
                content = content[:500] + "..."
            d.append(f'chunk_id: {chunk["chunk_id"]}\nchunk: {content}')
        p = '\n\n'.join(d)

        # 更明确的提示词格式要求
        regular_prompt = f"""请严格按照以下格式回答问题，并确保只回答关于患者 {target_patient} 的信息：

患者姓名：{target_patient}
答案：<简洁的回答>
解释：<为什么选择这些引用>
信心：<0-100分>
引用：
<使用的文本块>

问题：{question}

文本块：
{p}"""

        rag_response, _ = rag_call(regular_prompt)
        logging.info(f"RAG 原始响应: {rag_response}")

        def safe_extract_section(response_text, prefix):
            """安全地从响应中提取特定部分"""
            lines = response_text.split('\n')
            for line in lines:
                if line.startswith(prefix):
                    return line
            return f"{prefix}未找到"

        # 提取各个部分
        patient_name = f"患者姓名：{target_patient}"  # 直接使用提取的患者姓名
        answer = safe_extract_section(rag_response, "答案：")
        explanation = safe_extract_section(rag_response, "解释：")
        confidence = safe_extract_section(rag_response, "信心：")

        # 提取引用部分
        citations = []
        lines = rag_response.split('\n')
        in_citations = False
        current_citation = ""
        
        for line in lines:
            if line.startswith("引用："):
                in_citations = True
                continue
            
            if in_citations and line.strip():
                current_citation += line + "\n"
        
        if current_citation:
            citations.append({
                "chunk_id": "引用内容",
                "content": current_citation.strip()
            })

        # 如果没有提取到引用，使用原始文本块
        if not citations:
            for chunk in chunks[:2]:  # 只使用最相关的两个文本块
                citations.append({
                    "chunk_id": chunk["chunk_id"],
                    "content": chunk["chunk"][:500]  # 限制内容长度
                })

        # 构建新的响应格式
        formatted_response = {
            "patient_name": patient_name,
            "answer": answer,
            "explanation": explanation,
            "confidence": confidence,
            "citations": citations
        }

        return formatted_response

    except Exception as e:
        logging.error(f"回答问题时出错: {str(e)}")
        logging.error(traceback.format_exc())
        return {
            "patient_name": f"患者姓名：{target_patient if target_patient else '未知'}",
            "answer": f"答案：回答问题时出错: {str(e)}",
            "explanation": "解释：发生了处理错误",
            "confidence": "信心：0",
            "citations": []
        }

def rag_call(payload):
    """调用 RAG 系统"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system", 
                "content": "你是一个医疗助手，负责分析病历并回答问题。请严格按照以下格式回答：\n患者姓名：<姓名>\n答案：<回答>\n解释：<解释>\n信心：<0-100分>\n引用：<引用内容>"
            },
            {"role": "user", "content": payload}
        ]
    )
    return response.choices[0].message.content, response.usage