import json
import os
import tempfile
from typing import List
from PyMuPDF import fitz  # 修改这一行

import httpx
import lancedb
import openai
from lancedb.pydantic import LanceModel, Vector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich import print
from rich.console import Console
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel

from indexify.functions_sdk.graph import Graph
from indexify.functions_sdk.image import Image
from indexify.functions_sdk.indexify_functions import indexify_function, IndexifyFunction

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

# TODO User set this
contextual_retrieval_prompt = """
这是我们想要在整个文档中定位的文本块：
<chunk>
{chunk_content}
</chunk>

请提供一个简洁的上下文描述，以帮助我们理解这个文本块在整个文档中的位置和重要性，从而改善对这个文本块的搜索检索。
只需回答简洁的上下文描述，不要添加其他内容。请使用中文回答。
"""


image = (
    Image().name("tensorlake/contextual-rag")
    .run("pip install indexify")
    .run("pip install sentence-transformers")
    .run("pip install lancedb")
    .run("pip install openai")
    .run("pip install langchain")
)


client = openai.OpenAI(
    api_key="sk-iM6Jc42voEnIOPSKJfFY0ri7chsz4D13sozKyqg403Euwv5e",
    base_url="https://api.chatanywhere.tech/v1"
)


class ChunkContext(BaseModel):
    chunks: List[str]
    chunk_contexts: List[str]
    chunk_ids: List[str]  # 添加chunk_ids到输出

def extract_patient_name(text):
    # 尝试匹配"姓名 xxx"格式
    match = re.search(r'姓名\s+([^\s]+)', text)
    if match:
        return match.group(1).strip()
    
    # 如果上面失败，尝试匹配"xxx患者"格式
    match = re.search(r'([^\s]+患者)', text)
    if match:
        return match.group(1).strip()
    
    return "未知患者"

@indexify_function(image=image)
def generate_chunk_contexts(doc: str, file_name: str) -> ChunkContext:
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
        chunk_id = f"{patient_name}_{i}"  # 创建唯一的chunk ID
        chunk_with_name = f"患者姓名：{patient_name}\n\nchunk_id: {chunk_id}\n\n{chunk}"
        print(f"Processing chunk {i} of {len(chunks)} with size {len(chunk)}")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer precisely."},
                {"role": "system", "content": f"Answer using the contents of this document <document> {doc} </document>"},
                {"role": "user", "content": contextual_retrieval_prompt.format(chunk_content=chunk_with_name)}
            ]
        )

        print(f'oai prompt read from cache {response.usage}')

        chunks_list.append(chunk_with_name)
        chunks_context_list.append(response.choices[0].message.content)
        chunk_ids.append(chunk_id)

    output = ChunkContext(
        chunks=chunks_list,
        chunk_contexts=chunks_context_list,
        chunk_ids=chunk_ids,  # 添加chunk_ids输出
    )

    return output


class TextChunk(BaseModel):
    context_embeddings: List[List[float]]
    chunk_embeddings: List[List[float]]
    chunk: List[str]
    chunk_with_context: List[str]
    chunk_ids: List[str]  # 新增

class TextEmbeddingExtractor(IndexifyFunction):
    name = "text-embedding-extractor"
    description = "Extractor class that captures an embedding model"
    system_dependencies = []
    input_mime_types = ["text"]
    image = image

    def __init__(self):
        super().__init__()
        self.model = get_model()  # 使用 get_model 函数获取模型

    def run(self, input: ChunkContext) -> TextChunk:
        context_embeddings = []
        chunk_embeddings = []
        chunks = []
        chunk_with_contexts = []

        for chunk, context, chunk_id in zip(input.chunks, input.chunk_contexts, input.chunk_ids):
            context_embedding = self.model.encode(chunk + '-\n' + context)
            chunk_embedding = self.model.encode(chunk)

            context_embeddings.append(context_embedding.tolist())
            chunk_embeddings.append(chunk_embedding.tolist())
            chunks.append(chunk)
            chunk_with_contexts.append(context)

        return TextChunk(
            context_embeddings=context_embeddings,
            chunk_embeddings=chunk_embeddings,
            chunk=chunks,
            chunk_with_context=chunk_with_contexts,
            chunk_ids=input.chunk_ids  # 新增
        )


class ContextualChunkEmbeddingTable(LanceModel):
    vector: Vector(384)
    chunk: str
    chunk_with_context: str
    chunk_id: str  # 新增


class ChunkEmbeddingTable(LanceModel):
    vector: Vector(384)
    chunk: str
    chunk_id: str  # 新增


class LanceDBWriter(IndexifyFunction):
    name = "lancedb_writer_context_rag"
    image = image

    def __init__(self):
        super().__init__()
        self._client = lancedb.connect("vectordb.lance")
        self._contextual_chunk_table = self._client.create_table(
            "contextual-chunk-embeddings", schema=ContextualChunkEmbeddingTable, exist_ok=True
        )

        self._chunk_table = self._client.create_table(
            "chunk-embeddings", schema=ChunkEmbeddingTable, exist_ok=True
        )

    def run(self, input: TextChunk) -> bool:
        for context_embedding, chunk_embedding, chunk, context, chunk_id in (
                zip(input.context_embeddings, input.chunk_embeddings, input.chunk, input.chunk_with_context, input.chunk_ids)
        ):
            self._contextual_chunk_table.add(
                [
                    ContextualChunkEmbeddingTable(vector=context_embedding, chunk=chunk, chunk_with_context=context, chunk_id=chunk_id)
                ]
            )

            self._chunk_table.add(
                [
                    ChunkEmbeddingTable(vector=chunk_embedding, chunk=chunk, chunk_id=chunk_id)
                ]
            )

        return True


def rag_call(payload):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "你是一个AI助手，负责分析文档并回答相关问题。请用中文提供你的答案。在回答的开头，请严格按照'患者姓名：<姓名>'的格式提供患者姓名，然后换行继续你的回答。所有输出，包括'答案'、'解释'、'信心'等标签，以及任何上下文描述，都应该使用中文。"},
            {"role": "user", "content": payload}
        ]
    )

    return response.choices[0].message.content, response.usage


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


def process_document(doc, file_name):
    try:
        logging.info(f"开始处理文档: {file_name}")
        g: Graph = Graph("test", start_node=generate_chunk_contexts)
        g.add_edge(generate_chunk_contexts, TextEmbeddingExtractor)
        g.add_edge(TextEmbeddingExtractor, LanceDBWriter)
        g.run(block_until_done=True, doc=doc, file_name=file_name)
        logging.info(f"文档 {file_name} 处理完成")
    except Exception as e:
        logging.error(f"处理文档 {file_name} 时出错: {str(e)}")
        logging.error(traceback.format_exc())
        raise


def answer_question(question):
    try:
        logging.info(f"开始回答问题: {question}")
        model = get_model()
        l_client = lancedb.connect("vectordb.lance")
        question_embd = model.encode(question)

        # RAG Output
        chunks = l_client.open_table("chunk-embeddings").search(question_embd).limit(20).to_list()
        
        logging.info("检索到的所有chunks:")
        for chunk in chunks:
            logging.info(f"Chunk {chunk['chunk_id']} 前100字符:\n{chunk['chunk'][:100]}...\n")

        d = []
        for chunk in chunks:
            d.append(f'chunk_id: {chunk["chunk_id"]}')
            d.append(f'chunk: {chunk["chunk"]}')
        p = '\n'.join(d)

        regular_prompt = f"""
        你面前有一份文档的信息。文档已被分成多个块，我们提供了每个块的ID和内容。信息以以下形式呈现：

            chunk_id: <块ID>
            chunk: <文本内容>
            chunk_id: <块ID>
            chunk: <文本内容>

        请仅基于提供的信息块回答问题。你的回答应包括：
        1. 问题的答案
        2. 解释为什么选择特定的引用
        3. 对你的答案的信心分数（0-100）
        4. 你用于回答的完整文本块内容（包括chunk_id和chunk内容）

        请严格按以下格式回答：
        答案：<你的答案>
        解释：<你选择特定引用的解释>
        信心：<你的信心分数>
        引用：
        <chunk_id1>
        <chunk1的完整内容>
        <chunk_id2>
        <chunk2的完整内容>
        ...

        非常重要：请确保你引用的chunk内容与你实际使用的信息完全一致。只引用你真正用于回答的文本块。

        问题：{question}

            {p}
        """

        rag_response, _ = rag_call(regular_prompt)
        logging.info(f"RAG 原始响应: {rag_response}")
        
        # 提取模型引用的chunk内容
        cited_chunks = extract_cited_chunks(rag_response)
        logging.info(f"提取的 chunks: {cited_chunks}")

        # 准备折叠显示的引用内容
        folded_citations = []
        for chunk_id, chunk_content in cited_chunks:
            folded_citations.append({
                "chunk_id": chunk_id,
                "content": chunk_content
            })

        # 将响应拆分成各个部分
        response_parts = rag_response.split("\n")
        patient_name = response_parts[0]
        answer = next(part for part in response_parts if part.startswith("答案："))
        explanation = next(part for part in response_parts if part.startswith("解释："))
        confidence = next(part for part in response_parts if part.startswith("信心："))

        # 构建新的响应格式
        formatted_response = {
            "patient_name": patient_name,
            "answer": answer,
            "explanation": explanation,
            "confidence": confidence,
            "citations": folded_citations
        }

        # Contextual RAG Output
        contextual_chunks = l_client.open_table("contextual-chunk-embeddings").search(question_embd).limit(20).to_list()
        d = []
        for chunk in contextual_chunks:
            d.append(f'chunk_id: {chunk["chunk_id"]}')
            d.append(f'chunk: {chunk["chunk"]}')
            d.append(f'chunk_context: {chunk["chunk_with_context"]}')
        p = '\n'.join(d)

        contextual_prompt = f"""
        你面前有一份文档的信息。文档已被分成多个块，我们提供了每个块的ID、内容和上下文。信息以以下形式呈现：
            chunk_id: <块ID>
            chunk: <文本内容>
            chunk_context: <上下文>
            chunk_id: <块ID>
            chunk: <文本内容>
            chunk_context: <上下文>

        请仅基于提供的信息块回答问题。你的回答应包括：
        1. 问题的答案
        2. 解释为何选择特定的引用
        3. 对你的答案的信心分数（0-100）
        4. 你用于回答的完整文本块内容（包括chunk_id、chunk内容和chunk_context）

        请按以下格式回答：
        答案：<你的答案>
        解释：<你选择特定引用的解释>
        信心：<你的信心分数>
        引用：
        <chunk_id1>
        <chunk1的完整内容>
        <chunk1的上下文>
        <chunk_id2>
        <chunk2的完整内容>
        <chunk2的上下文>
        ...

        问题：{question}

            {p}
        """

        contextual_response, _ = rag_call(contextual_prompt)
        
        # 提取上下文感知RAG的引用内容
        contextual_cited_chunks = extract_cited_chunks_with_context(contextual_response)
        
        # 准备上下文感知RAG的折叠显示引用内容
        contextual_folded_citations = []
        for chunk_id, chunk_content, chunk_context in contextual_cited_chunks:
            contextual_folded_citations.append({
                "chunk_id": chunk_id,
                "content": chunk_content,
                "context": chunk_context
            })

        # 将上下文感知RAG响应拆分成各个部分
        contextual_response_parts = contextual_response.split("\n")
        contextual_patient_name = contextual_response_parts[0]
        contextual_answer = next(part for part in contextual_response_parts if part.startswith("答案："))
        contextual_explanation = next(part for part in contextual_response_parts if part.startswith("解释："))
        contextual_confidence = next(part for part in contextual_response_parts if part.startswith("信心："))

        # 构建新的上下文感知RAG响应格式
        formatted_contextual_response = {
            "patient_name": contextual_patient_name,
            "answer": contextual_answer,
            "explanation": contextual_explanation,
            "confidence": contextual_confidence,
            "citations": contextual_folded_citations
        }

        logging.info("问题回答完成")
        return formatted_response, formatted_contextual_response
    except Exception as e:
        logging.error(f"回答问题时出错: {str(e)}")
        logging.error(traceback.format_exc())
        raise


def get_model():
    global global_model
    if global_model is None:
        logging.info("正在加载 SentenceTransformer 模型...")
        global_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        logging.info("SentenceTransformer 模型加载完成")
    return global_model


def extract_patient_names_from_chunks(chunks):
    names = set()
    for chunk in chunks:
        chunk_text = chunk['chunk']
        match = re.search(r'患者姓名：([^\n]+)', chunk_text)
        if match:
            names.add(match.group(1).strip())
    return list(names)

def validate_patient_name(response, chunks, rag_type):
    patient_name_in_response = extract_patient_name(response)
    patient_names_in_chunks = extract_patient_names_from_chunks(chunks)
    
    if patient_name_in_response not in patient_names_in_chunks:
        warning_message = f"\n\n警告：{rag_type}回答中的患者姓名 ({patient_name_in_response}) 与文本块中的患者姓名 {patient_names_in_chunks} 不一致，请仔细核对。"
        logging.warning(warning_message)
        return response + warning_message
    
    # 添加内容一致性检查
    for chunk in chunks:
        if patient_name_in_response in chunk['chunk']:
            if not content_consistency_check(response, chunk['chunk']):
                warning_message = f"\n\n警告：回答内容与引用的文本块不一致，请仔细核对。"
                logging.warning(warning_message)
                return response + warning_message
    
    return response

def content_consistency_check(response, chunk_content):
    # 这里实现一个简单的内容一致性检查
    # 可以使用更复杂的NLP技术来提高准确性
    response_keywords = set(response.lower().split())
    chunk_keywords = set(chunk_content.lower().split())
    common_keywords = response_keywords.intersection(chunk_keywords)
    
    # 如共同关键词占响应关键词的比例低于某个阈值，认为不一致
    threshold = 0.5
    if len(common_keywords) / len(response_keywords) < threshold:
        return False
    return True

def additional_consistency_check(response, chunks):
    patient_name = extract_patient_name(response)
    relevant_chunks = [chunk for chunk in chunks if patient_name in chunk['chunk']]
    
    if not relevant_chunks:
        warning_message = f"\n\n警告：未找到与患者 {patient_name} 相关的文本块，请仔细核对。"
        logging.warning(warning_message)
        return response + warning_message
    
    for chunk in relevant_chunks:
        if not content_consistency_check(response, chunk['chunk']):
            warning_message = f"\n\n警告：回答内容与患者 {patient_name} 的相关文本块不一致，请仔细核对。"
            logging.warning(warning_message)
            return response + warning_message
    
    return response

def extract_cited_chunks(response):
    # 从回答中提取引用的chunk内容
    cited_chunks = []
    chunks = re.split(r'chunk_id:', response)
    for chunk in chunks[1:]:  # 跳过第一个元素，因为它是回答的其他部分
        lines = chunk.strip().split('\n', 1)
        if len(lines) == 2:
            chunk_id = lines[0].strip()
            chunk_content = lines[1].strip()
            cited_chunks.append((chunk_id, chunk_content))
    return cited_chunks

def print_cited_part(response):
    # 打印原始响应中的引用部分
    cited_part = re.search(r'引用：.*', response, re.DOTALL)
    if cited_part:
        logging.info(f"原始响应中的引用部分:\n{cited_part.group(0)}")
    else:
        logging.warning("在原始响应中未找到引用部分")

def get_frontend_chunk(chunk_id):
    l_client = lancedb.connect("vectordb.lance")
    chunk_table = l_client.open_table("chunk-embeddings")
    results = chunk_table.search().where(f"chunk_id = '{chunk_id}'").limit(1).to_list()
    
    if results:
        return results[0]['chunk']
    else:
        logging.warning(f"未找到 Chunk ID {chunk_id}")
        return None

def extract_cited_chunks_with_context(response):
    # 从上下文感知RAG回答中提取引用的chunk内容和上下文
    cited_chunks = []
    chunks = re.split(r'chunk_id:', response)
    for chunk in chunks[1:]:  # 跳过第一个元素，因为它是回答的其他部分
        lines = chunk.strip().split('\n', 2)
        if len(lines) == 3:
            chunk_id = lines[0].strip()
            chunk_content = lines[1].strip()
            chunk_context = lines[2].strip()
            cited_chunks.append((chunk_id, chunk_content, chunk_context))
    return cited_chunks

# 在文件末尾添加以下代码，确保在导入时就加载模型
get_model()

# ... [其他函数和类保持不变] ...