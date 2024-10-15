import json
import os
import tempfile
from typing import List
import fitz  # PyMuPDF库用于处理PDF

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
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
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

def extract_patient_name(chunk):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "你是一个助手，负责从医疗记录中提取患者姓名。只返回患者姓名，不要有其他任何内容。"},
            {"role": "user", "content": f"从以下文本中提取患者姓名：\n\n{chunk}"}
        ]
    )
    return response.choices[0].message.content.strip()

@indexify_function(image=image)
def generate_chunk_contexts(doc: str) -> ChunkContext:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=75,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_text(doc)
    
    # 从第一个分片提取患者姓名
    patient_name = extract_patient_name(chunks[0])
    
    # 将患者姓名添加到每个分片
    chunks = [f"患者姓名：{patient_name}\n\n{chunk}" for chunk in chunks]

    chunks_list = []
    chunks_context_list = []

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i} of {len(chunks)} with size {len(chunk)}")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer precisely."},
                {"role": "system", "content": f"Answer using the contents of this document <document> {doc} </document>"},
                {"role": "user", "content": contextual_retrieval_prompt.format(chunk_content=chunk)}
            ]
        )

        print(f'oai prompt read from cache {response.usage}')

        chunks_list.append(chunk)
        chunks_context_list.append(response.choices[0].message.content)

    output = ChunkContext(
        chunks=chunks_list,
        chunk_contexts=chunks_context_list,
    )

    return output


class TextChunk(BaseModel):
    context_embeddings: List[List[float]]
    chunk_embeddings: List[List[float]]

    chunk: List[str]
    chunk_with_context: List[str]

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

        for chunk, context in zip(input.chunks, input.chunk_contexts):
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
        )


class ContextualChunkEmbeddingTable(LanceModel):
    vector: Vector(384)
    chunk: str
    chunk_with_context: str


class ChunkEmbeddingTable(LanceModel):
    vector: Vector(384)
    chunk: str


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
        for context_embedding, chunk_embedding, chunk, context in (
                zip(input.context_embeddings, input.chunk_embeddings, input.chunk, input.chunk_with_context)
        ):
            self._contextual_chunk_table.add(
                [
                    ContextualChunkEmbeddingTable(vector=context_embedding, chunk=chunk, chunk_with_context=context,)
                ]
            )

            self._chunk_table.add(
                [
                    ChunkEmbeddingTable(vector=chunk_embedding, chunk=chunk,)
                ]
            )

        return True


def rag_call(payload):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "你是一个AI助手，负责分析文档并回答相关问题。请用中文提供你的答案，解释为什么选择特定的引用，并给出你对答案的信心分数（0-100）。在回答的开头，请以'患者姓名：<姓名>'的格式提供患者姓名。"},
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
        logging.info(f"PDF文件解析完成，共 {len(text)} 个字符")
        return text
    except Exception as e:
        logging.error(f"解析PDF文件时出错: {str(e)}")
        logging.error(traceback.format_exc())
        raise


def process_document(doc):
    try:
        logging.info("开始处理文档")
        g: Graph = Graph("test", start_node=generate_chunk_contexts)
        g.add_edge(generate_chunk_contexts, TextEmbeddingExtractor)
        g.add_edge(TextEmbeddingExtractor, LanceDBWriter)
        g.run(block_until_done=True, doc=doc)
        logging.info("文档处理完成")
    except Exception as e:
        logging.error(f"处理文档时出错: {str(e)}")
        logging.error(traceback.format_exc())
        raise


def answer_question(question):
    try:
        logging.info(f"开始回答问题: {question}")
        model = get_model()  # 这里不会重新加载模型，只会获取已加载的模型
        l_client = lancedb.connect("vectordb.lance")
        question_embd = model.encode(question)

        # RAG Output
        chunks = l_client.open_table("chunk-embeddings").search(question_embd).limit(20).to_list()
        d = []
        for chunk_id, i in enumerate(chunks):
            d.append('chunk_id: ' + str(chunk_id))
            d.append('chunk: ' + i['chunk'])
        p = '\n'.join(d)

        regular_prompt = f"""
        你面前有一份文档的信息。文档已被分成多个块，我们提供了每个块的ID和内容。信息以以下形式呈现：

            chunk_id: <块ID>
            chunk: <文本内容>
            chunk_id: <块ID>
            chunk: <文本内容>

        请仅基于提供的信息块回答问题。你的回答应包括：
        1. 患者姓名（格式：患者姓名：<姓名>）
        2. 问题的答案
        3. 解释为什么选择特定的引用
        4. 对你的答案的信心分数（0-100）

        请按以下格式回答：
        患者姓名：<患者姓名>
        答案：<你的答案>
        解释：<你选择特定引用的解释>
        信心：<你的信心分数>
        引用：<使用的块ID>

        问题：{question}

            {p}
        """

        rag_response, _ = rag_call(regular_prompt)

        # 对 RAG 输出进行验证
        rag_response = validate_patient_name(rag_response, chunks, "RAG")

        # Contextual RAG Output
        chunks = l_client.open_table("contextual-chunk-embeddings").search(question_embd).limit(20).to_list()
        d = []
        for chunk_id, i in enumerate(chunks):
            d.append('chunk_id: ' + str(chunk_id))
            d.append('chunk: ' + i['chunk'])
            d.append('chunk_context: ' + i['chunk_with_context'])
        p = '\n'.join(d)

        contextual_prompt = f"""
        你面前有一份文档的信息。文档已被分成多个块，我们供了每个块的ID、内容和上下文。信息以以下形式呈现：
            chunk_id: <块ID>
            chunk: <文本内容>
            chunk_context: <上下文>
            chunk_id: <块ID>
            chunk: <文本内容>
            chunk_context: <上下文>

        请仅基于提供的信息块回答问题。你的回答应包括：
        1. 患者姓名（格式：患者姓名：<姓名>）
        2. 问题的答案
        3. 解释为何选择特定的引用
        4. 对你的答案的信心分数（0-100）

        请按以下格式回答：
        患者姓名：<患者姓名>
        答案：<你的答案>
        解释：<你选择特定引用的解释>
        信心：<你的信心分数>
        引用：<使用的块ID>

        问题：{question}

            {p}
        """

        contextual_response, _ = rag_call(contextual_prompt)

        # 对上下文感知 RAG 输出进行验证
        contextual_response = validate_patient_name(contextual_response, chunks, "上下文感知 RAG")

        logging.info("问题回答完成")
        return rag_response, contextual_response
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


def extract_patient_name(text):
    # 使用正则表达式匹配 "患者姓名：<姓名>" 格式
    match = re.search(r'患者姓名：([^\n]+)', text)
    if match:
        return match.group(1).strip()
    return "未知患者"

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
    return response

# 在文件末尾添加以下代码，确保在导入时就加载模型
get_model()

# ... [其他函数和类保持不变] ...