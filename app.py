import streamlit as st
import os
from workflow import parse_pdf, process_document, answer_question, get_model
import logging
import traceback
import lancedb
import pyarrow as pa
import re

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 确保模型在应用启动时就加载
get_model()

# 将 display_chunk 函数移到这里
def display_chunk(chunk_id, chunk_content):
    with st.expander(f"文本块 {chunk_id}"):
        st.text_area("内容", chunk_content, height=100, key=f"chunk_{chunk_id}")

# 添加 submit_question 函数
def submit_question():
    st.session_state.submit_question = True

st.set_page_config(page_title="文档问答系统", page_icon="📚", layout="wide")

st.title("文档问答系统")

# 确保 data 目录存在
if not os.path.exists("data"):
    os.makedirs("data")

# 添加一个函数来处理 data 目录下的所有 PDF 文件
def process_all_pdfs():
    pdf_files = [f for f in os.listdir("data") if f.endswith(".pdf")]
    for pdf_file in pdf_files:
        file_path = os.path.join("data", pdf_file)
        try:
            with st.spinner(f"正在处理文档: {pdf_file}"):
                doc = parse_pdf(file_path)
                process_document(doc)
            st.success(f"文档 {pdf_file} 处理完成!")
        except Exception as e:
            st.error(f"处理文档 {pdf_file} 时发生错误: {str(e)}")
            logging.error(f"处理文档 {pdf_file} 时发生错误: {str(e)}")
            logging.error(traceback.format_exc())

# 添加一个按钮来处理所有 PDF 文件
if st.button("处理 data 目录下的所有 PDF 文档"):
    process_all_pdfs()

# 文档上传部分（保留原有功能）
uploaded_file = st.file_uploader("上传新的PDF文档", type="pdf")

if uploaded_file is not None:
    try:
        # 保存上传的文件到 data 目录
        file_path = os.path.join("data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"文档已上传并保存为: {file_path}")
        
        if st.button("处理新上传的文档"):
            with st.spinner("正在处理文档..."):
                doc = parse_pdf(file_path)
                process_document(doc)
            st.success("文档处理完成!")
    except Exception as e:
        st.error(f"处理文档时发生错误: {str(e)}")
        logging.error(f"处理文档时发生错误: {str(e)}")
        logging.error(traceback.format_exc())

# 答部分
st.subheader("文档问答")

# 修改问题输入框，添加 on_change 参数
question = st.text_input("请输入您的问题 (按回车提交):", key="question_input", on_change=submit_question)

# 检查是否应该提交问题
if 'submit_question' not in st.session_state:
    st.session_state.submit_question = False

if st.session_state.submit_question and question:
    try:
        with st.spinner("正在生成答案..."):
            rag_response, contextual_response = answer_question(question)
        
        st.subheader("RAG 输出:")
        st.write(rag_response)
        
        # 检查是否有警告信息
        if "警告：" in rag_response:
            st.warning(rag_response.split("警告：")[-1])

        # 提取引用的块ID（从模型输出中）
        rag_citations = [int(id) for id in re.findall(r'\d+', rag_response.split('引用：')[-1])]
        
        # 显示引用的文本块
        if rag_citations:
            st.subheader("RAG 引用的文本块:")
            l_client = lancedb.connect("vectordb.lance")
            table = l_client.open_table("chunk-embeddings")
            chunks = table.to_arrow()
            logging.info(f"RAG chunks 类型: {type(chunks)}")
            logging.info(f"RAG chunks 列名: {chunks.column_names}")
            logging.info(f"RAG chunks 长度: {len(chunks)}")
            logging.info(f"RAG 引用的块ID: {rag_citations}")
            for i, chunk_id in enumerate(rag_citations):
                try:
                    chunk_content = chunks.column('chunk')[chunk_id].as_py()
                    display_chunk(f"{chunk_id}_{i}", chunk_content)  # 使用 chunk_id 和计数器 i 创建唯一键
                except IndexError:
                    st.warning(f"引用的文本块 {chunk_id} 不存在")
                    logging.warning(f"引用的文本块 {chunk_id} 超出范围 (0-{len(chunks)-1})")
                except Exception as e:
                    st.error(f"访问文本块 {chunk_id} 时发生错误: {str(e)}")
                    logging.error(f"访问文本块 {chunk_id} 时发生错误: {str(e)}")
                    logging.error(traceback.format_exc())
        
        st.subheader("上下文感知 RAG 输出:")
        st.write(contextual_response)
        
        # 提取引用的块ID（从模型输出中）
        contextual_citations = [int(id) for id in re.findall(r'\d+', contextual_response.split('引用：')[-1])]
        
        # 显示引用的文本块（上下文感知 RAG）
        if contextual_citations:
            st.subheader("上下文感知 RAG 引用的文本块:")
            l_client = lancedb.connect("vectordb.lance")
            table = l_client.open_table("contextual-chunk-embeddings")
            chunks = table.to_arrow()
            logging.info(f"上下文感知 RAG chunks 类型: {type(chunks)}")
            logging.info(f"上下文感知 RAG chunks 列名: {chunks.column_names}")
            logging.info(f"上下文感知 RAG chunks 长度: {len(chunks)}")
            logging.info(f"上下文感知 RAG 引用的块ID: {contextual_citations}")
            for i, chunk_id in enumerate(contextual_citations):
                try:
                    chunk_content = chunks.column('chunk')[chunk_id].as_py()
                    display_chunk(f"contextual_{chunk_id}_{i}", chunk_content)  # 使用 chunk_id 和计数器 i 创建唯一键
                except IndexError:
                    st.warning(f"引用的文本块 {chunk_id} 不存在")
                    logging.warning(f"引用的文本块 {chunk_id} 超出范围 (0-{len(chunks)-1})")
                except Exception as e:
                    st.error(f"访问文本块 {chunk_id} 时发生错误: {str(e)}")
                    logging.error(f"访问文本块 {chunk_id} 时发生错误: {str(e)}")
                    logging.error(traceback.format_exc())

        # 重置提交状态
        st.session_state.submit_question = False
    except Exception as e:
        st.error(f"回答问题时发生错误: {str(e)}")
        logging.error(f"回答问题时发生错误: {str(e)}")
        logging.error(traceback.format_exc())
elif st.session_state.submit_question and not question:
    st.warning("请输入一个问题。")
    st.session_state.submit_question = False

st.sidebar.header("使用说明")
st.sidebar.markdown("""
1. 如果需要添加新文档，请上传PDF文档并点击"处理文档"按钮
2. 直接在问题输入框中输入问题，按回车键提交
3. 查看生成的答案
4. 您可以继续提问，无需每次都上传文档
""")
