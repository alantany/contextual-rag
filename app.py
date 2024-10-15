import streamlit as st
import os
from workflow import parse_pdf, process_document, answer_question, get_model
import logging
import traceback
import lancedb
import pyarrow as pa
import re
from indexify.functions_sdk.graph import Graph

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
                st.text(f"开始解析 PDF 文件: {pdf_file}")
                doc = parse_pdf(file_path)
                st.text(f"PDF 解析完成，开始处理文档内容")
                process_document(doc, pdf_file)
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
                process_document(doc, uploaded_file.name)
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
        st.markdown(f"**{rag_response['patient_name']}**")
        st.markdown(f"**{rag_response['answer']}**")
        st.markdown(f"**{rag_response['explanation']}**")
        st.markdown(f"**{rag_response['confidence']}**")
        
        # 创建一个可展开的部分来显示引用
        with st.expander("引用"):
            for citation in rag_response['citations']:
                st.write(f"Chunk ID: {citation['chunk_id']}")
                st.write(citation['content'])
                st.write("---")

        st.subheader("上下文感知 RAG 输出:")
        st.markdown(f"**{contextual_response['patient_name']}**")
        st.markdown(f"**{contextual_response['answer']}**")
        st.markdown(f"**{contextual_response['explanation']}**")
        st.markdown(f"**{contextual_response['confidence']}**")
        
        # 创建一个可展开的部分来显示上下文感知引用
        with st.expander("上下文感知引用"):
            for citation in contextual_response['citations']:
                st.write(f"Chunk ID: {citation['chunk_id']}")
                st.write("Content:")
                st.write(citation['content'])
                st.write("Context:")
                st.write(citation['context'])
                st.write("---")

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

def validate_lancedb_data():
    l_client = lancedb.connect("vectordb.lance")
    chunks = l_client.open_table("chunk-embeddings").to_arrow()
    for i, chunk in enumerate(chunks):
        patient_name = extract_patient_name(chunk['chunk'].as_py())
        print(f"Chunk {i}: Patient: {patient_name}, Content: {chunk['chunk'].as_py()[:100]}...")  # 打印前100个字符

def process_document(doc, file_name):
    g: Graph = Graph("test", start_node=generate_chunk_contexts)
    g.add_edge(generate_chunk_contexts, TextEmbeddingExtractor)
    g.add_edge(TextEmbeddingExtractor, LanceDBWriter)
    g.run(block_until_done=True, doc=doc, file_name=file_name)
