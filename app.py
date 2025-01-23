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

st.set_page_config(page_title="上下文增强RAG", page_icon="📚", layout="wide")

st.title("上下文增强RAG")

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

# 文档上传部分
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

# 问答部分
st.subheader("文档问答")

# 修改问题输入框，添加 on_change 参数
question = st.text_input("请输入您的问题 (按回车提交):", key="question_input", on_change=submit_question)

# 检查是否应该提交问题
if 'submit_question' not in st.session_state:
    st.session_state.submit_question = False

if st.session_state.submit_question and question:
    try:
        with st.spinner("正在生成答案..."):
            response = answer_question(question)
        
        st.subheader("RAG 输出:")
        st.markdown(f"**{response['patient_name']}**")
        st.markdown(f"**{response['answer']}**")
        st.markdown(f"**{response['explanation']}**")
        st.markdown(f"**{response['confidence']}**")
        
        # 创建一个可展开的部分来显示引用
        with st.expander("引用"):
            for citation in response['citations']:
                st.write(f"文本块 ID: {citation['chunk_id']}")
                st.write(f"内容: {citation['content']}")
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

st.sidebar.header("上下文增强RAG（Contextual Retrieval）的特点和优势")
st.sidebar.markdown("""
上下文增强RAG（Contextual Retrieval）是一种先进的检索增强生成技术，由 Anthropic 提出。它具有以下特点和优势：

1. 显著提高检索准确性：相比传统RAG方法，Contextual Retrieval可以减少49%的检索失败率。

2. 保留上下文信息：通过在编码时保留chunk的上下文信息，解决了传统RAG方法可能丢失重要上下文的问题。

3. 提高相关信息检索：通过结合语义嵌入和精确匹配（BM25），能更好地检索到相关信息，特别是对于包含唯一标识符或技术术语的查询。

4. 增强模型理解：通过提供更丰富的上下文，帮助模型更好地理解和回答复杂问题。

5. 适应性强：可以处理各种类型的文档和问题，适用于多种领域，如客户支持、法律分析等。

6. 可扩展性：能够处理大规模知识库，远超单个提示可以容纳的范围。

7. 实现简单：可以通过简单的提示工程来实现，无需复杂的模型训练。

8. 成本效益高：结合 Claude 的提示缓存功能，可以显著降低处理成本。

9. 与重排序结合效果更佳：当与重排序技术结合时，可以将检索失败率降低高达67%。
""")

# 在页面底部添加开发者信息
st.markdown("---")
st.markdown("开发者: Huaiyuan Tan")

def validate_lancedb_data():
    l_client = lancedb.connect("vectordb.lance")
    chunks = l_client.open_table("chunk-embeddings").to_arrow()
    for i, chunk in enumerate(chunks):
        patient_name = extract_patient_name(chunk['chunk'].as_py())
        print(f"Chunk {i}: Patient: {patient_name}, Content: {chunk['chunk'].as_py()[:100]}...")  # 打印前100个字符
