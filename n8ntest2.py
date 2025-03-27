import openai
import pinecone
import uuid
import numpy as np
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from datetime import datetime
import faiss
import streamlit as st

# Cấu hình OpenAI và Pinecone
openai.api_key = "YOUR_OPENAI_API_KEY"  # Đảm bảo rằng bạn đã có API key của OpenAI

# Khởi tạo Pinecone với cách mới
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(
    api_key="YOUR_PINECONE_API_KEY"
)

# Kiểm tra nếu index không có sẵn, tạo mới
index_name = "cvdataset"  # Tên chỉ mục Pinecone của bạn
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Đảm bảo dimension phù hợp với embeddings của bạn
        metric="euclidean",  # Bạn cũng có thể dùng 'cosine' hoặc 'dotproduct'
        spec=ServerlessSpec(
            cloud="aws",
            region="us-west-2"
        )
    )

# Kết nối với Pinecone
index = pc[index_name]

# Tạo session ID duy nhất cho mỗi phiên làm việc
def generate_session_id():
    return str(uuid.uuid4())

# Đọc và trích xuất văn bản từ file PDF
def read_pdfs(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Chia nhỏ văn bản nếu quá dài (split)
def split_text(text, chunk_size=8000):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
    return text_splitter.split_text(text)

# Tạo embedding từ văn bản bằng OpenAI
def create_embedding_from_text(text):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",  # Hoặc mô hình embedding khác bạn muốn sử dụng
        input=text
    )
    embeddings = response['data'][0]['embedding']
    return np.array(embeddings)

# Chèn dữ liệu vào Pinecone
def upload_to_pinecone(file_name, content, embeddings):
    metadata = {
        "file_name": file_name,
        "content": content[:500]  # Trích xuất phần đầu của nội dung để làm metadata
    }

    # Tạo vector cho file và đưa vào Pinecone
    upsert_response = index.upsert(
        vectors=[(str(uuid.uuid4()), embeddings.tolist(), metadata)]
    )
    return upsert_response

# Sử dụng FAISS để tạo vector store
def create_faiss_index(embeddings_list):
    dimension = len(embeddings_list[0])
    index = faiss.IndexFlatL2(dimension)  # Using L2 (Euclidean) distance
    faiss_index = faiss.IndexIDMap(index)
    
    ids = np.array([i for i in range(len(embeddings_list))])
    embeddings = np.vstack(embeddings_list)
    faiss_index.add_with_ids(embeddings, ids)
    
    return faiss_index

# Lấy input người dùng và xử lý
def user_input(user_question, pdf_docs, conversation_history):
    if pdf_docs is None:
        st.warning("Please upload PDF files before processing.")
        return

    text_chunks = split_text(get_pdf_text(pdf_docs), model_name="OpenAI")
    embeddings_list = []
    for chunk in text_chunks:
        embeddings = create_embedding_from_text(chunk)
        embeddings_list.append(embeddings)

    # Chèn vào Pinecone
    for embeddings, chunk in zip(embeddings_list, text_chunks):
        response = upload_to_pinecone(file_name="Uploaded PDF", content=chunk, embeddings=embeddings)
        st.success(f"✅ File processed and uploaded to Pinecone successfully!")

    # Tạo FAISS index từ embeddings
    faiss_index = create_faiss_index(embeddings_list)

    # Trả về phản hồi cho câu hỏi người dùng
    question_embedding = create_embedding_from_text(user_question)
    D, I = faiss_index.search(np.array([question_embedding]), k=5)  # Lấy 5 câu trả lời gần nhất

    # Tạo câu trả lời dựa trên câu trả lời gần nhất
    relevant_docs = [text_chunks[i] for i in I[0]]
    answer = " ".join(relevant_docs)

    conversation_history.append((user_question, answer, "OpenAI", datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    
    st.write("Answer: ", answer)

# Hàm để lấy văn bản từ PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Hàm chính để hiển thị giao diện và xử lý câu hỏi người dùng
def main():
    st.set_page_config(page_title="CV Recruitment AI", page_icon="💼")
    st.header("💼 CV Recruitment AI Assistant")

    # Lưu trữ lịch sử cuộc trò chuyện
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    model_name = st.sidebar.radio("Select the Model:", ("OpenAI"))

    # Chọn API Key
    api_key = st.sidebar.text_input("Enter your OpenAI API Key:")
    if not api_key:
        st.sidebar.warning("Please enter your OpenAI API Key to proceed.")
        return

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    st.success("Done")
            else:
                st.warning("Please upload PDF files before processing.")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question, pdf_docs, st.session_state.conversation_history)

if __name__ == "__main__":
    main()