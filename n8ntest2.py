import openai
import pinecone
import uuid
import numpy as np
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings  # Sử dụng OpenAI embeddings
from langchain.chat_models import ChatOpenAI  # Sử dụng OpenAI Chat model
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from datetime import datetime
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
    return embeddings

# Chèn dữ liệu vào Pinecone
def upload_to_pinecone(file_name, content, embeddings):
    metadata = {
        "file_name": file_name,
        "content": content[:500]  # Trích xuất phần đầu của nội dung để làm metadata
    }

    # Tạo vector cho file và đưa vào Pinecone
    upsert_response = index.upsert(
        vectors=[(str(uuid.uuid4()), embeddings, metadata)]
    )
    return upsert_response

# Hàm lấy conversational chain từ OpenAI
def get_conversational_chain(model_name="OpenAI", vectorstore=None):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3, api_key="YOUR_OPENAI_API_KEY")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

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

    # Trả về phản hồi cho câu hỏi người dùng
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key="YOUR_OPENAI_API_KEY")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(vectorstore=new_db)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    conversation_history.append((user_question, response['output_text'], "OpenAI", datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    
    st.write("Answer: ", response['output_text'])

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
