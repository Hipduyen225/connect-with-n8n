import streamlit as st
import openai
import pinecone
import uuid
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Cấu hình OpenAI và Pinecone
def configure_openai(api_key):
    openai.api_key = api_key

def configure_pinecone(api_key):
    pinecone.init(api_key=api_key, environment="us-west1-gcp")
    return pinecone.Index("cvdataset")

# Kết nối với Pinecone
index = None

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

# Kiểm tra tính hợp lệ của API Key
def validate_api_key(api_key):
    try:
        configure_openai(api_key)  # Kiểm tra kết nối với OpenAI
        openai.Completion.create(model="text-davinci-003", prompt="Test", max_tokens=5)  # Thử gọi API
        return True
    except Exception as e:
        st.error(f"Invalid API Key: {e}")
        return False

# Main function để xử lý tất cả các bước
def main():
    global index

    st.set_page_config(page_title="CV Recruitment AI", page_icon="💼")

    # Giao diện nhập API key
    api_key = st.sidebar.text_input("Enter your OpenAI API Key:")
    if api_key and validate_api_key(api_key):
        configure_openai(api_key)
        index = configure_pinecone(api_key)  # Kết nối Pinecone với API Key hợp lệ

        st.sidebar.success("API Key validated successfully!")

        # Sau khi xác thực API Key, cho phép người dùng tải lên các PDF
        uploaded_files = st.file_uploader("Upload PDF Files", accept_multiple_files=True)

        if uploaded_files:
            with st.spinner("Processing files..."):
                for file in uploaded_files:
                    # Đọc văn bản từ file PDF
                    raw_text = read_pdfs([file])

                    # Chia nhỏ văn bản nếu cần
                    text_chunks = split_text(raw_text)

                    # Tạo embeddings từ văn bản đã chia nhỏ
                    embeddings_list = []
                    for chunk in text_chunks:
                        embeddings = create_embedding_from_text(chunk)
                        embeddings_list.append(embeddings)

                    # Chèn vào Pinecone
                    for embeddings, chunk in zip(embeddings_list, text_chunks):
                        response = upload_to_pinecone(file.name, chunk, embeddings)
                        st.success(f"✅ File {file.name} processed and uploaded to Pinecone successfully!")
        else:
            st.warning("⚠️ Please upload at least one file.")
    else:
        st.sidebar.warning("Please enter a valid OpenAI API Key to proceed.")

if __name__ == "__main__":
    main()
