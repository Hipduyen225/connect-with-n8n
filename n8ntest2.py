import openai
import pinecone
import uuid
import numpy as np
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone  # Updated import

# Cấu hình OpenAI và Pinecone
openai.api_key = "YOUR_OPENAI_API_KEY"  # Đảm bảo rằng bạn đã có API key của OpenAI

# Create Pinecone instance
pc = Pinecone(api_key="YOUR_PINECONE_API_KEY")  # Replace pinecone.init() with this line

# Kết nối với Pinecone
index_name = "cvdataset"  # Tên chỉ mục Pinecone của bạn
index = pc.Index(index_name)  # Use pc.Index() instead of directly initializing

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

# Main function để xử lý tất cả các bước
def main():
    # Ví dụ về việc upload file từ người dùng
    st.set_page_config(page_title="CV Recruitment AI", page_icon="💼")
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

if __name__ == "__main__":
    main()
