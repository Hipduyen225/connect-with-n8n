import streamlit as st
import requests
import uuid
from PyPDF2 import PdfReader  # Import thư viện xử lý PDF

# Webhook URL đúng (production) cho từng trigger
WEBHOOK_URL_PDFS = "https://n8n.khtt.online/webhook/getpdfs"  # URL Webhook cho PDF
WEBHOOK_URL_CHAT = "https://n8n.khtt.online/webhook/cvdataset"  # URL Webhook cho Chat

# Header Auth đúng với n8n
HEADERS = {
    "phuongduyen": "phuongduyentestcvdataset",  # Thay bằng đúng token bạn đã nhập ở n8n
}

def generate_session_id():
    """
    Generate a unique session ID using UUID.
    """
    return str(uuid.uuid4())

def send_message_to_llm(session_id, user_message):
    """
    Gửi tin nhắn người dùng đến webhook và nhận phản hồi từ AI.
    """
    try:
        payload = {
            "sessionId": session_id,
            "chatInput": user_message
        }

        # Gửi request với header đúng
        response = requests.post(WEBHOOK_URL_CHAT, json=payload, headers=HEADERS)
        response.raise_for_status()

        return response.json().get('output', 'No response received')

    except requests.RequestException as e:
        st.error(f"Error sending message to LLM: {e}")
        return "Sorry, there was an error processing your message."

def send_pdfs_to_n8n(files):
    """
    Gửi file PDF tới webhook n8n để xử lý và lưu vào Pinecone, mỗi file riêng biệt.
    """
    webhook_url = WEBHOOK_URL_PDFS
    files_payload = []

    # Tách mỗi file PDF thành một mục riêng biệt
    for file in files:
        file_content = file.read()  # Đọc dữ liệu file
        files_payload.append(
            ('files', (file.name, file_content, 'application/pdf'))  # Gửi dưới dạng PDF binary
        )

    try:
        # Gửi từng file PDF riêng biệt vào webhook
        response = requests.post(webhook_url, headers=HEADERS, files=files_payload)
        response.raise_for_status()  # Kiểm tra nếu có lỗi trong việc gửi
        return response.json()  # Trả về kết quả từ n8n (ví dụ: "Success")
    except requests.RequestException as e:
        st.error(f"Upload failed: {e}")
        return None

def read_pdfs(pdf_files):
    """
    Đọc nhiều file PDF và trả về văn bản từ các file đó.
    """
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def main():
    """
    Ứng dụng chính của chatbot CV AI với phần upload nhiều file PDF.
    """
    st.set_page_config(page_title="CV Recruitment AI", page_icon="💼")

    if 'session_id' not in st.session_state:
        st.session_state.session_id = generate_session_id()

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    st.title("💼 CV Recruitment AI Assistant")
    st.write("An AI assistant to help find the most suitable candidates for your job description.")

    # Hiển thị lịch sử chat
    for message in st.session_state.chat_history:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # Giai đoạn 2: Xử lý input của người dùng và gửi qua webhook chat
    if prompt := st.chat_input("Enter your job description or candidate search query"):
        st.session_state.chat_history.append({
            'role': 'user',
            'content': prompt
        })

        with st.chat_message('user'):
            st.markdown(prompt)

        with st.chat_message('assistant'):
            with st.spinner('Searching for matching candidates...'):
                llm_response = send_message_to_llm(
                    st.session_state.session_id,
                    prompt
                )
                st.markdown(llm_response)

        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': llm_response
        })

    # Phần sidebar để upload nhiều file PDF
    with st.sidebar:
        st.title("Upload PDFs")
        pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True, type='pdf')

        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Uploading PDFs to n8n..."):
                    response = send_pdfs_to_n8n(pdf_docs)  # Gửi file PDF qua n8n
                    if response:
                        # Hiển thị message từ phản hồi JSON của n8n nếu có
                        if response.get("status") == "success":
                            st.success(response.get("message", "Files uploaded and processed successfully."))
                        else:
                            st.warning("Received response, but no success message found.")
            else:
                st.warning("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()
