import streamlit as st
import requests
import uuid
from PyPDF2 import PdfReader  # Import thư viện xử lý PDF

# Webhook URL đúng (production)
WEBHOOK_URL = "https://n8n.khtt.online/webhook/cvdataset"

# Header Auth đúng với n8n
HEADERS = {
    "phuongduyen": "phuongduyentestcvdataset",  # Thay bằng đúng token bạn đã nhập ở n8n
    "Content-Type": "application/json"
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
        response = requests.post(WEBHOOK_URL, json=payload, headers=HEADERS)
        response.raise_for_status()

        return response.json().get('output', 'No response received')

    except requests.RequestException as e:
        st.error(f"Error sending message to LLM: {e}")
        return "Sorry, there was an error processing your message."

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

    # Xử lý input của người dùng
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
                with st.spinner("Processing PDFs..."):
                    raw_text = read_pdfs(pdf_docs)  # Đọc và kết hợp văn bản từ nhiều file PDF
                    st.success("Processing Complete!")
                    st.write("Extracted text from PDFs:")
                    st.write(raw_text[:500])  # Hiển thị một phần văn bản đã được trích xuất
            else:
                st.warning("Please upload some PDF files.")

if __name__ == "__main__":
    main()
