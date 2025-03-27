import streamlit as st
import requests
import uuid

# ✅ Webhook URLs
WEBHOOK_URL_PDFS = "https://n8n.khtt.online/webhook/getpdfs"     # Webhook dùng cho upload file
WEBHOOK_URL_CHAT = "https://n8n.khtt.online/webhook/cvdataset"   # Webhook dùng cho truy vấn chatbot

# ✅ Header Auth (chỉ dùng cho GET/POST JSON – KHÔNG dùng Content-Type khi gửi files binary)
HEADERS_CHAT = {
    "phuongduyen": "phuongduyentestcvdataset",
    "Content-Type": "application/json"
}

HEADERS_FILE = {
    "phuongduyen": "phuongduyentestcvdataset"  # KHÔNG thêm content-type, requests sẽ tự xử lý
}

def generate_session_id():
    """Tạo session ID duy nhất."""
    return str(uuid.uuid4())

def send_message_to_llm(session_id, user_message):
    """Gửi tin nhắn từ người dùng đến LLM thông qua webhook chat."""
    try:
        payload = {
            "sessionId": session_id,
            "chatInput": user_message
        }

        response = requests.post(WEBHOOK_URL_CHAT, json=payload, headers=HEADERS_CHAT)
        response.raise_for_status()
        return response.json().get('output', 'No response received')

    except requests.RequestException as e:
        st.error(f"Error sending message to LLM: {e}")
        return "Sorry, there was an error processing your message."

def send_files_to_n8n(uploaded_files):
    """Gửi nhiều file (PDF, DOCX, TXT...) tới webhook n8n dưới dạng binary multipart/form-data."""
    files_payload = []

    for file in uploaded_files:
        file_bytes = file.read()
        files_payload.append(
            ('files', (file.name, file_bytes, file.type or 'application/octet-stream'))
        )

    try:
        response = requests.post(WEBHOOK_URL_PDFS, headers=HEADERS_FILE, files=files_payload)
        response.raise_for_status()
        return response.text  # hoặc response.json() nếu n8n trả về JSON

    except requests.RequestException as e:
        st.error(f"Upload failed: {e}")
        return None

def main():
    st.set_page_config(page_title="CV Recruitment AI", page_icon="💼")

    # Tạo session ID và lưu chat history
    if 'session_id' not in st.session_state:
        st.session_state.session_id = generate_session_id()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    st.title("💼 CV Recruitment AI Assistant")
    st.write("An AI assistant to help find the most suitable candidates for your job description.")

    # Chatbox chính
    for msg in st.session_state.chat_history:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    if prompt := st.chat_input("Enter your job description or candidate search query"):
        st.session_state.chat_history.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.markdown(prompt)

        with st.chat_message('assistant'):
            with st.spinner("Searching for matching candidates..."):
                response = send_message_to_llm(st.session_state.session_id, prompt)
                st.markdown(response)
                st.session_state.chat_history.append({'role': 'assistant', 'content': response})

    # Sidebar: Upload nhiều file
    with st.sidebar:
        st.subheader("📎 Upload candidate CVs")
        uploaded_files = st.file_uploader(
            "Upload PDF/DOCX/TXT files",
            type=None,
            accept_multiple_files=True
        )

        if st.button("📤 Upload to n8n"):
            if uploaded_files:
                with st.spinner("Uploading files to n8n..."):
                    result = send_files_to_n8n(uploaded_files)
                    if result:
                        st.success("✅ Files uploaded and processed successfully!")
                        st.code(result, language='json')
            else:
                st.warning("⚠️ Please select at least one file.")

if __name__ == "__main__":
    main()
