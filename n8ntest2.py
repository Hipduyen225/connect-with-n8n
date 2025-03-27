import streamlit as st
import requests
import uuid

# Webhook URL đúng (production) cho phần upload file
WEBHOOK_URL_PDFS = "https://n8n.khtt.online/webhook/getpdfs"  # URL Webhook cho PDF

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

def send_pdfs_to_n8n(files):
    """
    Gửi file PDF tới webhook n8n để xử lý và kiểm tra kết quả từ webhook.
    """
    webhook_url = WEBHOOK_URL_PDFS
    headers = HEADERS

    # Tạo payload dữ liệu của file để gửi đến webhook n8n
    files_payload = []
    for file in files:
        file_content = file.read()  # Đọc dữ liệu file
        files_payload.append({
            'name': file.name,
            'size': len(file_content),  # Kích thước file
            'content': file_content.decode('utf-8', 'ignore')  # Giải mã nội dung file thành chuỗi
        })
    
    # Hiển thị dữ liệu payload gửi đi để kiểm tra
    st.write("Dữ liệu gửi đi từ Streamlit: ")
    st.write(files_payload)

    try:
        response = requests.post(webhook_url, headers=headers, json={'files': files_payload})
        response.raise_for_status()  # Kiểm tra nếu có lỗi trong việc gửi
        return response.json()  # Trả về kết quả từ n8n (ví dụ: "Success")
    except Exception as e:
        st.error(f"Upload failed: {e}")
        return None

def main():
    """
    Ứng dụng chính của chatbot CV AI với phần upload nhiều file.
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

    # Phần sidebar để upload nhiều file (không ràng buộc vào PDF)
    with st.sidebar:
        st.title("Upload Files")
        uploaded_files = st.file_uploader("Upload your files", accept_multiple_files=True)

        if st.button("Submit & Process"):
            if uploaded_files:
                with st.spinner("Uploading files to n8n..."):
                    response = send_pdfs_to_n8n(uploaded_files)  # Gửi file tới n8n
                    if response:
                        st.success("Files uploaded and processed successfully.")
                        st.write(response)  # Hiển thị kết quả từ n8n (ví dụ: "Success")
            else:
                st.warning("Please upload at least one file.")

if __name__ == "__main__":
    main()
