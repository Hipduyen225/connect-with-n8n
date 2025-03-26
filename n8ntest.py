import streamlit as st
import requests
import uuid

# ✅ Webhook URL đúng (production)
WEBHOOK_URL = "https://n8n.khtt.online/webhook/cvdataset"

# ✅ Header Auth đúng với n8n: key là "phuongduyen", value bạn tự đặt khi tạo credential
HEADERS = {
    "phuongduyen": "phuongduyentestcvdataset",  # ❗ Thay bằng đúng token bạn đã nhập ở n8n
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

        # ✅ Gửi request với header đúng
        response = requests.post(WEBHOOK_URL, json=payload, headers=HEADERS)
        response.raise_for_status()

        return response.json().get('output', 'No response received')

    except requests.RequestException as e:
        st.error(f"Error sending message to LLM: {e}")
        return "Sorry, there was an error processing your message."

def main():
    """
    Ứng dụng chính của chatbot CV AI.
    """
    st.set_page_config(page_title="CV Recruitment AI", page_icon="💼")

    if 'session_id' not in st.session_state:
        st.session_state.session_id = generate_session_id()

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    st.title("💼 CV Recruitment AI Assistant")
    st.write("An AI assistant to help find the most suitable candidates for your job description.")

    for message in st.session_state.chat_history:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

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

if __name__ == "__main__":
    main()
