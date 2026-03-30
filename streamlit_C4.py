import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

st.title("PoC: Kiểm tra Logic Prompt Engineering")

context = st.text_area(
    "Ngữ cảnh giả định (Context):", 
    "Sinh viên ngành CNTT của SGU cần đạt điểm TOEIC 450 để đủ điều kiện xét tốt nghiệp."
)

question = st.text_input("Sinh viên hỏi:", "Chuẩn đầu ra tiếng Anh là bao nhiêu?")

if st.button("Gửi cho AI"):
    # Lấy API key từ file .env
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key or api_key == "AIzaSyCwMUrykaEcPkXlsO9nTfvm4Z1QmkKj2vk":
        st.error("🚨 Chưa có API Key! Vui lòng mở file .env và điền GOOGLE_API_KEY vào nhé.")
    else:
        prompt = f"Ngữ cảnh trích xuất: {context}\nCâu hỏi: {question}\nChỉ trả lời dựa trên ngữ cảnh được cung cấp."
        
        try:
            with st.spinner("AI đang suy nghĩ..."):
                #  Gọi mô hình AI xử lý logic
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
                response = llm.invoke(prompt)
                
                st.success("✨ Kết quả AI Trả lời:")
                st.write(response.content)
        except Exception as e:
            st.error(f"❌ Lỗi khi gọi AI: {e}")