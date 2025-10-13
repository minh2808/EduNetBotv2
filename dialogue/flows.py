# dialogue/flows.py
import json
from typing import Dict, List
from langchain_openai import ChatOpenAI

# --- LLM setup ---
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)  # temperature 0 → output ổn định

# --- Greet ---
def greet_flow(user_name: str) -> str:
    greetings = [
        f"Chào {user_name}, mình có thể giúp gì hôm nay?",
        "Xin chào! Bạn muốn học gì hôm nay?",
        "Hi! Mình có thể hướng dẫn bạn hoặc làm quiz."
    ]
    return greetings[0]

# --- Lesson ---
LESSONS = {
    "mang_may_tinh": "Mạng máy tính gồm 7 tầng, từ vật lý đến ứng dụng...",
    "tcp_ip": "TCP/IP là bộ giao thức chuẩn cho Internet...",
    "tang_giao_van": "Tầng giao vận (Transport Layer) đảm bảo truyền tin tin cậy, kiểm soát lỗi và quản lý luồng dữ liệu. Các giao thức phổ biến là TCP và UDP."
}

def lesson_flow(topic: str) -> str:
    return LESSONS.get(topic.lower(), "Mình chưa có thông tin về chủ đề này.")

# --- Quiz dynamic ---
def generate_quiz(topic: str, n: int = 5) -> List[Dict]:
    """
    Tạo n câu hỏi trắc nghiệm về topic, trả về dạng list dict JSON.
    """
    prompt = f"""
Bạn là trợ giảng Mạng máy tính.
Hãy tạo {n} câu hỏi trắc nghiệm về chủ đề '{topic}'.
Mỗi câu hỏi có:
- "question": nội dung câu hỏi
- "options": danh sách các lựa chọn
- "answer": đáp án đúng
Trả về **chỉ JSON**, dạng list các dict:
[
  {{"question": "...", "options": ["...","..."], "answer": "..."}},
  ...
]
KHÔNG trả thêm bất cứ text nào khác ngoài JSON.
"""
    try:
        result = llm.predict(prompt)
        quiz_list = json.loads(result)
        return quiz_list
    except Exception as e:
        print("LLM lỗi:", e)
        # fallback minimal nếu LLM lỗi
        return [
            {"question": f"Câu hỏi {i+1} về {topic}?", "options": ["A","B","C","D"], "answer":"A"}
            for i in range(n)
        ]

def quiz_flow_smart(topic: str) -> List[Dict]:
    return generate_quiz(topic, n=5)

# --- Troubleshoot dynamic ---
TROUBLESHOOT_SOLUTIONS = {
    "khong vao duoc wifi": "Bạn hãy kiểm tra mật khẩu, reset router, kiểm tra IP.",
    "ping khong duoc": "Kiểm tra IP, subnet mask, gateway, firewall.",
    "tcp_khong_hoat_dong": "Kiểm tra cổng TCP, firewall, và kết nối giữa client-server."
}

def troubleshoot_flow(issue: str) -> str:
    return TROUBLESHOOT_SOLUTIONS.get(issue.lower(), f"Mình chưa biết cách xử lý '{issue}', hãy hỏi giảng viên hoặc kỹ thuật viên.")

# --- Smart answer (tích hợp quiz / lesson / troubleshoot) ---
def answer_question_smart(user_question: str) -> str:
    uq_lower = user_question.lower()
    
    # Lesson
    if any(x in uq_lower for x in ["bài học", "học", "giải thích"]):
        topic = uq_lower.split()[-1]
        return lesson_flow(topic)
    
    # Quiz
    elif "quiz" in uq_lower or "câu hỏi" in uq_lower:
        topic = uq_lower.split()[-1] if len(uq_lower.split())>1 else "tang_giao_van"
        quiz = quiz_flow_smart(topic)
        res = ""
        for i, q in enumerate(quiz, start=1):
            res += f"Câu hỏi {i}: {q['question']}\nLựa chọn: {q['options']}\nĐáp án: {q['answer']}\n\n"
        return res
    
    # Troubleshoot
    elif any(x in uq_lower for x in ["lỗi", "không chạy", "vấn đề", "sai"]):
        return troubleshoot_flow(uq_lower)
    
    # Greet
    elif any(x in uq_lower for x in ["xin chào","chào","hi"]):
        return greet_flow("Bạn")
    
    # Không hiểu
    else:
        return "Xin lỗi, mình chưa hiểu. Bạn có thể hỏi về bài học, làm quiz, hoặc lỗi gặp phải."
