# dialogue/manager.py
from typing import List
from .flows import greet_flow, answer_question_smart

class DialogueManager:
    def __init__(self, user_name: str):
        self.user_name = user_name
        self.history: List[str] = []

    def handle_input(self, user_input: str) -> str:
        """
        Xử lý input người dùng:
        - Nếu là chào hỏi → trả greeting.
        - Mọi câu hỏi khác → gọi answer_question_smart (quiz / troubleshoot / bài học / free question)
        """
        self.history.append(f"User: {user_input}")
        user_input_lower = user_input.lower()

        # Chào hỏi
        if any(word in user_input_lower for word in ["xin chào", "chào", "hi"]):
            response = greet_flow(self.user_name)
        else:
            # Smart answer
            response = answer_question_smart(user_input)

        self.history.append(f"Bot: {response}")
        return response

    def get_history(self) -> List[str]:
        return self.history
