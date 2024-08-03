from src.feature.chat.chat_utils import MovieChatSystem

movie_chat_system = MovieChatSystem()

def handle_chat(session_id, query):
    # Handle chat interaction
    response = movie_chat_system.get_answer(session_id, query)
    return response