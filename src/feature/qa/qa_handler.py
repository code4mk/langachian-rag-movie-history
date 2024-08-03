from src.feature.qa.qa_utils import MovieQASystem

movie_qa_system = MovieQASystem()

def handle_qa(query):
    # Handle chat interaction
    response = movie_qa_system.get_answer(query)
    return response