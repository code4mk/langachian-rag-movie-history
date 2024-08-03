import os
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

class MovieChatSystem:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        
        # Set up API keys
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        self.pinecone_env = os.environ.get("PINECONE_ENV")

        # Initialize LLM
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        # Prepare document embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.index_name = "movie-quiz"
        
        # Initialize Pinecone Vector Store
        self.vectorstore = PineconeVectorStore(index_name=self.index_name, embedding=embeddings)
        
        # Contextualize question
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.vectorstore.as_retriever(), contextualize_q_prompt
        )
        
        # Answer question
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        
        self.rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        # Statefully manage chat history
        self.store = {}
    
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]
    
    def get_answer(self, session_id: str, query: str):
        conversational_rag_chain = RunnableWithMessageHistory(
            self.rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        result = conversational_rag_chain.invoke({"input": query}, {"session_id": session_id})
        return result["answer"]
