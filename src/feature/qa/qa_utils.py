import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain


load_dotenv()

# Set up API keys
openai_api_key = os.environ.get("OPENAI_API_KEY")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pincone_index = os.environ.get("PINECONE_INDEX")

class MovieQASystem:
    def __init__(self):
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.index_name = pincone_index
        self.vectorstore = PineconeVectorStore(index_name=self.index_name, embedding=embeddings)
        
        # Using VectorStoreRetriever
        self.retriever = self.vectorstore.as_retriever()
        
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo", # gpt-3.5-turbo,gpt-4o
            temperature=0,
            openai_api_key=openai_api_key
            )
    
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer or can't find it in the given context, just say "I'm sorry, but I couldn't find information about that in our movie history database."
        Do not use any other knowledge beyond what is provided in the context.
        {summaries}
        Question: {question}
        Answer:"""
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["summaries", "question"]
        )
        
        self.qa_chain = load_qa_with_sources_chain(llm, chain_type="stuff", prompt=PROMPT)
        

    def get_answer(self, query):
        docs = self.retriever.get_relevant_documents(query)
        if not docs:
            return "I'm sorry, but I couldn't find information about that in our movie history database."
        result = self.qa_chain.invoke({"input_documents": docs, "question": query}, return_only_outputs=True)
        return result["output_text"]