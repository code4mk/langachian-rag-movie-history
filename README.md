# RAG Application

This is a Retrieval-Augmented Generation (RAG) application using Langchain, OpenAI, and Pinecone for movie history.

## Features
- Chat
- Question Answering (QA)

## Setup

1. **Create a virtual environment:**

    ```bash
    python -m venv venv
    ```

2. **Activate the virtual environment:**

    - On Windows:
        ```bash
        venv\Scripts\activate
        ```
    - On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables:**

    Create a `.env` file in the root directory of your project and add the following variables:

    ```plaintext
    OPENAI_API_KEY=""
    PINECONE_API_KEY=""
    PINECONE_INDEX="movie-history"
    FEATURE_NAME="chat" # "qa"
    ```

    -> `FEATURE_NAME` will be `chat|qa` . diff is chat will preserve history

## Load Data into Pinecone VectorDB

1. **Load the data into Pinecone:**

    ```bash
    python ./src/vector_store/load_data.py
    ```

## Run the Project

1. start project with gradio ui

    ```bash
    python app.py
    ```

* `127.0.0.1:7860`

