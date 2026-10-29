## InfoTech Internal Assistant – RAG Streamlit UI
This project is a RAG chatbot built using Python, Streamlit, ChromaDB, and OpenAI.

## Features:
- Answers questions about company documents.
- Retrieves relevant text chunks using ChromaDB.
- Generates answers using OpenAI gpt-4o-mini.
- Displays source filenames and retrieved chunks.
- Maintains chat history during the session.

## It uses:
- Python – Application logic
- Streamlit – User interface
- ChromaDB – Vector database and document retrieval
- OpenAI GPT-4o-mini – Answer generation
- python-dotenv – Loading API keys from `.env`
- RAG – Retrieving relevant document chunks to generate context-based answers

## Setup:
1. Install dependencies: pip install -r requirements.txt
2. Add your OPENAI_API_KEY to .env.
3. Place the required .txt files in the data/ folder.
4. Run: streamlit run RAG-Streamlit.py

## RAG Workflow:
Document Loading → Chunking → ChromaDB Retrieval → Context Preparation → LLM Answer Generation → Display Answer and Sources.

## Limitations:
- Supports predefined text documents only.
- Retrieves up to three chunks per question.
- Uses in-memory ChromaDB by default.
- Does not use FastAPI or LangChain.