import os
from pathlib import Path

import streamlit as st
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------
# 1. Page configuration
# -----------------------------------
st.set_page_config(
    page_title="InfoTech Internal Assistant",
    page_icon="🤖",
    layout="wide",
)

# -----------------------------------
# 2. Custom UI styling
# -----------------------------------
st.markdown("""
<style>
    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1 {
        font-size: 2.4rem !important;
        font-weight: 750 !important;
    }

    .subtitle {
        color: #555555;
        margin-bottom: 2rem;
    }

    .status-banner {
        background: #173b2b;
        color: #73e49a;
        border-radius: 8px;
        padding: 15px;
        margin: 24px 0;
    }

    .source-label {
        color: #777777;
        font-size: 0.85rem;
        margin: 8px 0 16px 0;
    }

    .chunk-box {
        border: 1px solid #cccccc;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0 16px 0;
    }

    /* Black text for chat messages and input */
    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] li,
    div[data-testid="stChatMessage"] span {
        color: black !important;
    }

    div[data-testid="stChatInput"] textarea {
        color: black !important;
        caret-color: black !important;
    }

    div[data-testid="stChatInput"] textarea::placeholder {
        color: black !important;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

st.title("InfoTech Internal Assistant")

st.markdown(
    '<p class="subtitle">'
    'Ask anything about company policies, products, or procedures.'
    '</p>',
    unsafe_allow_html=True,
)

# -----------------------------------
# 3. Initialize RAG components
# -----------------------------------
@st.cache_resource
def initialize_rag():

    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing. Configure it in .env "
            "or Streamlit Secrets."
        )

    openai_client = OpenAI(api_key=api_key)

    data_dir = Path(__file__).parent / "data"

    document_files = {
        "HR Policy": "company_hr_policy.txt",
        "Engineering": "engineering_standards.txt",
        "Onboarding": "onboarding_guide.txt",
        "Product": "product_knowledge_base.txt",
        "Security": "security_policy.txt",
    }

    chunks = []

    # Read and split documents
    for source_name, filename in document_files.items():

        file_path = data_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Document not found: {file_path}"
            )

        text = file_path.read_text(encoding="utf-8")

        paragraphs = text.strip().split("\n\n")

        for paragraph in paragraphs:
            paragraph = paragraph.strip()

            if len(paragraph) < 50:
                continue

            if paragraph.startswith("===="):
                continue

            chunks.append({
                "text": paragraph,
                "source": filename,
            })

    if not chunks:
        raise ValueError("No document chunks were found.")

    # Initialize ChromaDB
    chroma_client = chromadb.Client()

    collection = chroma_client.get_or_create_collection(
        name="company_docs"
    )

    # Index chunks if the collection is empty
    if collection.count() == 0:
        collection.add(
            ids=[
                f"chunk_{i}"
                for i in range(len(chunks))
            ],
            documents=[
                item["text"] for item in chunks
            ],
            metadatas=[
                {"source": item["source"]}
                for item in chunks
            ],
        )

    return openai_client, collection


# -----------------------------------
# 4. RAG retrieval
# -----------------------------------
def retrieve_documents(question, collection):

    total_chunks = collection.count()

    if total_chunks == 0:
        return []

    results = collection.query(
        query_texts=[question],
        n_results=min(3, total_chunks),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    retrieved_chunks = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i, document in enumerate(documents):

        metadata = metadatas[i] or {}

        retrieved_chunks.append({
            "text": document,
            "source": metadata.get(
                "source", "Unknown source"
            ),
            "distance": distances[i],
        })

    return retrieved_chunks


# -----------------------------------
# 5. Generate answer using retrieved context
# -----------------------------------
def generate_answer(question, retrieved_chunks, client):

    if not retrieved_chunks:
        return (
            "No relevant document chunks were found. "
            "Please try another question."
        )

    # Combine retrieved chunks into context
    context = "\n\n".join(
        f"Source: {chunk['source']}\n"
        f"{chunk['text']}"
        for chunk in retrieved_chunks
    )

    prompt = f"""
You are a helpful company assistant.

Answer the user's question using only the context below.

Instructions:
- Use the retrieved documents as your source of information.
- Do not invent company policies or facts.
- If the context does not contain the answer, say so clearly.
- Provide a clear and easy-to-understand answer.

Retrieved context:
{context}

User question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content or (
        "No answer was generated."
    )


# -----------------------------------
# 6. Load the application
# -----------------------------------
try:
    client, collection = initialize_rag()

    total_chunks = collection.count()

    st.markdown(
        f'<div class="status-banner">'
        f'✅ Ready — {total_chunks} document chunks indexed.'
        f'</div>',
        unsafe_allow_html=True,
    )

except Exception as error:
    st.error(f"Could not initialize RAG: {error}")
    st.stop()


# -----------------------------------
# 7. Display chat history
# -----------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for item in st.session_state.chat_history:

    with st.chat_message(item["role"]):
        st.markdown(item["content"])

        if item["role"] == "assistant":

            if item["chunks"]:

                sources = list(dict.fromkeys(
                    chunk["source"]
                    for chunk in item["chunks"]
                ))

                st.caption(
                    "Sources: " + ", ".join(sources)
                )

            with st.expander(
                "🔎 View retrieved document chunks"
            ):

                if not item["chunks"]:
                    st.write("No chunks were retrieved.")

                for index, chunk in enumerate(
                    item["chunks"], 1
                ):

                    st.markdown(
                        f"**Chunk {index} — "
                        f"{chunk['source']}**"
                    )

                    st.markdown(
                        '<div class="chunk-box">',
                        unsafe_allow_html=True,
                    )

                    st.write(chunk["text"])

                    st.markdown(
                        '</div>',
                        unsafe_allow_html=True,
                    )

                    if chunk["distance"] is not None:
                        st.caption(
                            "ChromaDB distance: "
                            f"{chunk['distance']:.4f}"
                        )


# -----------------------------------
# 8. User input and basic RAG pipeline
# -----------------------------------
question = st.chat_input(
    "Ask about company policies or documents..."
)

if question:

    st.session_state.chat_history.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "Retrieving documents and generating an answer..."
        ):

            try:
                # Step 1: Retrieve relevant chunks
                retrieved_chunks = retrieve_documents(
                    question, collection
                )

                # Step 2: Generate answer from retrieved context
                answer = generate_answer(
                    question,
                    retrieved_chunks,
                    client,
                )

                # Step 3: Display answer
                st.markdown(answer)

                if retrieved_chunks:
                    sources = list(dict.fromkeys(
                        chunk["source"]
                        for chunk in retrieved_chunks
                    ))

                    st.caption(
                        "Sources: " + ", ".join(sources)
                    )

                # Step 4: Display retrieved chunks
                with st.expander(
                    "🔎 View retrieved document chunks"
                ):

                    if not retrieved_chunks:
                        st.write("No chunks were retrieved.")

                    for index, chunk in enumerate(
                        retrieved_chunks, 1
                    ):

                        st.markdown(
                            f"**Chunk {index} — "
                            f"{chunk['source']}**"
                        )

                        st.write(chunk["text"])

                        if chunk["distance"] is not None:
                            st.caption(
                                "ChromaDB distance: "
                                f"{chunk['distance']:.4f}"
                            )

                # Save conversation
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "chunks": retrieved_chunks,
                })

            except Exception as error:
                st.error(f"Error: {error}")
