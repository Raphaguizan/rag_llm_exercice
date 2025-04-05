import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv

from langchain_cohere import CohereEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferMemory

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from langchain_groq import ChatGroq
from langchain import hub

load_dotenv(find_dotenv())

# Chaves de API
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Config Streamlit
st.set_page_config(page_title="RAG Demo", layout="wide")
st.title("📄🔍 RAG com PDF + Cohere + Groq")

# Upload do PDF
uploaded_file = st.file_uploader("Faça upload de um PDF", type="pdf")

if uploaded_file:
    with st.spinner("Lendo o documento..."):
        file_path = f"temp/{uploaded_file.name}"
        os.makedirs("temp", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader(file_path)
        document = loader.load()

        # Split
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(document)

        st.success(f"{len(docs)} chunks gerados!")

        # Embeddings + VectorStore
        embeddins = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=COHERE_API_KEY)
        index_name = "rag-demo"

        PineconeVectorStore.from_documents(
            docs,
            index_name=index_name,
            embedding=embeddins,
        )

        vectorstore = PineconeVectorStore(
            index_name=index_name,
            embedding=embeddins,
            pinecone_api_key=PINECONE_API_KEY,
        )

        # LLM
        llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=GROQ_API_KEY, temperature=0.1)

        # Prompt e cadeia
        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

        qa_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

        # Input do usuário
        user_question = st.text_input("Digite sua pergunta sobre o PDF:", value="O que é um LLM?")
        if user_question:
            with st.spinner("Consultando o modelo..."):
                result = qa_chain.invoke({"input": user_question})
                st.markdown("### 🧠 Resposta:")
                st.success(result["answer"])


# como rodar
# streamlit run app.py