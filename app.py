import os
from dotenv import load_dotenv, find_dotenv

from langchain_cohere import CohereEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter

from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferMemory

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from langchain_groq import ChatGroq
from langchain import hub

load_dotenv(find_dotenv())

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


index_name = "rag-demo"
embeddins = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=COHERE_API_KEY
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)


print("Carregando os documentos ... ")
PATH_FILE = "data\\2210.03629v3.pdf"
loader = PyPDFLoader(PATH_FILE)
document = loader.load()

print("splitting the documments into small chunks ... \n\n")

text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0
)
docs = text_splitter.split_documents(document)
print(f"total chunks: {len(docs)}\n")

vectorstore_from_docs = PineconeVectorStore.from_documents(
    docs,
    index_name=index_name,
    embedding=embeddins,
)

vectorstore = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddins,
    pinecone_api_key=PINECONE_API_KEY
)

#query = "KNOWLEDGE-INTENSIVE REASONING TASKS"
#vectorstore.similarity_search(query)

llm = ChatGroq(
    model="Gemma2-9b-It",
    groq_api_key=GROQ_API_KEY,
    temperature=0.1
)

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
combine_docs_chain = create_stuff_documents_chain(
    llm, retrieval_qa_chat_prompt
)

qa = create_retrieval_chain(
    vectorstore.as_retriever(),
    combine_docs_chain
)

res = qa.invoke({"input":"what is a llm?"})
print(res['answer'])