from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import streamlit as st
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

google_key = os.getenv("GOOGLE_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")


@st.cache_resource
def get_chain():
    # ingestion
    pdf = PyPDFLoader("attention.pdf").load()

    # Chunking
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(pdf)
    # print(chunks[:20])

    # embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=google_key
    )

    # vector db
    db = Chroma.from_documents(chunks, embeddings)

    # prompt template

    prompt = ChatPromptTemplate.from_template("""
        Only answer the given question based on the context. If it is out of scope, just say you don't know"
        Context : {Context}
        Question : {Question}
        """)

    llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_key)

    # retriever
    retriever = db.as_retriever()

    # chain
    chain = (
        {"Context": retriever, "Question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


chain = get_chain()
st.title("QnA Chatbot on Attention is all you need")
query = st.text_input("Enter what you'd like to know")

if query:
    res = chain.invoke(query)
    st.write(res)
