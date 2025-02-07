from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from bs4 import BeautifulSoup
import streamlit as st


# input
st.title("URLChatBot")
query = st.chat_input("Enter your query here")


with st.sidebar:
    llm_api_key = st.text_input("Enter your LLM API Key", type="password")
    st.write("Don't have an API Key? Get one from https://aistudio.google.com")
    st.html("<br>")
    url = st.text_input("Enter the URL you want to chat about")

if not llm_api_key:
    st.toast("Api key is required")
    st.stop()

if not url:
    st.chat_message("ai").write("Please enter a URL to chat about")
    st.stop()

# step-1 load doucment
loader = WebBaseLoader(url)
docs = loader.load()

# step-2 split text
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=15)
splits = splitter.split_documents(docs)


# embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=llm_api_key,
)


# vector store
vectorstore = FAISS.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever()


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=llm_api_key,
)


# prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the following question based on the provided context:
    <context>
    {context}
    </context>
    Query: {input}
    """
)

# chain
doc_chain = create_stuff_documents_chain(llm, prompt)
retriever_chain = create_retrieval_chain(retriever, doc_chain)
output_parser = StrOutputParser()


if "messages" not in st.session_state:
    st.session_state.messages = []

if query:
    with st.spinner("Thinking..."):
        response = retriever_chain.invoke({"input": query})
        st.session_state.messages.append({"role": "human", "content": query})
        st.session_state.messages.append({"role": "ai", "content": response["answer"]})
    for message in st.session_state.messages:
        st.chat_message(message["role"]).write(output_parser.parse(message["content"]))
