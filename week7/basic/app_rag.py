import streamlit as st
import bs4
import os
from langchain import hub
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI


llm = ChatOpenAI(model="gpt-4o-mini")

st.title("ChatBot")
st.markdown("<style>.stText > div {text-wrap: auto;}</style>", unsafe_allow_html=True)
user_input = st.chat_input("Type your message...")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    loader = WebBaseLoader(
        web_paths=("https://spartacodingclub.kr/blog/all-in-challenge_winner",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("editedContent")
            )
        ),
    )
    docs = loader.load()
    

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=OpenAIEmbeddings(),
    )

    retriever = vectorstore.as_retriever()

    retrieved_docs = retriever.invoke(user_input)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    prompt = hub.pull("rlm/rag-prompt")

    user_prompt = prompt.invoke({"context": format_docs(retrieved_docs), "question": user_input})
    response = llm.invoke(user_prompt)
    content = response.content


    st.session_state.messages.append({"role": "assistant", "content": content})

    st.chat_message("user").text(user_input)
    st.chat_message("assistant").text(content)


