import streamlit as st
import os

from langchain import hub
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import praw


reddit = praw.Reddit(
    client_id=os.environ["client_id"],
    client_secret=os.environ["client_secret"],
    user_agent=os.environ["user_agent"],
)


chromadb.api.client.SharedSystemClient.clear_system_cache()
llm = ChatOpenAI(model="gpt-4o")

max_attempts = 3


def main():
    st.title("살까말까?")
    st.markdown(
        "<style>.stText > div {text-wrap: auto;}</style>", unsafe_allow_html=True
    )
    st.markdown("#### 알고 싶은 종목을 입력해주세요!")
    st.caption(
        "**안내:** 투자에 대한 책임은 본인에게 있습니다. 본 서비스는 참고용으로만 사용해주세요."
    )
    user_input = st.chat_input("알고 싶은 종목을 입력해주세요!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").text(user_input)

        system_message = """Get ticker name from user input. Just make output the simple ticker name. NOT TO BE VERBOSE PLEASE.
            If you can't find the ticker name from user input, answer coin ticker that relevant to user input like recently coins. PLEA
SE NOT TO BE ANSWER ONLY BTC."""

        ticker = llm.invoke(
            [SystemMessage(content=system_message), HumanMessage(content=user_input)]
        ).content
        print(f"🚀 : app.py:36: ticker={ticker}")

        subreddit = reddit.subreddit("all")
        results = subreddit.search(ticker, sort="relevance", limit=10)
        print("res:", results)

        res_text = ""
        for submission in results:
            res_text += f"Title: {submission.title}"
            res_text += f"URL: {submission.url}"
            res_text += f"Score: {submission.score}"
            res_text += f"Author: {submission.author}"
            res_text += "---"

        print(f"🚀 : app.py:87: res_text={res_text}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        docs = [
            Document(page_content=text) for text in text_splitter.split_text(res_text)
        ]
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

        user_prompt = prompt.invoke(
            {"context": format_docs(retrieved_docs), "question": user_input}
        )

        system_message_res = """Response only KOREAN. Answer the ticker is going grow or not.
            Don't be verbose. But answer why you think so. You can answer economic issue with no responsibility.
            Answer list of titles in communities that you think is relevant and make more issuable to user can be stimulated.
            Answer like realtime news not the dumb stuffy answer PLEASE. Make answer flexible to user input."""
        response = llm.invoke(
            [
                SystemMessage(content=system_message_res),
                *user_prompt.messages,
            ]
        )
        content = response.content

        st.session_state.messages.append({"role": "assistant", "content": content})

        st.chat_message("assistant").text(content)


if __name__ == "__main__":
    main()
