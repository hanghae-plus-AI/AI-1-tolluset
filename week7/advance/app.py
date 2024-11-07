import streamlit as st
import os
from datetime import datetime

from langchain import hub
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import praw
from uuid import uuid4
from dotenv import load_dotenv

load_dotenv()

reddit = praw.Reddit(
    client_id=os.environ["client_id"],
    client_secret=os.environ["client_secret"],
    user_agent=os.environ["user_agent"],
)


# chromadb.api.client.SharedSystemClient.clear_system_cache()

llm = ChatOpenAI(model="gpt-4o")

max_attempts = 3


def main():
    st.title(
        "살까말까? :chart_with_upwards_trend: :chart_with_downwards_trend: :thinking_face: "
    )
    st.markdown(
        """<style>
        .stText > div {text-wrap: auto;}
        [aria-label="Chat message from user"] {
            text-align: right !important;
        }
        [data-testid="stText"] div {
            width: 100%;
        }
        [data-testid="stChatMessageAvatarUser"] {
            display: none;
        }
        .katex .mathnormal {
            font-family: sans-serif;
            font-style: normal;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### 알고 싶은 종목을 입력해주세요!")
    st.caption(
        "**안내:** 투자에 대한 책임은 본인에게 있습니다. 본 서비스는 참고용으로만 사용해주세요."
    )
    user_input = st.chat_input("알고 싶은 종목을 입력해주세요!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(
            message["role"], avatar="🤔" if message["role"] == "assistant" else None
        ):
            st.markdown(message["content"])

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").text(user_input)

        today = datetime.now().strftime("%y-%m-%d")

        system_message = f"""Get ticker name from user input. Make search keyword for got answer to searching communities.
            Make only keyword that you made it. NOT TO BE VERBOSE PLEASE.
            Don't use old data, today is {today}.
            If you can't find the ticker name from user input, answer the user instruction.
            Thek keyword should be ENGLISH for more better searching.
            PLEASE NOT TO BE ANSWER ONLY BTC."""

        ticker = llm.invoke(
            [SystemMessage(content=system_message), HumanMessage(content=user_input)]
        ).content
        print(f"🚀 : app.py:36: ticker={ticker}")

        subreddit = reddit.subreddit("all")
        results = subreddit.search(ticker, sort="hot", time_filter="month", limit=8)

        res_text = ""
        for submission in results:
            res_text += f"Title: {submission.title}  "
            res_text += f"CreatedAt: {submission.created_utc}  "
            res_text += f"URL: {submission.url}  "
            res_text += f"Score: {submission.score}  "
            res_text += f"Author: {submission.author}  "
            res_text += "---\n\n"

        print(f"🚀 : app.py:87: res_text={res_text}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        docs = [
            Document(page_content=text) for text in text_splitter.split_text(res_text)
        ]
        splits = text_splitter.split_documents(docs)
        print(f"🚀 : app.py:109: splits={splits}")

        vs = Chroma(
            persist_directory="chroma",
            collection_name="reddit",
            embedding_function=OpenAIEmbeddings(),
        )

        vs.add_documents(
            ids=([str(uuid4()) for _ in range(len(splits))]),
            documents=splits,
            embedding_function=OpenAIEmbeddings(),
        )

        rt = vs.as_retriever()

        docs = rt.invoke(user_input)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        prompt = hub.pull("rlm/rag-prompt")

        user_prompt = prompt.invoke(
            {"context": format_docs(docs), "question": user_input}
        )

        system_message_res = """Response only KOREAN. Answer the ticker is going grow or not.
            Don't be verbose. But answer why you think so. You can answer economic issue with no responsibility.
            Answer list of titles in communities that you think is relevant and make more issuable to user can be stimulated.
            Answer like realtime news not the boring answer PLEASE. Make answer flexible to user input. Translate titles to KOREAN too.
            Add url for titles by follow markdown URL format like [TEXT](URL).
            """

        res = llm.stream(
            [
                SystemMessage(content=system_message_res),
                *user_prompt.messages,
            ],
        )

        with st.chat_message("assistant", avatar="🤔"):
            response = st.write_stream(res)

        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
