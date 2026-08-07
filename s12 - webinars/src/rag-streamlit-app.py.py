import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain import hub
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict

# API keys
env_path = Path(".") / "rag.env"
load_dotenv(dotenv_path=env_path, override=True)

# Chat model
llm = init_chat_model(
    "chatmodel",
    model_provider="openai",
    base_url=f"http://{os.environ['CHAT_SERVICE_URL']}/openai/v1/",
)

# Embedding model
hf_embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.environ["HF_TOKEN"],
    api_url=f"http://{os.environ['EMBEDDING_SERVICE_URL']}/v1/models/embmodel:predict",
)


class CEmbeddings(HuggingFaceInferenceAPIEmbeddings):
    def embed_documents(self, docs):
        return hf_embeddings.embed_documents(docs)["predictions"]


custom_embeddings = CEmbeddings(api_key=os.environ["HF_TOKEN"])

# Vector store
vector_store = PGVector(
    embeddings=custom_embeddings,
    collection_name="my_docs",
    connection=os.environ["PG_CONN_URL"],
)

# Define prompt and operations
prompt = hub.pull("rlm/rag-prompt")


class State(TypedDict):
    question: str
    context: list[Document]
    answer: str


def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Define graph of operations
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Streamlit setup
st.title("RAG App")
st.write("Welcome to the RAG (Retrieval-Augmented Generation) app.")
st.write("Please provide a link to the document to retrieve and your question.")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

qa = st.container()

with st.form("rag_form", clear_on_submit=True):
    rag_document = st.text_input("Document", "")
    question = st.text_input("Question", "")
    submit = st.form_submit_button("Submit")

if submit:
    # Load and chunk contents
    if question:
        if rag_document:
            loader = WebBaseLoader(
                web_paths=(rag_document,),
                bs_kwargs={},
            )
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            all_splits = text_splitter.split_documents(docs)

            # Index chunks
            _ = vector_store.add_documents(documents=all_splits)

        st.session_state.messages.append({"role": "user", "content": question})
        with qa.chat_message("user"):
            st.write(question)

        response = graph.invoke({"question": question})
        st.session_state.messages.append(
            {"role": "assistant", "content": response["answer"]}
        )
        with qa.chat_message("assistant"):
            st.write(response["answer"])
    else:
        with qa.chat_message("assistant"):
            st.write("You didn't provide a question!")
