import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
import os
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

from langchain import hub
from langchain_core.documents import Document
from typing_extensions import List, TypedDict

from langgraph.graph import START, StateGraph
from langchain.chat_models import init_chat_model


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def init(context):
    chat_model_name = os.environ["CHAT_MODEL_NAME"]
    chat_service_url = os.environ["CHAT_SERVICE_URL"]
    embedding_model_name = os.environ["EMBEDDING_MODEL_NAME"]
    embedding_service_url = os.environ["EMBEDDING_SERVICE_URL"]

    class CEmbeddings(OpenAIEmbeddings):
        def embed_documents(self, docs):
            client = OpenAI(api_key="ignored", base_url=f"{embedding_service_url}/v1")
            emb_arr = []
            for doc in docs:
                embs = client.embeddings.create(input=doc, model=embedding_model_name)
                emb_arr.append(embs.data[0].embedding)
            return emb_arr

    custom_embeddings = CEmbeddings(api_key="ignored")

    vector_store = PGVector(
        embeddings=custom_embeddings,
        collection_name="my_docs",
        connection=os.environ["PG_CONN_URL"],
    )

    os.environ["OPENAI_API_KEY"] = "ignore"

    llm = init_chat_model(
        chat_model_name, model_provider="openai", base_url=f"{chat_service_url}/v1/"
    )
    prompt = hub.pull("rlm/rag-prompt")

    def retrieve(state: State):
        retrieved_docs = vector_store.similarity_search(state["question"])
        return {"context": retrieved_docs}

    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke(
            {"question": state["question"], "context": docs_content}
        )
        response = llm.invoke(messages)
        return {"answer": response.content}

    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    setattr(context, "graph", graph)


def serve(context, event):
    graph = context.graph
    context.logger.info(f"Received event: {event}")

    if isinstance(event.body, bytes):
        body = json.loads(event.body)
    else:
        body = event.body

    question = body["question"]
    response = graph.invoke({"question": question})
    return {"answer": response["answer"]}
