import json
import os

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGEngine, PGVectorStore
from langgraph.graph import START, StateGraph
from openai import OpenAI
from typing_extensions import TypedDict

PG_USER = os.environ["DB_USERNAME"]
PG_PASS = os.environ["DB_PASSWORD"]
PG_HOST = os.environ["DB_HOST"]
PG_PORT = os.environ["DB_PORT"]
DB_NAME = os.environ["DB_DATABASE"]
ACCESS_TOKEN = os.environ["DHCORE_ACCESS_TOKEN"]


class State(TypedDict):
    question: str
    context: list[Document]
    answer: str


def init(context):
    chat_model_name = os.environ["CHAT_MODEL_NAME"]
    chat_service_url = os.environ["CHAT_SERVICE_URL"]
    embedding_model_name = os.environ["EMBEDDING_MODEL_NAME"]
    embedding_service_url = os.environ["EMBEDDING_SERVICE_URL"]

    class CEmbeddings(OpenAIEmbeddings):
        async def aembed_documents(self, docs):
            client = OpenAI(api_key="ignored", base_url=embedding_service_url)
            emb_arr = []
            for doc in docs:
                # sanitize string: replace NUL with spaces
                d = doc.replace("\x00", "-")
                embs = client.embeddings.create(input=d, model=embedding_model_name)
                emb_arr.append(embs.data[0].embedding)
            return emb_arr

    custom_embeddings = CEmbeddings(api_key="ignored")

    PG_CONN_URL = (
        f"postgresql+psycopg://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{DB_NAME}"
    )

    pg_engine = PGEngine.from_connection_string(url=PG_CONN_URL)

    vector_store = PGVectorStore.create_sync(
        engine=pg_engine,
        table_name=f"{embedding_model_name}_docs",
        # schema_name=SCHEMA_NAME,
        embedding_service=custom_embeddings,
    )

    os.environ["OPENAI_API_KEY"] = "ignore"

    llm = init_chat_model(
        chat_model_name, model_provider="openai", base_url=chat_service_url
    )
    prompt = ChatPromptTemplate.from_template("""
    Answer the question based only on the provided context.
    Context:
    {context}
    Question:
    {question}
    Answer:
    """)

    def retrieve(state: State):
        # print('question', state["question"])
        # print(type(vector_store))
        try:
            retrieved_docs = vector_store.similarity_search(state["question"])
        except Exception as error:
            raise RuntimeError(
                f"Retrieval failed (embedding API: {embedding_service_url}): "
                f"{error}; cause: {error.__cause__!r}"
            ) from error
        return {"context": retrieved_docs}

    def generate(state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = prompt.invoke(
            {"question": state["question"], "context": docs_content}
        )
        try:
            response = llm.invoke(messages)
        except Exception as error:
            raise RuntimeError(
                f"Chat generation failed at {chat_service_url}: "
                f"{error}; cause: {error.__cause__!r}"
            ) from error
        return {"answer": response.content}

    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    context.graph = graph


def serve(context, event):
    context.logger.info(f"Received event: {event}")

    if event.method == "GET":
        return {"status": "ok"}

    if isinstance(event.body, (bytes, str)):
        body = json.loads(event.body)
    else:
        body = event.body

    if not isinstance(body, dict):
        raise ValueError("Expected a JSON object with a non-empty 'question' string")
    if not isinstance(body.get("question"), str) or not body["question"].strip():
        raise ValueError("Expected a JSON object with a non-empty 'question' string")

    question = body["question"]
    response = context.graph.invoke({"question": question})
    return {"answer": response["answer"]}
