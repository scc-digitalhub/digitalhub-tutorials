import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
import os
import requests
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI


PG_USER = os.environ["DB_USERNAME"]
PG_PASS = os.environ["DB_PASSWORD"]
PG_HOST = os.environ["DB_HOST"]
PG_PORT = os.environ["DB_PORT"]
DB_NAME = os.environ["DB_DATABASE"]
ACCESS_TOKEN = os.environ["DHCORE_ACCESS_TOKEN"]


def process(input):
    print(f"process input {input.id}...")
    url = (
        os.environ["DHCORE_ENDPOINT"]
        + "/api/v1/-/"
        + input.project
        + "/"
        + input.kind
        + "s/"
        + input.id
        + "/files/download"
    )
    print(f"request download link for input {input.id} from {url}")
    res = requests.get(url, headers={"Authorization": f"Bearer {ACCESS_TOKEN}"})
    print(f"Download url status {res.status_code}")
    if res.status_code == 200:
        j = res.json()
        print(f"{j}")
        #if "url" in j:
            #return embed(j["url"])

    print("End.")


def embed(url):
    PG_CONN_URL = (
        f"postgresql+psycopg://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{DB_NAME}"
    )
    print(f"process url {url}...")
    embedding_service_url = os.environ["EMBEDDING_SERVICE_URL"]
    embedding_model_name = os.environ["EMBEDDING_MODEL_NAME"]
    
    class CEmbeddings(OpenAIEmbeddings):
        def embed_documents(self, docs):
            client = OpenAI(api_key="ignored", base_url=f"{embedding_service_url}/v1")
            emb_arr = []
            for doc in docs:
                #sanitize string: replace NUL with spaces
                d=doc.replace("\x00", "-")
                embs = client.embeddings.create(
                    input=d,
                    model=embedding_model_name
                )
                emb_arr.append(embs.data[0].embedding)
            return emb_arr

    custom_embeddings = CEmbeddings(api_key="ignore")

    vector_store = PGVector(
        embeddings=custom_embeddings,
        collection_name=f"{embedding_model_name}_docs",
        connection=PG_CONN_URL,
    )

    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content")
            )
        ),
    )    
    docs = loader.load()
    print(f"document loaded, generate embeddings...")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    print(f"store documents in vector db...")

    vector_store.add_documents(documents=all_splits)
    print("Done.")