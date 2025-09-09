import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
import os
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

def embed(url):
    service_url = os.environ["EMBEDDING_SERVICE_URL"]

    class CEmbeddings(OpenAIEmbeddings):
        def embed_documents(self, docs):
            client = OpenAI(api_key="ignored", base_url=f"{service_url}/v1")
            emb_arr = []
            for doc in docs:
                embs = client.embeddings.create(
                    input=doc,
                    model=os.environ["EMBEDDING_MODEL_NAME"]
                )
                emb_arr.append(embs.data[0].embedding)
            return emb_arr

    custom_embeddings = CEmbeddings(api_key="ignored")

    vector_store = PGVector(
        embeddings=custom_embeddings,
        collection_name="my_docs",
        connection=os.environ["PG_CONN_URL"],
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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    vector_store.add_documents(documents=all_splits)
