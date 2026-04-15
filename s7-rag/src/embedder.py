from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_postgres import PGEngine, PGVectorStore
from bs4 import BeautifulSoup
from langchain_postgres import PGEngine
from langchain_postgres import PGVectorStore
import os
import requests
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
import re
from sqlalchemy.exc import ProgrammingError
import uuid
from transformers import AutoTokenizer

PG_USER = os.environ["DB_USERNAME"]
PG_PASS = os.environ["DB_PASSWORD"]
PG_HOST = os.environ["DB_HOST"]
PG_PORT = os.environ["DB_PORT"]
DB_NAME = os.environ["DB_DATABASE"]
ACCESS_TOKEN = os.environ["DHCORE_ACCESS_TOKEN"]

def process(input):
    return embed(input)
    print("End.")

tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5")

def truncate_to_512(text: str, max_tokens: int = 500) -> str:
    encoded = tokenizer(
        text,
        max_length=max_tokens,
        truncation=True,
        add_special_tokens=False
    )

    return tokenizer.decode(encoded["input_ids"])
    
def embed(html_doc_obj):
    
    url = html_doc_obj.id
    html_doc_path = html_doc_obj.download(overwrite=True)
    
    CONNECTION_STRING = (f"postgresql+psycopg://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{DB_NAME}")
    engine = PGEngine.from_connection_string(url=CONNECTION_STRING)
    
    # Replace the vector size with your own vector size
    VECTOR_SIZE = 768

    print(f"process document id {url}")
    embedding_service_url = os.environ["EMBEDDING_SERVICE_URL"]
    embedding_model_name = os.environ["EMBEDDING_MODEL_NAME"] 
    

    TABLE_NAME = f"{embedding_model_name}_docs"

    try: 
        engine.init_vectorstore_table(
            table_name=TABLE_NAME,
            vector_size=VECTOR_SIZE)
    except ProgrammingError:
        print("Table already exists. Skipping creation.")
    
    class CEmbeddings(OpenAIEmbeddings):
        async def aembed_documents(self, docs):
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

    with open(html_doc_path) as f: html_content = f.read()
    soup = BeautifulSoup(html_content, 'html.parser')
  
    tot_text = soup.getText()
    tot_text = re.sub(r'https\S+', '', tot_text)
    tot_text = tot_text.replace("\x00", "-")
    tot_text = tot_text.strip()
  
    text_input = truncate_to_512(tot_text, max_tokens=500)

    print(len(tokenizer.encode(text_input)))
    
    docs = [
        Document(
            id=str(uuid.uuid4()),
            page_content=text_input,
            # metadata={"description": "red", "content": "1", "category": "fruit"},
        )
    ]

    custom_embeddings = CEmbeddings(api_key="ignore")
    
    store = PGVectorStore.create_sync(
        engine=engine,
        table_name=TABLE_NAME,
        embedding_service=custom_embeddings)

    store.add_documents(documents=docs)
    
    print("Done.")
    
    
    