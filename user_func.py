from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
import chromadb
import os
import time

model = os.environ.get("MODEL", "mistral-instruct")
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

from constants import CHROMA_SETTINGS

def initialize_qa():
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    llm = Ollama(model=model)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa

def get_answer_and_documents(qa, query):
    start = time.time()
    res = qa(query)
    answer, docs = res['result'], res['source_documents']
    end = time.time()
    
    relevant_docs = []
    for document in docs:
        relevant_docs.append({
            "source": document.metadata["source"],
            "content": document.page_content
        })

    return answer, relevant_docs

def process_query(query):
    #qa = initialize_qa()
    #answer, docs = get_answer_and_documents(qa, query)
    
    # result = {
    #     "answer": answer,
    #     "documents": docs
    # }


    result = {
        "answer": "This is ans",
        "documents": "this is doc"
    }



    
    return result

