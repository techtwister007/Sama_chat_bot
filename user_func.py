from sama import llm, rag

SYSTEM = ("You are Sama, an assistant for defensive security and software work. "
          "Answer using the provided context when relevant and cite sources by filename.")


def process_query(query):
    docs = rag.retrieve(query)
    context = "\n\n".join(f"[{d['source']}]\n{d['content']}" for d in docs)
    user = f"Context:\n{context}\n\nQuestion: {query}" if context else query
    answer = llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
    return {"answer": answer, "documents": docs}
