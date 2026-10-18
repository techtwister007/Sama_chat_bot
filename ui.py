import chainlit as cl

from sama import llm, rag
from user_func import process_query

HELP = ("Commands:\n- `upload` : add local files (pdf, docx, txt, code...)\n"
        "- `/github owner/repo [ref] [token]` : index a GitHub repo (token only for private repos)\n"
        "- `/model https://...trycloudflare.com` : point at the Kaggle model\n- `/folder path` : index a local folder\n- anything else : ask a question")


@cl.on_chat_start
async def start():
    await cl.Message(content="Hi, I'm Sama. Chat with your files and GitHub repos.\n\n" + HELP).send()


@cl.on_message
async def main(message: cl.Message):
    text = message.content.strip()

    if text == "upload":
        files = await cl.AskFileMessage(content="Upload files to index.", accept=["*/*"],
                                        max_files=20, max_size_mb=50).send()
        n = 0
        for f in files or []:
            with open(f.path, "rb") as fh:
                n += await cl.make_async(rag.add_file)(f.name, fh.read())
        await cl.Message(content=f"Indexed {n} chunks.").send()
    elif text.startswith("/github"):
        parts = text.split()[1:]
        if not parts:
            await cl.Message(content=HELP).send()
            return
        n = await cl.make_async(rag.add_github_repo)(*parts[:3])
        await cl.Message(content=f"Indexed {n} chunks from {parts[0]}.").send()
    elif text.startswith("/model"):
        llm.set_url(text[len("/model"):])
        await cl.Message(content="Model URL updated.").send()
    elif text.startswith("/folder"):
        n = await cl.make_async(rag.add_local_folder)(text[len("/folder"):].strip())
        await cl.Message(content=f"Indexed {n} chunks.").send()
    else:
        try:
            result = await cl.make_async(process_query)(text)
        except Exception as e:
            await cl.Message(content=f"Error: {e}").send()
            return
        sources = sorted({d["source"] for d in result["documents"]})
        suffix = "\n\n**Sources:** " + ", ".join(sources) if sources else ""
        await cl.Message(content=result["answer"] + suffix).send()
