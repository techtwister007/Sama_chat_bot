import chainlit as cl
from user_func import process_query
import time


@cl.on_chat_start
async def start():
    """
    Initializes the bot when a new chat starts.

    This asynchronous function creates a new instance of the retrieval QA bot,
    sends a welcome message, and stores the bot instance in the user's session.
    """
    chain="chain" #buffer sentece
    welcome_message = cl.Message(content="Starting the bot...")
    await welcome_message.send()
    welcome_message.content = (
        "Hi, Welcome to Chat With Documents using Ollama (mistral model) and LangChain."
    )
    await welcome_message.update()
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):

    # Your custom logic goes here...
    query=message.content

    if query=="upload" :
        files = None
        # Wait for the user to upload a file
        while files == None:
            files = await cl.AskFileMessage(
                content="Please upload a text file to begin!", accept=["text/plain"]
            ).send()

        text_file = files[0]

        with open(text_file.path, "r", encoding="utf-8") as f:
            text = f.read()

        # Let the user know that the system is ready
        await cl.Message(
            content=f"`{text_file.name}` uploaded, it contains {len(text)} characters!"
        ).send()

        

        await cl.Message(
            content="done "
        ).send()


    else :
        result = process_query(query)
        answer = result["answer"]
        documents = result["documents"]

        # Send a response back to the user
        await cl.Message(
            content=answer,
        ).send()
