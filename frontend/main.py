import chainlit as cl
from initial import Chatbot

chatbot = Chatbot()


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="Connected to Chainlit!").send()


@cl.on_message
async def on_message(message: cl.Message):
    prompt = message.content
    # answer = cl.Message(content="")

    # Initialize loader
    await answer.send()

    # Retrieve answer from chatbot
    cb = cl.AsyncLangchainCallbackHandler(stream_final_answer=True, force_stream_final_answer=True, answer_prefix_tokens=["Final", "Answer"])
    cb.answer_reached = True
    for chunk in await cl.make_async(chatbot.qa_chain.stream)(prompt, config=RunnableConfig(callbacks=[cb2])):
        await answer.stream_token(chunk)

    # Send back answer from chatbot
    await answer.send()