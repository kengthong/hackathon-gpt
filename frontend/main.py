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
<<<<<<< Updated upstream
    #await answer.send()

    # Retrieve answer from chatbot
    cb = cl.AsyncLangchainCallbackHandler(stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"])
    cb.answer_reached = True
    result = await chatbot.qa_chain.acall(prompt, callbacks=[cb])

    # Send back answer from chatbot
    # answer.content = result['result']
    # await answer.send()
    await cl.Message(content=result['result']).send()

=======
    await answer.send()
    cb2 = cl.AsyncLangchainCallbackHandler(stream_final_answer=True, force_stream_final_answer=True, answer_prefix_tokens=["Final", "Answer"])
    cb2.answer_reached = True
    result = await cl.make_async(chatbot.qa_chain.invoke)(prompt, config=RunnableConfig(callbacks=[cb2]))
    for chunk in await cl.make_async(chatbot.qa_chain.stream)(prompt, config=RunnableConfig(callbacks=[cb2])):
        await answer.stream_token(chunk)

    # Send back answer from chatbot
    await answer.send()
>>>>>>> Stashed changes
