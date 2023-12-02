from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain import hub
from chromadb.config import Settings
import chromadb

class Chatbot:
    def __init__(self):
        self.persist_directory = "../chroma_db/"
        self.qa_chain = None
        self.init_rag()

    def init_rag(self):
        embedding = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en-v1.5")

        CHROMA_SETTINGS = Settings(
            persist_directory=self.persist_directory,
            anonymized_telemetry=False
        )

        client = chromadb.PersistentClient(path=self.persist_directory)
        vectordb = Chroma(client=client, persist_directory=self.persist_directory, embedding_function=embedding, client_settings=CHROMA_SETTINGS)

        QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")

        # LLM
        llm = Ollama(
            model="llama2",
            verbose=True,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            temperature=0
        )

        # QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vectordb.as_retriever(),
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        )

