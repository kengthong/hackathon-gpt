# %%
#  ██████  ██████  ███    ██ ███████ ██  ██████
# ██      ██    ██ ████   ██ ██      ██ ██
# ██      ██    ██ ██ ██  ██ █████   ██ ██   ███
# ██      ██    ██ ██  ██ ██ ██      ██ ██    ██
#  ██████  ██████  ██   ████ ██      ██  ██████
#
config = {
    "persist_directory": "../chroma_db/",
    "confluence_url": "https://confluence.<company_name>.com",
    "api_key": "<API_KEY>",
}

persist_directory = config.get("persist_directory", None)
confluence_url = config.get("confluenc  e_url", None)
api_key = config.get("api_key", None)

# %%
# ██       ██████   █████  ██████
# ██      ██    ██ ██   ██ ██   ██
# ██      ██    ██ ███████ ██   ██
# ██      ██    ██ ██   ██ ██   ██
# ███████  ██████  ██   ██ ██████
#
# Extract using llamaindex
from llama_hub.confluence import ConfluenceReader
import os

os.environ["CONFLUENCE_API_TOKEN"] = api_key
os.environ["TOKENIZERS_PARALLELISM"] = "true"

reader = ConfluenceReader(base_url=confluence_url)
# Clarity Known Issues
documents = reader.load_data(page_ids=['618704581'], include_children=True)
# ! 400 All Technical
# ! Depre because macro not found
# documents.extend(reader.load_data(page_ids=['434086627'], include_children=True))

# Transform to langchain format
documents = [
    doc.to_langchain_format()
    for doc in documents
]

# Extra using langchain Confluence Loader (BS4)
from langchain.document_loaders import ConfluenceLoader

# Extract the documents using Langchain Confluence Loader
loader = ConfluenceLoader(url=confluence_url, token=api_key)
document_langchain = loader.load(space_key='GLSPS', max_pages=99999999)

# Concatenate all documents together
documents.extend(document_langchain)

# %%
# ███████ ██████  ██      ██ ████████
# ██      ██   ██ ██      ██    ██
# ███████ ██████  ██      ██    ██
#      ██ ██      ██      ██    ██
# ███████ ██      ███████ ██    ██
#

# 2. Splitting doucments into text snippets
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ! Impl 1 for tokenization
# ! Performance not as good
# text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
# texts = text_splitter.split_documents(documents)
# text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=10, encoding_name="cl100k_base")  # This the encoding for text-embedding-ada-002
# texts = text_splitter.split_documents(texts)

# * Impl 2 for tokenization
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
texts = text_splitter.split_documents(documents)
# %%
# ███████ ███    ███ ██████  ███████ ██████  ██████  ██ ███    ██  ██████  ███████
# ██      ████  ████ ██   ██ ██      ██   ██ ██   ██ ██ ████   ██ ██       ██
# █████   ██ ████ ██ ██████  █████   ██   ██ ██   ██ ██ ██ ██  ██ ██   ███ ███████
# ██      ██  ██  ██ ██   ██ ██      ██   ██ ██   ██ ██ ██  ██ ██ ██    ██      ██
# ███████ ██      ██ ██████  ███████ ██████  ██████  ██ ██   ████  ██████  ███████
#

# 3. Persist and store embeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from chromadb.config import Settings
import chromadb

CHROMA_SETTINGS = Settings(
    persist_directory=persist_directory,
    anonymized_telemetry=False
)

# embedding = HuggingFaceEmbeddings()
embedding = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-base-en-v1.5")

client = chromadb.PersistentClient(path=persist_directory)
vectordb = Chroma(client=client, persist_directory=persist_directory, embedding_function=embedding, client_settings=CHROMA_SETTINGS)
# vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory) # Persist texts into Chroma DB

# %%
# ██████   █████   ██████
# ██   ██ ██   ██ ██
# ██████  ███████ ██   ███
# ██   ██ ██   ██ ██    ██
# ██   ██ ██   ██  ██████
#

# For Quick testing of scraped data
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

# RAG prompt
from langchain import hub
QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")

# LLM
llm = Ollama(
    model="llama2",
    verbose=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    temperature=0.2
)

# QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
)

# %%
question = "What happens when I encounter an error identifier is too long in the app?"
result = qa_chain({"query": question})
