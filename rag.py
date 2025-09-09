import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


llm = ChatOpenAI(
    model = "gpt-4.1-mini",
    api_key= os.getenv("OPENAI_API_KEY")
)

# Embeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_text()