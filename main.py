import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
        model = 'gpt-4.1-mini',
        api_key= os.getenv('OPENAI_API_KEY')
    )
