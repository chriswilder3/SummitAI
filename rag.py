import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from typing import TypedDict, List
from langgraph.graph import StateGraph, START

def create_rag(transcript_text):
    llm = ChatOpenAI(
        model = "gpt-4.1-mini",
        api_key= os.getenv("OPENAI_API_KEY")
    )

    # Embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    # split into docs
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = [ Document(page_content=doc) for doc in text_splitter.split_text(transcript_text)]

    # vector store
    vector_store = InMemoryVectorStore(embedding=embeddings)
    something = vector_store.add_documents(documents= docs)

    # Create a schema/model for the state of application
    class State(TypedDict):
        question : str
        context : List[str]
        answer : str

    def retrieve(state :State):
        """ This function fetches similar docs from vectorstore
            using the question inside passed state"""
        retrived_docs = vector_store.similarity_search(state['question'])
        # These retrived_docs act as context for llm further
        return {"context":retrived_docs}

    def generate(state : State):
        """ This function prepares the inputs to be passed to
            the llm, and invokes the chain """
        # There will be multiple docs in context we got from vector store
        # Turn them into single string
        docs_content ="\n\n".join([ doc.page_content for doc in state["context"]])
        
        # pull prompt from langchain hub for this format of rag
        prompt = hub.pull('rlm/rag-prompt')
        
        chain = prompt | llm

        # We need to pass only question, context to llm. It will
        # return response, which is passed to graph as  
        # a answer dict,so it can pack into State and passed around
        response = chain.invoke(
                {
                    "question":state["question"],
                    "context":docs_content,
            }
        )
        return {"answer":response.content}
    
    graph_builder = StateGraph(state_schema= State)
    graph_builder.add_sequence([retrieve,generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()
    return graph

# StateGraph is the constructor for creating a LangGraph state
#   graph. State is a custom class, usually a TypedDict or Pydantic
#   BaseModel, that defines the structure of the data that will be 
#   passed between the nodes in the graph

# add_sequence takes a list of nodes which are runnables/functions
#   that will be run in the order provided

# add_edge(START, "retrieve"), add_edge is used to create 
#   a directed edge(a connection) between two nodes. Here it 
#   defines the start point, that is retrieve fun.

# compile() performs basic checks on the graph,
#   returns a ready-to-use runnable object

# Note that graph takes in the question dict from userinput(invoke)
# automatically handles packing it into State and passing around
# the nodes of the graph.


def main():
    transcript_text = """
        Hello everyone. Today's topic for group discussion is Artificial Intelligence. 
        Artificial Intelligence  is the ability of a computer program to learn and think.
        Anything can be considered as Artificial  Intelligence if it involves a program 
        doing something that we would normally rely on our  intelligence.  I am here to talk about the advantages of Artificial Intelligence. One of the biggest  advantages is that it reduces human error. Human error is when humans make mistakes like  usually and it happens all the time. So, computers however do not make all these mistakes.  With Artificial Intelligence, the decisions are taken with the previously gathered information  and it uses a set of algorithms. So, errors are reduced and chances of reaching accuracy  is high.  Yes, Hema, I agree with you. But isn't this AI making the humans so lazy with its applications?  Most of its applications are automated. Majority of its work is being automated. So, if humans  get addicted to this type of inventions, this might cause a lot of problems to the future  generations like they might become so lazy.  It is not making people lazy. Instead, people will be still working on their creativity  part. For the decisions where they have to think and make something, people will be working  on that. Whereas, for the jobs which are repetitive 
        like sending mails and making documents and  in the documents where the error should be reduced. For those particular things, the  robots or the AI machines will be working. This will reduce people from working on their  boring parts and instead use their mind and energy on the creative part.  Yeah, Ayesha has said that I think for the repetitive jobs and the boring jobs you are  talking about, I don't think there is any reason to have robots which are made of higher  cost and also it requires maintenance of these robots. So, for these repetitive jobs, I don't  think you need to put this much price and you need to get a robot to do this silly job
    """
    with open('gate_elig_sample.txt',"r") as f:
        text2 = f.read()
    print(text2)        
    graph = create_rag(text2)

    while True:
        user_input = input("Enter Question : ")
        if user_input == "exit":
            break
        response = graph.invoke({"question":user_input})
        print(response["answer"])

if __name__ =="__main__":
    main()


