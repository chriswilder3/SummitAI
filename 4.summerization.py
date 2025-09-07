from transcription_complete import transcribe_audio
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model = "gpt-4.1-mini",
    api_key= os.getenv('OPENAI_API_KEY')
)

from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

def summarize(llm, transcript):
    print("---Summerizing---")
    summary_prompt = PromptTemplate.from_template(
        template="""
            You are an AI meeting assistant. Summarize the following meeting transcript:
            {transcript}

            Provide:
            1. A concise summary of the discussion.
            2. Key decisions made.
            3. Action items (who needs to do what).
            """
    )
    chain = summary_prompt | llm
    response = chain.invoke({"transcript":transcript})
    return response.content


def map_reduce_summarize(llm, transcript_segments):

    
    # Step 1: Flatten transcript into grouped chunks
    transcript_text = " ".join([f"{seg['speaker']}: {seg['text']}"
                                 for seg in transcript_segments])
    
    # Step 2: Split into Documents (chunks of ~2000 chars)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 2000,
        chunk_overlap = 200,
        separators = ["\n\n", "\n", ".", " "]
    )
    docs = [ Document(page_content=chunk)  
                for chunk in text_splitter.split_text(transcript_text)
            ]
    
    map_prompt = """
        You are a meeting assistant. Summarize the following transcript chunk.

        Transcript:
        {text}

        Your summary should include:
        - Main discussion points
        - Speaker perspectives
        - Any implied disagreements or concerns

        Write at least 3 bullet points.
    """

    reduce_prompt = """
        Combine the following partial summaries into a final meeting summary.

        Partial summaries:
        {text}

        Write:
        1. A detailed summary.
        2. Key decisions made.
        3. Action items (with responsible person if available).
    """
    # Step 3: Load map-reduce summarization chain
    chain = load_summarize_chain(
            llm, chain_type="map_reduce",
            map_prompt= PromptTemplate(template=map_prompt, 
                                       input_variables=["text"]),
            combine_prompt= PromptTemplate(template=reduce_prompt,
                                        input_variables=["text"]),
            verbose=True)
    
    # Step 4.
    response = chain.invoke(docs)
    return response

def main():
    input_audio_file = "E:/downloads/meeting_preprocessed.wav"
    
    # Step 1: Transcribe audio
    transcript_segments = transcribe_audio(input_audio_file)
    
    # Convert JSON transcript into plain text for summarization
    transcript_text = " ".join([seg["text"] for seg in transcript_segments])
    
    # Step 2: Summarize
    # summary = summarize(llm, transcript_text)
    summary = map_reduce_summarize(llm, transcript_segments)

    print("📌 Meeting Summary:\n")
    print(summary['output_text'])

if __name__ == "__main__":
    main()
