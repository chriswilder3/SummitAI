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

def main():
    input_audio_file = "E:/downloads/meeting_preprocessed.wav"
    
    # Step 1: Transcribe audio
    transcript_segments = transcribe_audio(input_audio_file)
    
    # Convert JSON transcript into plain text for summarization
    transcript_text = " ".join([seg["text"] for seg in transcript_segments])
    
    # Step 2: Summarize
    summary = summarize(llm, transcript_text)
    
    print("📌 Meeting Summary:\n")
    print(summary)

if __name__ == "__main__":
    main()
