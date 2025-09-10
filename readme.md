# SummitAI

> **From Meetings to Meaning — AI that Transcribes, Summarizes, and Surfaces Insights.**

SummitAI is an **AI-powered meeting companion** that transforms raw meeting recordings into **structured knowledge**.  
Users upload a video file, and SummitAI automatically extracts the audio, transcribes speech using Whisper, and applies LLM-powered pipelines for **summarization, question answering, and retrieval-augmented Q&A chat**.  

---

## ✨ Features

### 1. Video & Audio Processing
  - Uploads **MP4 video** via Streamlit frontend.
  - Load Video Segments using `moviepy`.  
  - Extracts audio (`ffmpeg`).  
  - Preprocesses audio (mono, **16kHz mono WAV**)     (Whisper-ready)  
  - Supports chunked transcription to avoid long input issues.   

### 2. Transcription
  - **Speech-to-Text**  
    - Uses **OpenAI Whisper API** for accurate transcripts.
    - Returns structured transcript with:  
      ```json
      { "start": float, "end": float, "text": str }    
  - **Text Cleaning & Structuring**  
    - Removing filler words.  

### 3. Meeting Summarization
- Implemented **Map-Reduce Summarizations**
  - map_prompt: chunk-wise summaries (main points + perspectives).
  - reduce_prompt: combines into *Decisions*, *Action Items*, *Open Questions*.
  - Output is concise, structured, and useful for meeting follow-ups.

### 4. RAG-based QA 
  - Chunk transcripts into vector DB. For developement, stored in InMemoryVectorStore.
  - Embedding function is based on openAI's text-embedding-3-large.
  - Question answering flow:
    - Retrieve relevant transcript chunks.
    - Pass context + question into LLM (via rlm/rag-prompt)  - - - Return grounded answers.  
  - Natural language queries like: *“What was discussed about budget in the meeting?”*
  - Only summary, transcript, and graph are stored in session (not raw video/audio).  

### 4. RAG-based QA
  - Upload & Process video (single click).
  - Sidebar: Meeting Summary.
  - Transcript Viewer: hidden under expander to avoid clutter.
  - Chatbot Interface: 
    - Persistent chat history via st. session_state.messages.
    - User can ask natural questions about the meeting.
    - Powered by RAG backend.
  - Only summary, transcript, and graph are stored in session (not raw video/audio).

- **Multi-Meeting Knowledge Base**  
  - Search across multiple past meetings.  
  - Cross-meeting queries for recurring topics.  

- **Dashboard & Integrations**  
  - Web UI (Streamlit/Next.js).  
  - Export summaries (PDF/Notion/Trello).  
  - Slack/Email integration for automatic meeting notes delivery.  

---

## ⚙️ Tech Stack
- **Video/Audio Handling**: `moviepy`, `pydub`  
- **Speech-to-Text**: [OpenAI Whisper](https://github.com/openai/whisper)  
- **LLM Framework**: [LangChain](https://www.langchain.com/)  
- **Vector Store**: InMemoryVectorStore (easily changeable to Chroma)  
- **Embeddings**: OpenAI text-embedding-3-large
- **Summarization**: LangChain Map-Reduce
- **Frontend**: Streamlit (dev) 

---

## 📌 Project Status
- **Phase 1 (Done)**: Video loading, audio extraction, test clip generation.  
- **Phase 2 (Done)**: Whisper transcription + cleaning pipeline.  
- **Phase 3 (Done)**: Summarization + RAG-based QA.  
- **Phase 4 (Done)**: UI dashboard + integrations.  

---

## 📖 Example Use Case
- Upload a 30 min Zoom/Meet recording.  
- SummitAI extracts and transcribes speech.  
- Get a structured meeting summary with **key decisions and action items**.  
- Ask questions like:  
  > *"What did was discussed about the Q4 budget?"*  
  > *"List all action items assigned to Bob."*  

---

## Special Notes 
- If you want to change the clip length considered for the
  SummitAI, change the **clip_length** variable to desired time(in seconds) in **extract_audio_from_video** function of **audio_extract.py**
