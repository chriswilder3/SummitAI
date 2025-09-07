# SummitAI

> **From Meetings to Meaning — AI that Transcribes, Summarizes, and Surfaces Insights.**

SummitAI is an **AI-powered meeting companion** that transforms raw meeting recordings into **structured knowledge**.  
Users upload a video file, and SummitAI automatically extracts the audio, transcribes speech using Whisper, and applies LLM-powered pipelines for **summarization, question answering, and retrieval-augmented insights**.  

---

## ✨ Features (Current & Planned)

### ✅ Implemented so far
- **Video → Audio Extraction**  
  - Load `.mp4` (or other formats) using `moviepy`.  
  - Extract audio track, resampled to **16kHz mono WAV** (Whisper-ready).  
  - Dev-friendly shortcut: generate short test clips (e.g., 1 min from 2:00–3:00).  

- **Clean Dev Workflow**  
  - Test with smaller video chunks before scaling to full-length meetings.  
  - Support for both `moviepy` (Python-based) and `ffmpeg` (fast CLI).  

---

### 🛠️ Work in Progress
- **Speech-to-Text**  
  - Whisper integration for accurate transcripts.  
  - Preprocessing (volume normalization, silence trimming).  
  - Speaker diarization (separating who said what).  

- **Text Cleaning & Structuring**  
  - Removing filler words.  
  - JSON transcript structure with timestamps + speakers.  

---

### 🔮 Next Steps (Planned)
- **Meeting Summaries**  
  - Concise summaries grouped into *Decisions*, *Action Items*, *Open Questions*.  

- **RAG-based QA**  
  - Chunk transcripts into vector DB.  
  - Natural language queries like: *“What was discussed about budget last week?”*  

- **Multi-Meeting Knowledge Base**  
  - Search across multiple past meetings.  
  - Cross-meeting queries for recurring topics.  

- **Dashboard & Integrations**  
  - Web UI (Streamlit/Next.js).  
  - Export summaries (PDF/Notion/Trello).  
  - Slack/Email integration for automatic meeting notes delivery.  

---

## ⚙️ Tech Stack
- **Video/Audio Handling**: `moviepy`, `ffmpeg`, `pydub`  
- **Speech-to-Text**: [OpenAI Whisper](https://github.com/openai/whisper)  
- **LLM Framework**: [LangChain](https://www.langchain.com/)  
- **Vector Store**: FAISS / Pinecone / Weaviate  
- **Backend**: FastAPI  
- **Frontend**: Streamlit (dev) → Next.js (production)  

---

## 🚧 Development Notes
- For testing, we currently reduce input video to **1-minute clips** (e.g., 2:00–3:00 mark) to speed up iteration.  
- Full-meeting support will be added once the transcription pipeline is stable.  
- Keeping preprocessing modular so Whisper + RAG can be swapped or upgraded later.  

---

## 📌 Project Status
- **Phase 1 (Done)**: Video loading, audio extraction, test clip generation.  
- **Phase 2 (Done)**: Whisper transcription + cleaning pipeline.  
- **Phase 3 (In Progress)**: Summarization + RAG-based QA.  
- **Phase 4 (Planned)**: UI dashboard + integrations.  

---

## 📖 Example Use Case
- Upload a 1-hour Zoom/Meet recording.  
- SummitAI extracts and transcribes speech.  
- Get a structured meeting summary with **key decisions and action items**.  
- Ask questions like:  
  > *"What did Alice say about the Q4 budget?"*  
  > *"List all action items assigned to Bob."*  

---
