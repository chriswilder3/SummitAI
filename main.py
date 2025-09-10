import streamlit as st
from summerization import map_reduce_summarize
from transcription import transcribe_audio
from audio_extract import extract_audio_from_video
from audio_preprocess import preprocess_audio
from rag import create_rag
from langgraph.graph import StateGraph
import os

st.title("🎥 SummitAI: Your AI Meeting & Conference Companion")
st.write("Upload your video recording to analyze now")

uploaded_file = st.file_uploader("Upload an MP4 video", type=["mp4"])


if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")
    
    # Temporary paths for processing
    raw_video_file = f"temp_{uploaded_file.name}"
    extracted_audio_file = "temp_extracted.wav"
    preprocessed_audio_file = "temp_preprocessed.wav"

    # Save uploaded file locally
    with open(raw_video_file, "wb") as f:
        f.write(uploaded_file.read())

    if st.button("Process Video"):
        with st.spinner("Extracting audio..."):
            extract_audio_from_video(raw_video_file, extracted_audio_file)
        
        with st.spinner("Preprocessing audio..."):
            preprocess_audio(extracted_audio_file, preprocessed_audio_file)
        
        with st.spinner("Transcribing audio..."):
            transcript_segments = transcribe_audio(preprocessed_audio_file)
            st.session_state.transcript_segments = transcript_segments
        
        with st.spinner("Summarizing meeting..."):
            summary = map_reduce_summarize( transcript_segments)  # Pass your llm object if needed
            st.session_state.summary = summary

        transcript_text = "\n".join([seg["text"] for seg in transcript_segments])
        
        # Create the graph from RAG app and save it for session
        graph = create_rag(transcript_text)
        st.session_state.graph = graph

        # Clear messages if it exists before
        if "messages" in st.session_state:
            st.session_state.pop("messages")

if "summary" in st.session_state:
    st.sidebar.subheader("📌 Meeting Summary")
    st.sidebar.write(st.session_state.summary["output_text"])

if "graph" in st.session_state:
    # Chat state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.write("💬 Chat history:")
    with st.container(height=340):
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # User input
        if user_input := st.chat_input("Ask question about the meeting:"):
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Call RAG Application for response
            graph = st.session_state.graph
            response = graph.invoke({"question":user_input})
            with st.chat_message("assistant"):
                st.markdown(response["answer"])
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

    # Display complete transcript for reference                    
    if "transcript_segments" in st.session_state:
        st.subheader("📝 Transcript")
        with st.expander("Show full transcript"):
            for seg in st.session_state.transcript_segments:
                st.write(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")