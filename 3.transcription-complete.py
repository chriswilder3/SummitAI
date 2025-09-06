from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

from pyannote.audio import Pipeline
hf_token = os.getenv('YOUR_HF_TOKEN')

def transcribe_with_openai(audio_file: str,
                            model: str = "whisper-1"):
    
    client = OpenAI( api_key= os.getenv('OPENAI_API_KEY'))

    with open(audio_file,"rb") as f:
        transcript = client.audio.transcriptions.create(
            model= model,
            file= f,
            response_format="verbose_json"
        )
    return transcript

def diarize_and_merge(audio_file, hf_token, transcript):
    
    # --- Step 1: Run diarization ---
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                        use_auth_token=hf_token)
    diarization = pipeline(audio_file)

    diar_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        diar_segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })

    # --- Step 2: Get Whisper transcript (with timestamps) ---
    whisper_segments = transcript.segments

    # --- Step 3: Merge labels ---
    merged = []
    for seg in whisper_segments:
        seg_start, seg_end = seg.start, seg.end  # dot notation
        seg_text = seg.text

        # Find overlapping diarization label
        speaker = "UNKNOWN"
        for d in diar_segments:
            if not (seg_end < d["start"] or seg_start > d["end"]):
                speaker = d["speaker"]
                break

        merged.append({
            "speaker": speaker,
            "start": seg_start,
            "end": seg_end,
            "text": seg_text
        })
    return merged

from pydub import AudioSegment, silence

def fast_heuristic_diarization(audio_file, min_silence_len=500, silence_thresh=-40):
    """
    Rough speaker separation based on silences:
    - Splits audio into chunks separated by silences
    - Alternates speaker labels for each chunk (SPEAKER_0, SPEAKER_1)
    """
    sound = AudioSegment.from_file(audio_file, format="wav")

    # Split on silence
    chunks = silence.split_on_silence(
        sound,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh
    )

    diarized_segments = []
    speaker_toggle = 0  # alternate speakers
    current_time = 0.0

    for chunk in chunks:
        duration_sec = len(chunk) / 1000.0  # convert ms to sec
        diarized_segments.append({
            "speaker": f"SPEAKER_{speaker_toggle:02d}",
            "start": current_time,
            "end": current_time + duration_sec,
            "text": ""  # placeholder, will merge with Whisper transcript later
        })
        current_time += duration_sec
        speaker_toggle = 1 - speaker_toggle  # alternate speakers
    
    return diarized_segments

def combine_transcript_with_diarization(whisper_segments,diar_segments):
    merged = []
    for seg in whisper_segments:
        seg_start, seg_end = seg.start, seg.end  # dot notation
        seg_text = seg.text

        # Find overlapping diarization label
        speaker = "UNKNOWN"
        for d in diar_segments:
            if not (seg_end < d["start"] or seg_start > d["end"]):
                speaker = d["speaker"]
                break

        merged.append({
            "speaker": speaker,
            "start": seg_start,
            "end": seg_end,
            "text": seg_text
        })
    return merged

def main():
    input_audio_file  = "E:/downloads/meeting_preprocessed.wav"
    transcript = transcribe_with_openai(input_audio_file)
    print(" Transcription : ", transcript.text)

    # final_transcript = diarize_and_merge(input_audio_file,
    #                                hf_token, transcript)
    
    diarized_segments = fast_heuristic_diarization(input_audio_file)
    final_transcript = combine_transcript_with_diarization(
                                    transcript.segments,
                                    diarized_segments )
    print("Final ",final_transcript)
if __name__ == "__main__":
    main()