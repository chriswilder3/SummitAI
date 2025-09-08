import time
from openai import OpenAI, APIConnectionError
import os
from dotenv import load_dotenv
load_dotenv()


# hf_token = os.getenv('YOUR_HF_TOKEN')

from pydub import AudioSegment, silence

def safe_transcribe_with_openai(audio_file):
    retries = 3
    for attempt in range(retries):
        try:
            return transcribe_with_openai(audio_file)
        except APIConnectionError as e:
            print(f"Connection failed (attempt {attempt+1}), retrying...")
            time.sleep(2 ** attempt)  # exponential backoff
    raise Exception("Failed after retries")

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
    
    # from pyannote.audio import Pipeline
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


def fast_heuristic_diarization(audio_file, min_silence_len=500, silence_thresh=-40):
    """
    Rough speaker separation based on silences:
    - Splits audio into chunks separated by silences
    - Alternates speaker labels for each chunk (SPEAKER_0, SPEAKER_1)
    """
    sound = AudioSegment.from_file(audio_file, format="wav")

    print("---loaded audio---")
    # Split on silence
    chunks = silence.split_on_silence(
        sound,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh
    )
    print("---Split into chunks---")

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
    print("---Diarized the audio---")
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
    print("---Combined audio with diarization---")
    return

def combine_transcript_with_diarization_2(transcript_segments,
                                        diarized_segments,
                                        round_digits=2,
                                        merge_adjacent=True,
                                        max_gap_merge=0.5):
    """
    Merge a list of transcript segments with diarization segments.

    transcript_segments: list of either
       - dicts: {'start': float, 'end': float, 'text': str}
       - objects: with attributes .start .end .text (e.g. Whisper segments)
    diarized_segments: list of dicts: {'start': float, 'end': float, 'speaker': str}
       (if you used pyannote earlier you probably already converted to this form)

    Returns: list of dicts:
       [{'speaker': 'SPEAKER_00', 'start': 0.0, 'end': 4.5, 'text': '...'}, ...]
    """

    # --- helper to read a transcript segment (dict or object) ---
    def _read_transcript_seg(seg):
        if isinstance(seg, dict):
            s = float(seg.get("start", 0.0))
            e = float(seg.get("end", 0.0))
            txt = seg.get("text", "") or seg.get("content", "")
        else:
            # object-style (Whisper TranscriptionSegment)
            s = float(getattr(seg, "start", 0.0))
            e = float(getattr(seg, "end", 0.0))
            txt = getattr(seg, "text", "") or getattr(seg, "content", "")
        return s, e, str(txt).strip()

    # --- normalize diarization list to dicts with floats ---
    diar = []
    for d in diarized_segments:
        if isinstance(d, dict):
            ds = {"start": float(d.get("start", 0.0)),
                  "end": float(d.get("end", 0.0)),
                  "speaker": d.get("speaker", d.get("label", "SPEAKER_UNKNOWN"))}
        else:
            # fallback for objects with attributes (less common here)
            ds = {"start": float(getattr(d, "start", 0.0)),
                  "end": float(getattr(d, "end", 0.0)),
                  "speaker": getattr(d, "speaker", getattr(d, "label", "SPEAKER_UNKNOWN"))}
        diar.append(ds)

    # sort diarization segments by time (safety)
    diar.sort(key=lambda x: x["start"])

    merged_segments = []

    # For each transcript segment, find the diarization speaker with max overlap
    for seg in transcript_segments:
        t_start, t_end, text = _read_transcript_seg(seg)

        # compute overlap with each diar segment
        best_speaker = "SPEAKER_UNKNOWN"
        best_overlap = 0.0
        for d in diar:
            overlap = max(0.0, min(t_end, d["end"]) - max(t_start, d["start"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = d["speaker"]

        # if no overlap found, optionally pick nearest diar segment (commented out)
        # if best_overlap == 0.0:
        #     # find diar segment with minimal time distance
        #     distances = [(abs((d["start"]+d["end"])/2 - (t_start+t_end)/2), d["speaker"]) for d in diar]
        #     if distances:
        #         best_speaker = min(distances, key=lambda x: x[0])[1]

        merged_segments.append({
            "speaker": best_speaker,
            "start": round(t_start, round_digits),
            "end": round(t_end, round_digits),
            "text": text
        })

    # Optionally merge adjacent transcript segments that are from the same speaker
    # if merge_adjacent and merged_segments:
    #     compact = [merged_segments[0].copy()]
    #     for seg in merged_segments[1:]:
    #         last = compact[-1]
    #         gap = seg["start"] - last["end"]
    #         if seg["speaker"] == last["speaker"] and gap <= max_gap_merge:
    #             # merge: extend end and append text
    #             last["end"] = seg["end"]
    #             # keep a single space between stitched texts
    #             if last["text"] and seg["text"]:
    #                 last["text"] = last["text"].rstrip() + " " + seg["text"].lstrip()
    #             else:
    #                 last["text"] = (last["text"] or "") + (seg["text"] or "")
    #         else:
    #             compact.append(seg.copy())
    #     merged_segments = compact

    return merged_segments 

def split_audio(input_audio_file, chunk_length_ms=2*60*1000):
    audio = AudioSegment.from_file(input_audio_file, format="wav")
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i+chunk_length_ms]
        chunks.append((i/1000.0, chunk))  # store start time in seconds
    return chunks

def transcribe_audio_in_chunks(input_audio_file):
    chunks = split_audio(input_audio_file)
    all_segments = []

    for start_time, chunk in chunks:
        temp_file = "temp_chunk.wav"
        chunk.export(temp_file, format ="wav")
        transcript = safe_transcribe_with_openai(temp_file)

        for seg in transcript.segments:
            seg_start = seg.start + start_time
            seg_end = seg.end + start_time
            all_segments.append({
                "start": round(seg.start + start_time, 2),  # 2 decimal places
                "end": round(seg.end + start_time, 2),
                "text": seg.text
            })
    return all_segments

def transcribe_audio(input_audio_file):
    # input_audio_file  = "E:/downloads/meeting_preprocessed.wav"
    
    transcript = transcribe_audio_in_chunks(input_audio_file)
    print(" Transcription : ", transcript)

    # final_transcript = diarize_and_merge(input_audio_file,
    #                                hf_token, transcript)
    
    # diarized_segments = fast_heuristic_diarization(input_audio_file)
    # final_transcript = combine_transcript_with_diarization_2(
    #                                transcript,
    #                                 diarized_segments )
    return transcript

def main():
    input_audio_file  = "E:/downloads/meeting_preprocessed.wav"
    final_transcript = transcribe_audio(input_audio_file)
    # print("Final ",final_transcript)

if __name__ == "__main__":
    main()