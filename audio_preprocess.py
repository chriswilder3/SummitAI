from pydub import AudioSegment, effects, silence
import noisereduce as nr
import librosa
import soundfile as sf
import os

def preprocess_audio(input_wav: str,
                             output_wav: str,
                             silence_thresh: int = -40,
                             min_silence_len: int = 700):
    """
    Preprocess meeting audio for Whisper:
    1. Normalize volume
    2. Remove long silences
    3. Reduce background noise
    4. Save as 16kHz mono WAV
    """

    if not os.path.exists(input_wav):
        raise FileNotFoundError(f"Input file not found: {input_wav}")

    # --- Step 1: Normalize & remove silences with pydub ---
    sound = AudioSegment.from_file(input_wav, format="wav")

    # Normalize volume
    normalized_sound = effects.normalize(sound)

    # Split on silence
    chunks = silence.split_on_silence(
        normalized_sound,
        min_silence_len=min_silence_len,  # ms
        silence_thresh=silence_thresh     # dBFS
    )

    # Recombine chunks with small padding (keeps flow natural)
    processed = AudioSegment.silent(duration=200)
    for chunk in chunks:
        processed += chunk + AudioSegment.silent(duration=200)

    # Save cleaned file
    processed.export(output_wav, format="wav")

    # # --- Step 2: Noise reduction with noisereduce ---
    # y, sr = librosa.load(temp_file, sr=16000, mono=True)  # resample to 16kHz mono
    # reduced = nr.reduce_noise(y=y, sr=sr)

    # # Save final output
    # sf.write(output_wav, reduced, sr, subtype="PCM_16")

    # # Cleanup
    # os.remove(temp_file)

def main():
    input_audio_file  = "E:/downloads/meeting_16k_mono.wav"
    output_audio_file = "E:/downloads/meeting_preprocessed.wav"
    preprocess_audio(input_audio_file,output_audio_file )
    print("Saved cleaned audio → meeting_processed")

if __name__ == "__main__":
    main()