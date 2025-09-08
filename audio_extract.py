import os
from moviepy import (
    VideoClip,
    VideoFileClip,
    AudioFileClip,
    AudioClip,
)

def extract_audio_from_video(input_vid_path:str, 
                             output_wav_path:str,
                             sample_rate: int = 16000,
                             mono: bool = True):
    """
    Extract audio from a video and save as a WAV file suitable for Whisper.
    - sample_rate: target sample rate in Hz (Whisper-friendly = 16000)
    - mono: if True, forces 1 channel
    """
    if not os.path.exists(input_vid_path):
        raise FileNotFoundError(f"Video not found: {input_vid_path}")
    
    # Load the Video file clip and check for audio

    clip = VideoFileClip(input_vid_path)
    if clip.duration >= 500:
        clip = clip.subclipped(1, 500)

    # clip = VideoFileClip(input_vid_path)
    if clip.audio is None:
        clip.close()
        raise RuntimeError("No audio track found in the video.")
    
    # Set mono
    ffmpeg_params = ["-ac", "1"] if mono else None

    # write to audio file
    clip.audio.write_audiofile(output_wav_path,fps=sample_rate,ffmpeg_params=ffmpeg_params)
    clip.close()

def main():
    input_vid_file = "E:/downloads/conference_video.mp4"
    output_audio_file  = "E:/downloads/meeting_16k_mono.wav"
    extract_audio_from_video(input_vid_file, output_audio_file)
    print(" Saved. ")

if __name__ == "__main__":
    main()