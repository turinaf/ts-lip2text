import os
import subprocess

def extract_audio(video_path, output_audio_path, sample_rate=16000):
    """
    Extracts audio from a video using ffmpeg.
    """
    if os.path.exists(output_audio_path):
        return True

    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)
    
    # We use ffmpeg to easily strip the audio track and store as wav
    # -y overwrites, -vn disables video, -ac 1 is mono, -ar is sample rate
    command = [
        "ffmpeg", "-y", "-i", video_path, 
        "-vn", "-ac", "1", "-ar", str(sample_rate), 
        "-loglevel", "error", output_audio_path
    ]
    
    try:
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio from {video_path}: {e}")
        return False
