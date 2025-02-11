import whisper
from pydub import AudioSegment
import os

def convert_mp3_to_wav(mp3_path, wav_path):
    """Convert MP3 to WAV format using pydub."""
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")

def transcribe_audio(mp3_path, model_size="base"):
    """
    Transcribe an MP3 file using Whisper.

    Parameters:
    - mp3_path (str): Path to the MP3 file.
    - model_size (str): Whisper model size (tiny, base, small, medium, large).

    Returns:
    - str: Transcription of the audio.
    """
    # Convert MP3 to WAV
    wav_path = mp3_path.replace(".mp3", ".wav")
    convert_mp3_to_wav(mp3_path, wav_path)

    # Load Whisper model
    model = whisper.load_model(model_size)

    # Transcribe the audio
    result = model.transcribe(wav_path)

    # Delete the WAV file after transcription
    os.remove(wav_path)

    return result["text"]

# Example usage
if __name__ == "__main__":
    mp3_file = "sample.mp3"  # Replace with your MP3 file path
    transcription = transcribe_audio(mp3_file, model_size="base")
    print("Transcription:", transcription)
