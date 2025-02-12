import whisper
from pydub import AudioSegment
import os
import numpy as np

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
    - tuple: (Transcription text, Confidence score)
    """
    try:
        # Convert MP3 to WAV
        wav_path = mp3_path.replace(".mp3", ".wav")
        convert_mp3_to_wav(mp3_path, wav_path)

        # Load Whisper model
        model = whisper.load_model(model_size)

        # Transcribe the audio
        result = model.transcribe(wav_path)
        os.remove(wav_path)  # Cleanup WAV file

        transcribed_text = result.get("text", "").strip()

        # Compute confidence score
        if "segments" in result:
            logprobs = [segment.get("avg_logprob", None) for segment in result["segments"] if "avg_logprob" in segment]
            confidence = np.exp(np.mean(logprobs)) * 100 if logprobs else 0.0
        else:
            confidence = 0.0

        return transcribed_text, round(confidence, 2)  # Always return TWO values

    except Exception as e:
        print(f"Error in transcribe_audio: {e}")
        return "Error processing audio", 0.0  # Default fallback values

# Example usage
if __name__ == "__main__":
    mp3_file = "sample.mp3"  # Replace with your MP3 file path
    transcription, confidence = transcribe_audio(mp3_file, model_size="base")
    print(f"\nEstimated Confidence: {confidence:.2f}%\nTranscription: {transcription}")
