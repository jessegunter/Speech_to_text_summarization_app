from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from speech_to_text import transcribe_audio  # Import from your speech-to-text file
from text_to_translation import summarize_text, translate_text  # Import from your text summarization file


# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------ API Routes ------------------

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Retrieve user settings
    user_choice = request.form.get("choice")  # 'full' or 'summary'
    language = request.form.get("language", "english")
    translate = request.form.get("translate", "false")  # 'true' or 'false'
    target_language = request.form.get("target_language", "english")  # Default to English

    # Step 1: Transcribe the audio file using `speech_to_text.py`
    transcription = transcribe_audio(file_path)
    
    # Step 2: Summarization (if user selects 'summary') using `text_to_translation.py`
    if user_choice == "summary":
        transcription = summarize_text(transcription)
    
    # Step 3: Translation (if translation is requested) using `text_to_translation.py`
    if translate.lower() == "true":
        transcription = translate_text(transcription, target_language)

    # Cleanup uploaded file after processing
    os.remove(file_path)

    return jsonify({
        "message": "Processing complete!",
        "transcription": transcription
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)


