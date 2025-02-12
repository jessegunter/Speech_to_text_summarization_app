from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from speech_to_text import transcribe_audio  
from text_to_translation import summarize_large_text, translate_large_text  

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Retrieve user settings
    user_choice = request.form.get("choice", "full")
    translate = request.form.get("translate", "false").lower() == "true"
    target_language = request.form.get("target_language", "english")

    try:
        # ✅ Step 1: Transcribe the audio and get confidence score
        transcription, confidence = transcribe_audio(file_path)  

        # ✅ Step 2: Summarize (if selected)
        if user_choice == "summary":
            transcription = summarize_large_text(transcription)

        # ✅ Step 3: Translate (if requested)
        if translate:
            transcription = translate_large_text(transcription, target_language)

        os.remove(file_path)  # ✅ Cleanup uploaded file

        return jsonify({
            "message": "Processing complete!",
            "transcription": transcription,
            "confidence": confidence  # ✅ Confidence score added to response
        })

    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)