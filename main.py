import os
import json
import PyPDF2
from flask import Flask, request, jsonify,render_template
from flask_cors import CORS # Import CORS
from google import genai

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
client = genai.Client(api_key="")

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text
@app.route('/')
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "resume" not in request.files or "job_description" not in request.form:
        return jsonify({"error": "Missing data"}), 400

    resume_file = request.files["resume"]
    jd_text = request.form.get("job_description")
    
    pdf_path = os.path.join(UPLOAD_FOLDER, resume_file.filename)
    resume_file.save(pdf_path)
    resume_text = extract_text_from_pdf(pdf_path)

    # We use ONE prompt to get structured JSON for the UI
    prompt = f"""
    You are an ATS system. Analyze the following Resume and Job Description.
    
    Resume Text: {resume_text}
    Job Description: {jd_text}
    
    Return ONLY a JSON object with this exact structure:
    {{
      "score": (a number between 0 and 10),
      "skills": ["skill1", "skill2", "skill3"],
      "strengths": ["point1", "point2"],
      "improvements": ["point1", "point2"]
    }}
    """
    
    response = client.models.generate_content(
        model="gemini-2.5-flash", # Use the stable gemini-2.0-flash
        contents=prompt,
        config={'response_mime_type': 'application/json'} # Forces JSON output
    )
    
    # Parse the string response into a Python dictionary
    analysis_data = json.loads(response.text)
    return jsonify(analysis_data)

if __name__ == "__main__":
    app.run(debug=True, port=8080)
