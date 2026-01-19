import json
import logging
import os

import functions_framework
import google.auth
import vertexai
from googleapiclient.discovery import build
from vertexai.generative_models import GenerativeModel

# --- CONFIGURATION ---
# We use defaults to prevent crashes if env vars are missing during build
SHEET_ID = os.environ.get("SHEET_ID")
SHEET_RANGE = os.environ.get("SHEET_RANGE", "Sheet1!A:D")
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
REGION = os.environ.get("REGION", "us-central1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gemini-1.5-flash-001")

PROMPT_TEMPLATE: str = os.environ.get(
    "PROMPT_TEMPLATE",
    """
    **ROLE:** Expert Technical Recruiter.
    **INPUT JD:** {jd_text}
    **RESUME POOL:** {context_str}

    **TASK:**
    1. Analyze the JD.
    2. Pick the BEST resume. (Lower priority for 'Archived' unless >90% match).
    3. Identify GAPS between the best resume and the JD.
    4. GENERATE 3-5 quantitative, high-impact bullet points to bridge those gaps.

    **CONSTRAINTS:**
    - Output **MARKDOWN(.md) ONLY**. Do not output JSON.
    - Ensure the total output length is sufficient to cover the details but remains under 10,000 CHARACTERS.

    **OUTPUT FORMAT:**
    Please follow this exact Markdown structure:

    # [Exact Name and Path of Best Resume]

    ## Analysis
    [Brief analysis of the resume's profile against the JD]

    ## Reasoning
    [Explanation of why this resume was chosen over others]

    ## Suggested Improvements (Bridging Gaps)
    * **[Target Section Name]**: [Content of the bullet point]
    * **[Target Section Name]**: [Content of the bullet point]
    * **[Target Section Name]**: [Content of the bullet point]
    """,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for lazy loading
model = None
sheets_service = None


def get_model():
    """Lazy load Vertex AI to prevent cold-start crashes."""
    global model
    if model is None:
        vertexai.init(project=PROJECT_ID, location=REGION)
        model = GenerativeModel(MODEL_NAME)
    return model


def get_sheets_service():
    """Lazy load Sheets Service."""
    global sheets_service
    if sheets_service is None:
        creds, _ = google.auth.default()
        sheets_service = build("sheets", "v4", credentials=creds)
    return sheets_service


def fetch_resumes_from_sheet(service):
    """Fetches data from Cols A-D."""
    try:
        if not SHEET_ID:
            logger.error("SHEET_ID is missing from Environment Variables")
            return []

        sheet = service.spreadsheets()
        result = sheet.values().get(spreadsheetId=SHEET_ID, range=SHEET_RANGE).execute()
        rows = result.get("values", [])

        resumes = []
        for row in rows:
            if len(row) < 2:
                continue
            name = row[0]
            content = row[1]
            status = row[2] if len(row) > 2 else ""
            path_url = row[3] if len(row) > 3 else "#"
            is_archived = "archived" in status.lower()

            resumes.append(
                {
                    "name": name,
                    "content": content,
                    "is_archived": is_archived,
                    "path": path_url,
                }
            )
        return resumes
    except Exception as e:
        logger.error(f"Error reading sheet: {e}")
        return []


def analyze_with_gemini(jd_text, resumes):
    model_instance = get_model()

    context_str = ""
    for r in resumes:
        status = "[ARCHIVED]" if r["is_archived"] else "[ACTIVE]"
        context_str += f"\n--- RESUME: {r['name']}, path_to_resume: {r['path']}, {status} ---\n{r['content']}\n"

    prompt = PROMPT_TEMPLATE.format(jd_text=jd_text, context=context_str)

    response = model_instance.generate_content(prompt)
    try:
        clean_json = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except Exception as e:
        logger.error(f"AI Error: {response.text}")
        return {"error": "AI parsing failed"}


# --- HTML TEMPLATES ---
HTML_FORM = """
<!DOCTYPE html>
<html>
<head>
    <title>JD Screener</title>
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; }
        textarea { width: 100%; height: 200px; margin-bottom: 20px; padding: 10px; }
        button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; font-size: 16px; }
        button:hover { background: #0056b3; }
        h2 { border-bottom: 2px solid #eee; padding-bottom: 10px; }
        .spinner { display: none; color: #666; }
    </style>
</head>
<body>
    <h1>📄 JD Screener & Resume Picker</h1>
    <form method="POST" onsubmit="document.getElementById('spin').style.display='block'">
        <label><b>Paste Job Description:</b></label><br>
        <textarea name="jd" required placeholder="Paste JD here..."></textarea><br>
        <button type="submit">Analyze Resumes</button>
        <span id="spin" class="spinner">Processing (this takes ~10s)...</span>
    </form>
</body>
</html>
"""

HTML_RESULT = """
<!DOCTYPE html>
<html>
<head>
    <title>Analysis Result</title>
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; line-height: 1.6; }
        .card { border: 1px solid #ddd; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .winner { color: #2e7d32; font-weight: bold; font-size: 1.2em; }
        a { color: #007bff; text-decoration: none; }
        ul { background: #f9f9f9; padding: 20px 40px; border-radius: 5px; }
        .back-btn { display: inline-block; margin-top: 20px; color: #666; }
    </style>
</head>
<body>
    <h1>✅ Analysis Complete</h1>
    <div class="card">
        <p><b>🏆 Top Match:</b> <a href="{url}" target="_blank" class="winner">{name}</a></p>
        <p><b>Analysis:</b> {analysis}</p>
        <p><b>Why it won:</b> {reasoning}</p>
        <hr>
        <h3>Suggested Improvements:</h3>
        <ul>
            {bullets}
        </ul>
    </div>
    <a href="/" class="back-btn">← Scan Another JD</a>
</body>
</html>
"""


@functions_framework.http
def handle_chat(request):
    """Handles both GET (Form) and POST (Analysis)."""

    # 1. Define CORS Headers ---
    # These headers allow your React app (on any domain) to talk to this function.
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Max-Age": "3600",
    }

    # 2. Handle Pre-flight Request (OPTIONS) ---
    # Browsers send this first to check if they are allowed to connect.
    if request.method == "OPTIONS":
        return ("", 204, headers)

    # 3. Serve the Form
    if request.method == "GET":
        return HTML_FORM

    # 4. Handle the Analysis
    if request.method == "POST":
        try:
            # Handle Form Data (Web) or JSON (API)
            if request.content_type == "application/json":
                data = request.get_json()
                jd_text = data.get("message", {}).get("text", "")
            else:
                data = request.form
                jd_text = data.get("jd", "")

            if not jd_text:
                return "Error: No JD provided.", 400

            # Run Logic
            svc = get_sheets_service()
            resumes = fetch_resumes_from_sheet(svc)

            if not resumes:
                return (
                    "Error: No resumes found in Sheet. (Check Sheet Permissions and Sheet Name)",
                    500,
                )

            result = analyze_with_gemini(jd_text, resumes)

            if "error" in result:
                return f"AI Error: {result.get('error')}", 500

            # Format HTML Result
            top_name = result.get("top_match_name", "Unknown")
            # Safe access to resumes list
            top_resume_url = "#"
            for r in resumes:
                if r["name"] == top_name:
                    top_resume_url = r["path"]
                    break

            bullets_html = ""
            for b in result.get("bullets", []):
                bullets_html += f"<li><b>{b['section']}:</b> {b['text']}</li>"

            return HTML_RESULT.format(
                name=top_name,
                url=top_resume_url,
                analysis=result.get("analysis", ""),
                reasoning=result.get("reasoning", ""),
                bullets=bullets_html,
            )

        except Exception as e:
            logger.error(e)
            return f"System Error: {str(e)}", 500
