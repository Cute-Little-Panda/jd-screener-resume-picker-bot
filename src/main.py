import json
import logging
import os

# --- NEW: Firebase Imports ---
import firebase_admin
import functions_framework
import google.auth
import vertexai
from firebase_admin import auth
from flask import jsonify
from googleapiclient.discovery import build
from vertexai.generative_models import GenerativeModel

# Initialize Firebase Admin (Safe for Cloud Run hot-reloads)
try:
    firebase_admin.get_app()
except ValueError:
    firebase_admin.initialize_app()

# --- CONFIGURATION ---
SHEET_ID = os.environ.get("SHEET_ID")
SHEET_RANGE = os.environ.get("SHEET_RANGE", "Sheet1!A:D")
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
REGION = os.environ.get("REGION")
MODEL_NAME = os.environ.get("MODEL_NAME")

PROMPT_TEMPLATE = """
**ROLE:** Resume Picker based on Job Description.
**INPUT JD:** {jd_text}
**RESUME POOL:** {context_str}

**TASK:**
1. Analyze the JD.
2. Pick the BEST resume. (Lower priority for 'Archived' unless >90% match).
3. Identify GAPS between the best resume and the JD.
4. GENERATE 3-5 quantitative, high-impact bullet points to bridge those gaps.

**CONSTRAINTS:**
- Output **MARKDOWN** ONLY. Do not output JSON.
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
"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
sheets_service = None


def get_model():
    global model
    if model is None:
        vertexai.init(project=PROJECT_ID, location=REGION)
        model = GenerativeModel(MODEL_NAME)
    return model


def get_sheets_service():
    global sheets_service
    if sheets_service is None:
        creds, _ = google.auth.default()
        sheets_service = build("sheets", "v4", credentials=creds)
    return sheets_service


def verify_firebase_token(request):
    """
    Decodes the Firebase ID Token from the Authorization header.
    Returns the user dictionary if valid, None if invalid.
    """
    auth_header = request.headers.get("Authorization")

    if not auth_header or not auth_header.startswith("Bearer "):
        logger.warning("Auth: Missing or invalid Bearer header")
        return None

    token = auth_header.split("Bearer ")[1]

    try:
        # verifying the token verifies signature, expiration, and project matching
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except Exception as e:
        logger.warning(f"Auth: Token verification failed: {e}")
        return None


def fetch_resumes_from_sheet(service):
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

    prompt = PROMPT_TEMPLATE.format(jd_text=jd_text, context_str=context_str)

    try:
        response = model_instance.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"AI Error: {e}")
        return f"Error generating content: {str(e)}"


# --- HTML TEMPLATES ---
HTML_FORM = """
<!DOCTYPE html>
<html>
<body>
    <h1>JD Screener</h1>
    <p><i>Note: API now requires Authentication. This form may not work without a token.</i></p>
    <form method="POST">
        <textarea name="jd" style="width:100%; height:150px;" placeholder="Paste JD..."></textarea><br>
        <button type="submit">Analyze</button>
    </form>
</body>
</html>
"""

HTML_RAW_OUTPUT = """
<!DOCTYPE html>
<html>
<body>
    <h1>Markdown Result</h1>
    <textarea style="width:100%; height:500px;">{markdown}</textarea>
    <br><a href="/">Back</a>
</body>
</html>
"""


@functions_framework.http
def handle_chat(request):
    # 1. Update CORS to accept Authorization header
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",  # <--- Added Authorization
    }

    if request.method == "OPTIONS":
        return ("", 204, headers)

    if request.method == "GET":
        return (HTML_FORM, 200)

    if request.method == "POST":
        try:
            # 2. Authentication Check (Firebase)
            user = verify_firebase_token(request)
            if not user:
                return (
                    jsonify(
                        {"error": "Unauthorized: Invalid or missing Firebase Token"}
                    ),
                    401,
                    headers,
                )

            logger.info(f"Processing request for user: {user.get('email')}")

            # 3. Process Request
            is_json_request = request.content_type == "application/json"

            if is_json_request:
                data = request.get_json(silent=True) or {}
                jd_text = data.get("message", {}).get("text", "") or data.get("jd", "")
            else:
                data = request.form
                jd_text = data.get("jd", "")

            if not jd_text:
                return ("Error: No JD provided.", 400, headers)

            svc = get_sheets_service()
            resumes = fetch_resumes_from_sheet(svc)

            if not resumes:
                return ("Error: No resumes found in Sheet.", 500, headers)

            logger.info(f"{len(resumes)} Resumes fetched.")

            markdown_result = analyze_with_gemini(jd_text, resumes)

            if is_json_request:
                return (jsonify({"markdown": markdown_result}), 200, headers)
            else:
                return (HTML_RAW_OUTPUT.format(markdown=markdown_result), 200, headers)

        except Exception as e:
            logger.exception("System Error")
            return (f"System Error: {str(e)}", 500, headers)
