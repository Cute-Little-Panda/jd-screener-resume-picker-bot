import logging
import os

import functions_framework
import google.auth
import vertexai
from flask import jsonify
from googleapiclient.discovery import build
from vertexai.generative_models import GenerativeModel

# --- CONFIGURATION ---
SHEET_ID = os.environ.get("SHEET_ID")
SHEET_RANGE = os.environ.get("SHEET_RANGE", "Sheet1!A:D")
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
REGION = os.environ.get("REGION", "us-central1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-1.5-flash-001")

# FIX: Hardcoded prompt to prevent Environment Variable truncation
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
    logger.info(f"Prompt sent to AI (Length: {len(prompt)})")

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
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }

    if request.method == "OPTIONS":
        return ("", 204, headers)

    if request.method == "GET":
        return (HTML_FORM, 200)

    if request.method == "POST":
        try:
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

            markdown_result = analyze_with_gemini(jd_text, resumes)

            if is_json_request:
                return (jsonify({"markdown": markdown_result}), 200, headers)
            else:
                return (HTML_RAW_OUTPUT.format(markdown=markdown_result), 200, headers)

        except Exception as e:
            logger.exception("System Error")
            return (f"System Error: {str(e)}", 500, headers)
