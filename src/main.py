import os
import logging
import json
import functions_framework
import google.auth
from googleapiclient.discovery import build
import vertexai
from vertexai.generative_models import GenerativeModel

# --- CONFIGURATION ---
SHEET_ID = os.environ.get("SHEET_ID")
SHEET_RANGE = "Sheet1!A:D"  
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
REGION = "us-central1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

vertexai.init(project=PROJECT_ID, location=REGION)
model = GenerativeModel("gemini-1.5-flash-001")

def get_sheets_service():
    creds, _ = google.auth.default()
    return build('sheets', 'v4', credentials=creds)

def fetch_resumes_from_sheet(service):
    """Fetches data from Cols A-D."""
    try:
        sheet = service.spreadsheets()
        result = sheet.values().get(spreadsheetId=SHEET_ID, range=SHEET_RANGE).execute()
        rows = result.get('values', [])
        
        resumes = []
        for row in rows:
            if len(row) < 2: continue
            name = row[0]
            content = row[1]
            status = row[2] if len(row) > 2 else ""
            path_url = row[3] if len(row) > 3 else "#"
            is_archived = "archived" in status.lower()

            resumes.append({
                "name": name, 
                "content": content, 
                "is_archived": is_archived,
                "path": path_url
            })
        return resumes
    except Exception as e:
        logger.error(f"Error reading sheet: {e}")
        return []

def analyze_with_gemini(jd_text, resumes):
    context_str = ""
    for r in resumes:
        status = "[ARCHIVED]" if r['is_archived'] else "[ACTIVE]"
        context_str += f"\n--- RESUME: {r['name']} {status} ---\n{r['content']}\n"

    prompt = f"""
    **ROLE:** Expert Technical Recruiter.
    **INPUT JD:** {jd_text}
    **RESUME POOL:** {context_str}
    
    **TASK:**
    1. Analyze the JD.
    2. Pick the BEST resume. (Lower priority for 'Archived' unless >90% match).
    3. Identify GAPS.
    4. GENERATE 3-5 quantitative bullet points to bridge those gaps.
    
    **OUTPUT JSON ONLY:**
    {{
      "top_match_name": "Exact Name",
      "analysis": "Brief analysis",
      "reasoning": "Reasoning",
      "bullets": [
        {{ "section": "Section Name", "text": "Bullet content" }}
      ]
    }}
    """
    
    response = model.generate_content(prompt)
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
    
    # 1. Serve the Form
    if request.method == "GET":
        return HTML_FORM

    # 2. Handle the Analysis
    if request.method == "POST":
        try:
            # Handle Form Data (Web) or JSON (API)
            if request.content_type == 'application/json':
                data = request.get_json()
                jd_text = data.get('message', {}).get('text', '')
            else:
                data = request.form
                jd_text = data.get('jd', '')

            if not jd_text:
                return "Error: No JD provided.", 400

            # Run Logic
            sheets_service = get_sheets_service()
            resumes = fetch_resumes_from_sheet(sheets_service)
            
            if not resumes:
                return "Error: No resumes found in Sheet.", 500
                
            result = analyze_with_gemini(jd_text, resumes)
            
            if "error" in result:
                return f"AI Error: {result.get('error')}", 500

            # Format HTML Result
            top_name = result.get('top_match_name', 'Unknown')
            top_resume_url = next((r['path'] for r in resumes if r['name'] == top_name), "#")
            
            bullets_html = ""
            for b in result.get('bullets', []):
                bullets_html += f"<li><b>{b['section']}:</b> {b['text']}</li>"

            return HTML_RESULT.format(
                name=top_name,
                url=top_resume_url,
                analysis=result.get('analysis', ''),
                reasoning=result.get('reasoning', ''),
                bullets=bullets_html
            )

        except Exception as e:
            logger.error(e)
            return f"System Error: {str(e)}", 500