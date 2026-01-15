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

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=REGION)
model = GenerativeModel("gemini-1.5-flash-001")

def get_sheets_service():
    creds, _ = google.auth.default()
    return build('sheets', 'v4', credentials=creds)

def fetch_resumes_from_sheet(service):
    """
    Fetches data from Cols A-D.
    Returns list of dicts: {name, content, is_archived, path}
    """
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
    1. Analyze the JD for hard/soft skills and culture.
    2. Pick the BEST resume. (Lower priority for 'Archived' unless >90% match).
    3. Identify GAPS in the best resume vs the JD.
    4. GENERATE 3-5 quantitative bullet points to bridge those gaps.
    
    **OUTPUT JSON ONLY:**
    {{
      "top_match_name": "Exact Name of the file as provided",
      "analysis": "Brief JD analysis",
      "reasoning": "Why this resume wins",
      "gaps": ["gap1", "gap2"],
      "bullets": [
        {{ "section": "Where to insert", "text": "Bullet point content" }}
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

@functions_framework.http
def handle_chat(request):
    event = request.get_json()
    if not event or 'message' not in event:
        return {"text": "Send me a JD to start."}
    
    jd_text = event['message'].get('text', '')
    
    try:
        sheets_service = get_sheets_service()
        resumes = fetch_resumes_from_sheet(sheets_service)
        
        if not resumes:
            return {"text": "No resumes found in the Google Sheet."}
            
        result = analyze_with_gemini(jd_text, resumes)
        
        if "error" in result:
            return {"text": "Sorry, I couldn't process that analysis."}

        top_name = result.get('top_match_name', '')
        top_resume_url = next((r['path'] for r in resumes if r['name'] == top_name), "#")

        widgets = [
            {"textParagraph": {"text": f"<b>🏆 Match:</b> <a href='{top_resume_url}'>{top_name}</a>"}},
            {"textParagraph": {"text": f"<b>Analysis:</b> {result.get('analysis')}"}},
            {"textParagraph": {"text": f"<b>Why:</b> {result.get('reasoning')}"}},
            {"divider": {}},
            {"textParagraph": {"text": "<b>SUGGESTED BULLETS:</b>"}}
        ]
        
        for b in result.get('bullets', []):
            widgets.append({"textParagraph": {"text": f"<b>{b['section']}:</b><br>• {b['text']}"}})

        return {
            "cardsV2": [{
                "cardId": "resumeAnalysis",
                "card": {
                    "header": {"title": "JD Screener & Resume Picker"}, # UPDATED NAME HERE
                    "sections": [{"widgets": widgets}]
                }
            }]
        }

    except Exception as e:
        logger.error(e)
        return {"text": f"System Error: {str(e)}"}