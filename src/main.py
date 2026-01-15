import os
import logging
import json
import functions_framework
import google.auth
from googleapiclient.discovery import build
import vertexai
from vertexai.generative_models import GenerativeModel

# --- CONFIGURATION ---
# Best practice: Set these as environment variables in Cloud Run
RESUME_FOLDER_ID = os.environ.get("RESUME_FOLDER_ID", "YOUR_HARDCODED_ID_HERE")
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
REGION = "us-central1"

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Vertex AI
vertexai.init(project=PROJECT_ID, location=REGION)
model = GenerativeModel("gemini-1.5-flash-001")

def get_drive_service():
    """Authenticates using the Cloud Run Service Account."""
    creds, _ = google.auth.default()
    return build('drive', 'v3', credentials=creds)

def fetch_resumes(service, folder_id):
    """Fetches .md files from main folder and 'Archived' subfolder."""
    resumes = []
    
    # 1. Fetch Active Resumes (Root of folder)
    query = f"'{folder_id}' in parents and mimeType = 'text/markdown' and trashed = false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    
    for file in results.get('files', []):
        content = service.files().get_media(fileId=file['id']).execute().decode('utf-8')
        resumes.append({"name": file['name'], "content": content, "is_archived": False})

    # 2. Find and Fetch Archived Resumes
    query_archived = f"'{folder_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and name = 'Archived' and trashed = false"
    arch_res = service.files().list(q=query_archived, fields="files(id)").execute()
    arch_folders = arch_res.get('files', [])

    if arch_folders:
        arch_id = arch_folders[0]['id']
        q_arch_files = f"'{arch_id}' in parents and mimeType = 'text/markdown' and trashed = false"
        arch_files = service.files().list(q=q_arch_files, fields="files(id, name)").execute()
        
        for file in arch_files.get('files', []):
            content = service.files().get_media(fileId=file['id']).execute().decode('utf-8')
            resumes.append({"name": file['name'], "content": content, "is_archived": True})
            
    return resumes

def analyze_with_gemini(jd_text, resumes):
    """Sends the context to Gemini 1.5 Flash."""
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
      "top_match": "Filename",
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
    """Entry point for Google Chat."""
    event = request.get_json()
    if not event or 'message' not in event:
        return {"text": "Hello! Send me a Job Description to start."}
    
    jd_text = event['message'].get('text', '')
    
    try:
        drive = get_drive_service()
        resumes = fetch_resumes(drive, RESUME_FOLDER_ID)
        
        if not resumes:
            return {"text": "No markdown resumes found in Drive."}
            
        result = analyze_with_gemini(jd_text, resumes)
        
        if "error" in result:
            return {"text": "Sorry, I couldn't process that analysis."}

        # Format Card for Google Chat
        widgets = [
            {"textParagraph": {"text": f"<b>🏆 Match:</b> {result.get('top_match')}"}},
            {"textParagraph": {"text": f"{result.get('analysis')}"}},
            {"divider": {}},
            {"textParagraph": {"text": "<b>SUGGESTED BULLETS:</b>"}}
        ]
        
        for b in result.get('bullets', []):
            widgets.append({"textParagraph": {"text": f"<b>{b['section']}:</b><br>{b['text']}"}})

        return {
            "cardsV2": [{
                "cardId": "unique-card-id",
                "card": {
                    "header": {"title": "JD Screener Resume Picker Bot"},
                    "sections": [{"widgets": widgets}]
                }
            }]
        }

    except Exception as e:
        logger.error(e)
        return {"text": f"System Error: {str(e)}"}