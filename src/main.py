import json
import logging
import os
from datetime import datetime

# --- Firebase & Google Cloud Imports ---
import firebase_admin
import functions_framework
import google.auth
from firebase_admin import auth
from flask import jsonify
from googleapiclient.discovery import build

# --- GenAI SDK ---
import google.generativeai as genai

# Initialize Firebase Admin
try:
    firebase_admin.get_app()
except ValueError:
    firebase_admin.initialize_app()

# --- CONFIGURATION ---
SHEET_ID = os.environ.get("SHEET_ID")
SHEET_RANGE = os.environ.get("SHEET_RANGE", "Sheet1!A:D")
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
REGION = os.environ.get("REGION", "us-central1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-1.5-pro")

PROMPT_TEMPLATE = """
**ROLE:** Ruthless Technical Screener & Resume Auditor.
**INPUT JD:** {jd_text}
**RESUME POOL:** {context_str}

**CURRENT DATE:** {current_date}

**MINDSET:**
You are a skeptical, high-bar technical recruiter at a FAANG-level company. You do not offer praise for "participation." You only care about exact matches, verifiable metrics, and specific evidence. If a resume is vague, assume the candidate does not have the skill. If a resume is "promising" but misses key keywords, it is a failure. Be objective, harsh, and direct. Avoid words like "impressive," "strong," or "solid" unless the evidence is undeniable (top 1% percentile).

**IMPORTANT:** For any roles marked as "Present" or "Current", calculate the duration from the start date to {current_date}. Show your calculation clearly.

**GOAL:**
1.  **Select the Survivor:** detailedly scan the `RESUME POOL` and select the single resume that survives the initial filter against the `INPUT JD`.
    * *Note: 'Archived' files are irrelevant unless they are a 95%+ match where the active ones fail.*
2.  **The Tear-Down (Evaluation):** Conduct a forensic audit of the selected resume against the JD. Assign a strict percentage match.
    * *Scoring Rule:* Mere mention of a skill = 10%. Usage in a project = 50%. Usage with quantitative impact in a professional setting = 100%.
3.  **Bridge the Gap (Data Mining):**
    * Identify the critical flaws/missing skills in the selected resume.
    * **SEARCH:** Look through the *entire* `RESUME POOL` (other versions/files) to see if the candidate has mentioned this missing skill elsewhere.
    * **INTEGRATE:** If found in another file, draft a bullet point using that data.
    * **FABRICATE:** If *not* found in any file, create a hypothetical "NEW SUGGESTION" bullet point that describes what a successful candidate *would* have written.

**CONSTRAINTS:**
-   **TONE:** Clinical, cold, and objective. No sugar-coating. If the resume is bad, say it.
-   **BULLET POINT FORMAT:** Use the "Google XYZ Formula": "Accomplished [X] as measured by [Y], by doing [Z]."
-   **OUTPUT:** Markdown ONLY.

**OUTPUT FORMAT:**

# [Exact Name and Path of Selected Resume]

## 1. Executive Summary (The Verdict)
[3-4 sentences. State clearly why this resume was picked over the others, but focus on why it is still imperfect. Explicitly state the biggest red flag that would cause a rejection in an interview.]

## 2. Forensic Match Evaluation
[Comparison Table. Be strict with scoring.]

| JD Requirement | Evidence in Resume | Match Score (%) | Brutal Analysis / Discrepancy |
| :--- | :--- | :---: | :--- |
| [e.g., 3+ Years Elite SWE Exp] | [e.g., Software Engineer II (3.5 yrs)] | [e.g., 100%] | [Pass.] |
| [e.g., Coding Agents] | [e.g., "Code Remediation Agent"] | [e.g., 60%] | [Academic project only. No professional production usage. Weak evidence.] |
| [e.g., Complex DB Schema] | [e.g., None.] | [e.g., 0%] | **CRITICAL FAILURE:** Candidate lists SQL but zero evidence of designing schemas from scratch. |
| ... | ... | ... | ... |

**Weighted Match Score:** [Calculated Average %]

## 3. The Risk Assessment
[One paragraph explaining exactly why a hiring manager might reject this candidate based on the current resume. Do not mitigate the risk; simply state it.]

## 4. Remediation Plan (Bridging Gaps)
[Generate 3-5 quantitative, high-impact bullet points to fix the <80% rows. STRICTLY follow the data source logic below.]

* **Target Section:** [e.g., Professional Experience - Company X]
    * **Source:** [e.g., Found in 'Resume_Vamsi_Backend.pdf']
    * **Suggestion:** [Drafted XYZ Bullet]

* **Target Section:** [e.g., Projects]
    * **Source:** [e.g., NEW SUGGESTION (Data not found in pool)]
    * **Suggestion:** [Drafted XYZ Bullet]
"""

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
sheets_service = None

def initialize_genai():
    """Initialize GenAI with Cloud credentials"""
    try:
        # Get default credentials for Cloud Run
        credentials, project = google.auth.default()
        
        # Configure genai to use Vertex AI
        genai.configure(
            credentials=credentials,
            project=PROJECT_ID,
            location=REGION
        )
        logger.info(f"GenAI configured for project: {PROJECT_ID}, region: {REGION}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize GenAI: {e}")
        return False

def get_model():
    global model
    if model is None:
        initialize_genai()
        
        system_instruction = """You are a ruthless technical screener and resume auditor.

CORE PRINCIPLES:
- Be harsh, objective, and data-driven
- Only reward concrete evidence with metrics
- Vague claims = automatic failure
- Calculate dates precisely using the current date provided
- Cross-reference all resume versions in the pool
- Follow the exact output format requested"""

        # Create model with tools
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=system_instruction,
            tools='google_search_retrieval',  # Enable Google Search
        )
        logger.info(f"Model initialized: {MODEL_NAME} with Google Search")
    return model

def get_sheets_service():
    global sheets_service
    if sheets_service is None:
        creds, _ = google.auth.default()
        sheets_service = build("sheets", "v4", credentials=creds)
    return sheets_service

def verify_firebase_token(request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.warning("Auth: Missing or invalid Bearer header")
        return None
    token = auth_header.split("Bearer ")[1]
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except Exception as e:
        logger.warning(f"Auth: Token verification failed: {e}")
        return None

def fetch_resumes_from_sheet(service):
    try:
        if not SHEET_ID:
            logger.error("SHEET_ID is missing")
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
            resumes.append({
                "name": name,
                "content": content,
                "is_archived": is_archived,
                "path": path_url
            })
        logger.info(f"Fetched {len(resumes)} resumes from sheet")
        return resumes
    except Exception as e:
        logger.error(f"Error reading sheet: {e}")
        return []

def analyze_with_gemini(jd_text, resumes):
    model_instance = get_model()

    # Build context from resumes
    context_str = ""
    for r in resumes:
        status = "[ARCHIVED]" if r["is_archived"] else "[ACTIVE]"
        context_str += f"\n--- RESUME: {r['name']}, path_to_resume: {r['path']}, {status} ---\n{r['content']}\n"

    # Get current date
    current_date = datetime.now().strftime("%B %d, %Y")
    
    full_prompt = PROMPT_TEMPLATE.format(
        jd_text=jd_text,
        context_str=context_str,
        current_date=current_date
    )
    
    logger.info(f"Prompt length: {len(full_prompt)} characters")
    logger.info(f"Current date used: {current_date}")

    try:
        # Generation config
        generation_config = genai.GenerationConfig(
            temperature=0.2,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
        )

        # Generate content with automatic grounding
        response = model_instance.generate_content(
            full_prompt,
            generation_config=generation_config,
            safety_settings={
                'HARASSMENT': 'BLOCK_NONE',
                'HATE_SPEECH': 'BLOCK_NONE',
                'SEXUALLY_EXPLICIT': 'BLOCK_NONE',
                'DANGEROUS_CONTENT': 'BLOCK_NONE',
            }
        )
        
        # Check if response was blocked
        if not response.candidates:
            logger.error("No candidates in response")
            return "Error: No response generated. The content may have been filtered."
        
        # Check for blocking
        if hasattr(response, 'prompt_feedback'):
            block_reason = getattr(response.prompt_feedback, 'block_reason', None)
            if block_reason and block_reason != 0:  # 0 = BLOCK_REASON_UNSPECIFIED
                logger.error(f"Response blocked: {block_reason}")
                return f"Error: Response was blocked. Reason: {block_reason}"

        # Extract text
        try:
            result_text = response.text
            logger.info(f"Generated response length: {len(result_text)} characters")
            return result_text
        except ValueError as e:
            logger.warning(f"Could not get response.text: {e}")
            # Try extracting from parts
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    text_parts = []
                    for part in candidate.content.parts:
                        if hasattr(part, 'text'):
                            text_parts.append(part.text)
                    if text_parts:
                        return "\n".join(text_parts)
            
            return "Error: Could not extract text from response."

    except Exception as e:
        logger.exception("Error during Gemini generation")
        error_msg = str(e)
        
        # Provide helpful error messages
        if "quota" in error_msg.lower():
            return "Error: API quota exceeded. Please try again later or check your quota settings."
        elif "permission" in error_msg.lower():
            return "Error: Permission denied. Please check your GCP project settings and API enablement."
        elif "not found" in error_msg.lower():
            return f"Error: Model '{MODEL_NAME}' not found. Please verify the model name."
        else:
            return f"Error generating content: {error_msg}"

# --- HTML TEMPLATES ---
HTML_FORM = """
<!DOCTYPE html>
<html>
<head>
    <title>JD Screener Bot</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 40px auto;
            background: white;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
            font-size: 32px;
        }
        .subtitle {
            color: #666;
            margin-bottom: 30px;
            font-size: 16px;
        }
        .info {
            background: #e7f3ff;
            padding: 16px;
            border-radius: 8px;
            margin-bottom: 24px;
            border-left: 4px solid #2196F3;
        }
        .info strong { color: #1976D2; }
        label {
            font-weight: 600;
            color: #444;
            display: block;
            margin-bottom: 10px;
            font-size: 15px;
        }
        textarea { 
            width: 100%; 
            height: 300px; 
            padding: 16px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 14px;
            border: 2px solid #ddd;
            border-radius: 8px;
            resize: vertical;
            transition: border-color 0.3s;
            line-height: 1.6;
        }
        textarea:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        button { 
            padding: 14px 32px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            margin-top: 20px;
            transition: transform 0.2s, box-shadow 0.2s;
            width: 100%;
        }
        button:hover { 
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        button:active {
            transform: translateY(0);
        }
        .emoji { font-size: 24px; margin-right: 8px; }
        @media (max-width: 600px) {
            .container { padding: 24px; }
            h1 { font-size: 24px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1><span class="emoji">🎯</span>Resume Screener AI</h1>
        <p class="subtitle">Powered by Gemini 1.5 Pro with Google Search</p>
        
        <div class="info">
            <strong>📋 How it works:</strong><br>
            Paste your job description below and our AI will ruthlessly screen all resumes from the database, 
            providing a detailed technical evaluation with match scores and improvement suggestions.
        </div>
        
        <form action="/" method="post">
            <label for="jd">📄 Job Description:</label>
            <textarea 
                id="jd" 
                name="jd" 
                required 
                placeholder="Paste the complete job description here...

Example:
Senior Software Engineer - AI/ML
Requirements:
- 5+ years of software engineering experience
- Strong background in Python and machine learning
- Experience with LLMs and vector databases
- Cloud platform experience (AWS/GCP)
..."></textarea>
            <button type="submit">🔍 Analyze Resumes</button>
        </form>
    </div>
</body>
</html>
"""

@functions_framework.http
def handle_chat(request):
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
    }
    
    if request.method == "OPTIONS":
        return ("", 204, headers)
    
    if request.method == "GET":
        return (HTML_FORM, 200, headers)
    
    if request.method == "POST":
        try:
            # Optional: Enforce auth if needed
            # user = verify_firebase_token(request)
            # if not user:
            #     return (jsonify({"error": "Unauthorized"}), 401, headers)
            
            is_json = request.content_type and "application/json" in request.content_type
            if is_json:
                data = request.get_json(silent=True) or {}
                jd_text = data.get("message", {}).get("text", "") or data.get("jd", "")
            else:
                jd_text = request.form.get("jd", "")

            if not jd_text:
                error_response = "Error: No job description provided."
                if is_json:
                    return (jsonify({"error": error_response}), 400, headers)
                return (error_response, 400, headers)

            svc = get_sheets_service()
            resumes = fetch_resumes_from_sheet(svc)
            if not resumes:
                error_response = "Error: No resumes found in Google Sheet. Please check SHEET_ID and SHEET_RANGE."
                if is_json:
                    return (jsonify({"error": error_response}), 500, headers)
                return (error_response, 500, headers)

            logger.info(f"Processing JD (length: {len(jd_text)}) against {len(resumes)} resumes")
            markdown_result = analyze_with_gemini(jd_text, resumes)

            if is_json:
                return (jsonify({"markdown": markdown_result, "resume_count": len(resumes)}), 200, headers)
            else:
                # HTML output with syntax highlighting
                html_output = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Analysis Result</title>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <style>
                        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
                        body {{ 
                            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                            background: #f5f7fa;
                            padding: 20px;
                        }}
                        .container {{
                            max-width: 1200px;
                            margin: 0 auto;
                            background: white;
                            padding: 40px;
                            border-radius: 12px;
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                        }}
                        h1 {{
                            color: #333;
                            margin-bottom: 20px;
                            padding-bottom: 15px;
                            border-bottom: 3px solid #667eea;
                        }}
                        pre {{ 
                            white-space: pre-wrap;
                            background: #f8f9fa;
                            padding: 24px;
                            border-radius: 8px;
                            border-left: 4px solid #667eea;
                            overflow-x: auto;
                            line-height: 1.6;
                            font-family: 'Consolas', 'Monaco', monospace;
                            font-size: 14px;
                            color: #333;
                        }}
                        .actions {{
                            margin-top: 24px;
                            display: flex;
                            gap: 12px;
                        }}
                        a, button {{ 
                            display: inline-block;
                            padding: 12px 24px;
                            color: white;
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            text-decoration: none;
                            border-radius: 8px;
                            font-weight: 600;
                            border: none;
                            cursor: pointer;
                            transition: transform 0.2s;
                        }}
                        a:hover, button:hover {{
                            transform: translateY(-2px);
                            box-shadow: 0 6px 12px rgba(102, 126, 234, 0.3);
                        }}
                        .copy-btn {{
                            background: #28a745;
                        }}
                        .meta {{
                            background: #e7f3ff;
                            padding: 12px;
                            border-radius: 6px;
                            margin-bottom: 20px;
                            font-size: 14px;
                        }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>📊 Resume Analysis Report</h1>
                        <div class="meta">
                            <strong>Generated:</strong> {datetime.now().strftime("%B %d, %Y at %I:%M %p")} | 
                            <strong>Resumes Analyzed:</strong> {len(resumes)}
                        </div>
                        <pre id="result">{markdown_result}</pre>
                        <div class="actions">
                            <a href="/">← New Analysis</a>
                            <button class="copy-btn" onclick="copyToClipboard()">📋 Copy to Clipboard</button>
                        </div>
                    </div>
                    <script>
                        function copyToClipboard() {{
                            const text = document.getElementById('result').textContent;
                            navigator.clipboard.writeText(text).then(() => {{
                                const btn = event.target;
                                const original = btn.textContent;
                                btn.textContent = '✓ Copied!';
                                setTimeout(() => {{ btn.textContent = original; }}, 2000);
                            }});
                        }}
                    </script>
                </body>
                </html>
                """
                return (html_output, 200, headers)

        except Exception as e:
            logger.exception("System Error in request handler")
            error_msg = f"System Error: {str(e)}"
            if is_json:
                return (jsonify({"error": error_msg}), 500, headers)
            
            error_html = f"""
            <html>
            <head>
                <title>Error</title>
                <style>
                    body {{ font-family: sans-serif; padding: 40px; max-width: 800px; margin: 0 auto; }}
                    .error {{ background: #fee; border-left: 4px solid #c00; padding: 20px; border-radius: 4px; }}
                    a {{ display: inline-block; margin-top: 20px; color: #007bff; }}
                </style>
            </head>
            <body>
                <div class="error">
                    <h1>❌ Error</h1>
                    <pre>{error_msg}</pre>
                </div>
                <a href='/'>← Back to Form</a>
            </body>
            </html>
            """
            return (error_html, 500, headers)