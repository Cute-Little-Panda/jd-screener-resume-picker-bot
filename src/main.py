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

# --- Vertex AI SDK ---
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, HarmCategory, HarmBlockThreshold

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
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-1.5-pro-002")  # Changed to more stable model

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log configuration on startup
logger.info(f"=== Configuration ===")
logger.info(f"PROJECT_ID: {PROJECT_ID}")
logger.info(f"REGION: {REGION}")
logger.info(f"MODEL_NAME: {MODEL_NAME}")
logger.info(f"SHEET_ID: {SHEET_ID[:10]}..." if SHEET_ID else "SHEET_ID: Not set")

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

model = None
sheets_service = None

def initialize_genai():
    """Initialize Vertex AI with proper authentication"""
    try:
        # Get project ID from environment
        project_to_use = PROJECT_ID
        
        if not project_to_use:
            raise ValueError("No project ID found. Set GCP_PROJECT_ID environment variable.")
        
        # Initialize Vertex AI with project and location
        # Cloud Run automatically handles service account authentication
        vertexai.init(project=project_to_use, location=REGION)
        
        logger.info(f"✓ Vertex AI initialized successfully")
        logger.info(f"  Project: {project_to_use}")
        logger.info(f"  Region: {REGION}")
        logger.info(f"  Model: {MODEL_NAME}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Failed to initialize Vertex AI: {e}")
        logger.exception("Full traceback:")
        return False

def get_model():
    global model
    if model is None:
        if not initialize_genai():
            raise RuntimeError("Failed to initialize GenAI")
        
        system_instruction = """You are a ruthless technical screener and resume auditor.

CORE PRINCIPLES:
- Be harsh, objective, and data-driven
- Only reward concrete evidence with metrics
- Vague claims = automatic failure
- Calculate dates precisely using the current date provided
- Cross-reference all resume versions in the pool
- Follow the exact output format requested"""

        try:
            # Create Vertex AI GenerativeModel
            logger.info(f"Creating model: {MODEL_NAME}")
            model = GenerativeModel(
                model_name=MODEL_NAME,
                system_instruction=system_instruction,
            )
            logger.info(f"✓ Model created successfully: {MODEL_NAME}")
        except Exception as e:
            logger.error(f"✗ Failed to create model: {e}")
            logger.exception("Full traceback:")
            raise
    
    return model

def get_sheets_service():
    global sheets_service
    if sheets_service is None:
        creds, _ = google.auth.default()
        sheets_service = build("sheets", "v4", credentials=creds)
    return sheets_service

def verify_firebase_token(request):
    """Verify Firebase ID token from Authorization header"""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        logger.warning("Auth: Missing or invalid Bearer header")
        return None
    
    token = auth_header.split("Bearer ")[1]
    try:
        decoded_token = auth.verify_id_token(token)
        logger.info(f"Auth successful for user: {decoded_token.get('uid')}")
        return decoded_token
    except Exception as e:
        logger.warning(f"Auth: Token verification failed: {e}")
        return None

def fetch_resumes_from_sheet(service):
    """Fetch all resumes from Google Sheet"""
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
    """Analyze resumes against JD using Gemini"""
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
        generation_config = GenerationConfig(
            temperature=0.2,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
        )

        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        logger.info("Generating content with Gemini...")
        response = model_instance.generate_content(
            full_prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        logger.info("Response received, processing...")
        
        if not response.candidates:
            logger.error("No candidates in response")
            return "Error: No response generated. The content may have been filtered."
        
        if hasattr(response, 'prompt_feedback'):
            block_reason = getattr(response.prompt_feedback, 'block_reason', None)
            if block_reason and block_reason != 0:
                logger.error(f"Response blocked: {block_reason}")
                return f"Error: Response was blocked. Reason: {block_reason}"

        try:
            result_text = response.text
            logger.info(f"✓ Generated response length: {len(result_text)} characters")
            return result_text
        except ValueError as e:
            logger.warning(f"Could not get response.text: {e}")
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    text_parts = []
                    for part in candidate.content.parts:
                        if hasattr(part, 'text'):
                            text_parts.append(part.text)
                    if text_parts:
                        result = "\n".join(text_parts)
                        logger.info(f"✓ Extracted text from parts: {len(result)} characters")
                        return result
            
            logger.error("Could not extract any text from response")
            return "Error: Could not extract text from response."

    except Exception as e:
        logger.exception("Error during Gemini generation")
        error_msg = str(e)
        
        # Detailed error analysis
        if "403" in error_msg or "permission" in error_msg.lower():
            return (
                f"Error: Permission denied\n\n"
                f"Details: {error_msg}\n\n"
                f"Configuration:\n"
                f"- Project: {PROJECT_ID}\n"
                f"- Region: {REGION}\n"
                f"- Model: {MODEL_NAME}\n\n"
                f"Your service account has the right permissions. Try:\n"
                f"1. Wait 10-15 minutes for IAM propagation\n"
                f"2. Check if Vertex AI API is enabled:\n"
                f"   https://console.cloud.google.com/apis/library/aiplatform.googleapis.com?project={PROJECT_ID}\n"
                f"3. Try redeploying the Cloud Function"
            )
        elif "quota" in error_msg.lower():
            return "Error: API quota exceeded. Please check your quota limits in GCP Console."
        elif "not found" in error_msg.lower() or "404" in error_msg:
            return (
                f"Error: Model or endpoint not found\n\n"
                f"Current config: {MODEL_NAME} in {REGION}\n\n"
                f"Try these combinations:\n"
                f"- MODEL_NAME=gemini-1.5-pro-002, REGION=us-central1\n"
                f"- MODEL_NAME=gemini-1.5-flash-002, REGION=us-central1\n"
                f"- MODEL_NAME=gemini-2.0-flash-exp, REGION=us-central1"
            )
        else:
            return f"Error: {error_msg}"

@functions_framework.http
def handle_chat(request):
    """Main HTTP handler for resume screening API"""
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
    }
    
    # Handle CORS preflight
    if request.method == "OPTIONS":
        return ("", 204, headers)
    
    # Only accept POST requests
    if request.method != "POST":
        return (jsonify({"error": "Method not allowed. Use POST."}), 405, headers)
    
    try:
        # Verify Firebase authentication
        user = verify_firebase_token(request)
        if not user:
            logger.warning("Unauthorized request attempt")
            return (jsonify({"error": "Unauthorized. Invalid or missing Firebase token."}), 401, headers)
        
        # Parse request body
        data = request.get_json(silent=True)
        if not data:
            return (jsonify({"error": "Invalid JSON in request body"}), 400, headers)
        
        # Extract JD text
        jd_text = data.get("message", {}).get("text", "") or data.get("jd", "")
        
        if not jd_text:
            return (jsonify({"error": "No job description provided. Send 'jd' field in request body."}), 400, headers)
        
        logger.info(f"Request from user {user.get('uid')}: JD length {len(jd_text)} chars")
        
        # Fetch resumes from Google Sheet
        svc = get_sheets_service()
        resumes = fetch_resumes_from_sheet(svc)
        
        if not resumes:
            return (jsonify({
                "error": "No resumes found in Google Sheet. Check SHEET_ID and SHEET_RANGE configuration."
            }), 500, headers)
        
        # Analyze with Gemini
        logger.info(f"Processing JD against {len(resumes)} resumes")
        markdown_result = analyze_with_gemini(jd_text, resumes)
        
        # Return response
        response_data = {
            "markdown": markdown_result,
            "resume_count": len(resumes),
            "model": MODEL_NAME,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"✓ Successfully processed request for user {user.get('uid')}")
        return (jsonify(response_data), 200, headers)
        
    except Exception as e:
        logger.exception("System error in request handler")
        return (jsonify({
            "error": f"System error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500, headers)