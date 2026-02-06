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

# --- NEW: Vertex AI Tool Imports ---
from vertexai.generative_models import (
    GenerativeModel,
    Tool,
    GoogleSearchRetrieval,
    CodeExecution,
    ToolConfig,
)

# Initialize Firebase Admin
try:
    firebase_admin.get_app()
except ValueError:
    firebase_admin.initialize_app()

# --- CONFIGURATION ---
SHEET_ID = os.environ.get("SHEET_ID")
SHEET_RANGE = os.environ.get("SHEET_RANGE", "Sheet1!A:D")
PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
REGION = os.environ.get("REGION")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-1.5-pro-002") # Recommend 1.5 Pro or Flash for tool use

PROMPT_TEMPLATE = """
**ROLE:** Ruthless Technical Screener & Resume Auditor.
**INPUT JD:** {jd_text}
**RESUME POOL:** {context_str}

**MINDSET:**
You are a skeptical, high-bar technical recruiter at a FAANG-level company. You do not offer praise for "participation." You only care about exact matches, verifiable metrics, and specific evidence. If a resume is vague, assume the candidate does not have the skill. If a resume is "promising" but misses key keywords, it is a failure. Be objective, harsh, and direct. Avoid words like "impressive," "strong," or "solid" unless the evidence is undeniable (top 1% percentile).

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
sheets_service = None

def get_model():
    global model
    if model is None:
        vertexai.init(project=PROJECT_ID, location=REGION)
        # Note: Ensure you are using a model version that supports tools (gemini-1.5-pro or flash)
        model = GenerativeModel(MODEL_NAME)
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
    # (Same implementation as before)
    try:
        if not SHEET_ID:
            logger.error("SHEET_ID is missing")
            return []
        sheet = service.spreadsheets()
        result = sheet.values().get(spreadsheetId=SHEET_ID, range=SHEET_RANGE).execute()
        rows = result.get("values", [])
        resumes = []
        for row in rows:
            if len(row) < 2: continue
            name = row[0]
            content = row[1]
            status = row[2] if len(row) > 2 else ""
            path_url = row[3] if len(row) > 3 else "#"
            is_archived = "archived" in status.lower()
            resumes.append({"name": name, "content": content, "is_archived": is_archived, "path": path_url})
        return resumes
    except Exception as e:
        logger.error(f"Error reading sheet: {e}")
        return []

def analyze_with_gemini(jd_text, resumes):
    model_instance = get_model()

    # 1. Define Tools
    # Google Search: For current date grounding
    search_tool = Tool.from_google_search_retrieval(
        google_search_retrieval=GoogleSearchRetrieval()
    )
    
    # Code Execution: For accurate math (years of experience, percentages)
    code_tool = Tool.from_code_execution(
        code_execution=CodeExecution()
    )

    # 2. Tool Config
    # Auto: The model decides when to use which tool
    tool_config = ToolConfig(
        function_calling_config=ToolConfig.FunctionCallingConfig(
            mode=ToolConfig.FunctionCallingConfig.Mode.AUTO,
        )
    )

    context_str = ""
    for r in resumes:
        status = "[ARCHIVED]" if r["is_archived"] else "[ACTIVE]"
        context_str += f"\n--- RESUME: {r['name']}, path_to_resume: {r['path']}, {status} ---\n{r['content']}\n"

    # 3. Prompt Engineering with Tool Instructions
    system_instruction = (
        "SYSTEM INSTRUCTION: \n"
        "1. DATE CHECK: First, use the Google Search tool to find 'current date today'. "
        "Print this date clearly at the top of your response.\n"
        "2. CALCULATION: If you need to calculate years of experience (e.g., Jan 2020 to Present), or any other calculations "
        "use the Code Interpreter (Python) to get the exact duration. Do not guess.\n"
        "3. EVALUATION: Use the fetched date as the baseline for 'Present' roles.\n"
        "4. Use the arsenal of tools, don't assume."
        "---------------------------------------------------\n"
    )

    full_prompt = system_instruction + PROMPT_TEMPLATE.format(jd_text=jd_text, context_str=context_str)
    logger.log("Prompt length: " + len(full_prompt))

    try:
        # 4. Generate with Tools
        response = model_instance.generate_content(
            full_prompt,
            tools=[search_tool, code_tool],
            tool_config=tool_config,
        )
        return response.text
    except Exception as e:
        logger.error(f"AI Error: {e}")
        return f"Error generating content: {str(e)}"

# --- HTML TEMPLATES (Same as before) ---
HTML_FORM = """... (Keep your existing HTML) ..."""
HTML_RAW_OUTPUT = """... (Keep your existing HTML) ..."""

@functions_framework.http
def handle_chat(request):
    # (Keep your existing handle_chat implementation)
    # ...
    # This part remains identical to your previous code
    # ...
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
    }
    if request.method == "OPTIONS":
        return ("", 204, headers)
    if request.method == "GET":
        return (HTML_FORM, 200)
    if request.method == "POST":
        try:
            user = verify_firebase_token(request)
            if not user:
                return (jsonify({"error": "Unauthorized"}), 401, headers)
            
            is_json = request.content_type == "application/json"
            if is_json:
                data = request.get_json(silent=True) or {}
                jd_text = data.get("message", {}).get("text", "") or data.get("jd", "")
            else:
                jd_text = request.form.get("jd", "")

            if not jd_text:
                return ("Error: No JD provided.", 400, headers)

            svc = get_sheets_service()
            resumes = fetch_resumes_from_sheet(svc)
            if not resumes:
                return ("Error: No resumes found.", 500, headers)

            markdown_result = analyze_with_gemini(jd_text, resumes)

            if is_json:
                return (jsonify({"markdown": markdown_result}), 200, headers)
            else:
                # Basic HTML output
                return (f"<html><body><pre>{markdown_result}</pre></body></html>", 200, headers)

        except Exception as e:
            logger.exception("System Error")
            return (f"System Error: {str(e)}", 500, headers)
