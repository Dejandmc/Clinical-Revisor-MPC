import os
from typing import TypedDict, List
from dotenv import load_dotenv
from pypdf import PdfReader
from PIL import Image
from google import genai  # New SDK client
from langgraph.graph import StateGraph, END
from tavily import TavilyClient
import datetime

# --- 1. CONFIGURATION AND LLMOps SETTINGS ---
load_dotenv()

try:
    import langchain
    langchain.debug = False
except (ImportError, AttributeError):
    print("ℹ️ LangChain debug setting skipped.")
# Ова го решава AttributeError: module 'langchain' has no attribute 'debug'
try:
    langchain.debug = False
except AttributeError:
    pass

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# LLMOps Pricing (Flash Lite average $0.0001 per 1k tokens)
COST_PER_1K_TOKENS = 0.0001 

class AgentState(TypedDict):
    pdf_data: str            # Patient Medical Record
    vision_description: str  # Lab results/scans analysis
    research_data: str       # Medical research (PubMed/Tavily)
    draft: str               # Clinical Summary
    insta_post: str          # Patient educational content
    critic_feedback: str     # Safety Audit Feedback
    iteration_count: int
    total_tokens: int
    estimated_cost: float
    voice_script: str        # Patient instructions (Audio script)
    english_version: str     # Global Medical Report
    master_rules: str        # Clinical Safety Rules
    lessons_learned: str     # Past medical insights/experience

def track_usage(response, state: AgentState):
    """Function for detailed LLMOps resource tracking"""
    usage = response.usage_metadata
    state["total_tokens"] += usage.total_token_count
    state["estimated_cost"] += (usage.total_token_count / 1000) * COST_PER_1K_TOKENS
    return state

# --- 2. VISION NODE (Medical Lab/Scan Analysis) ---
def vision_node(state: AgentState):
    print("📸 [V1.1] VISION AGENT: Analyzing medical lab report/scan...")
    try:
        img = Image.open("patient_lab_result.jpg") 
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=[
                "Extract all numerical values and identify abnormalities (high/low). Highlight critical safety issues.", 
                img
            ]
        )
        state = track_usage(response, state)
        return {"vision_description": response.text, "total_tokens": state["total_tokens"], "estimated_cost": state["estimated_cost"]}
    except FileNotFoundError:
        print("⚠️ Warning: patient_lab_result.jpg not found. Skipping Vision.")
        return {"vision_description": "No visual data available."}

# --- 3. RESEARCH NODE (Clinical Knowledge) ---
def research_node(state: AgentState):
    print("🔍 [V1.1] RESEARCH AGENT: Searching for medical contraindications...")
    # Searching for drug-disease interactions based on patient data
    query = f"drug interactions and side effects for clinical context: {state['pdf_data'][:200]}"
    search_result = tavily.search(query=query)
    return {"research_data": str(search_result)}

# --- 4. WRITER NODE (Clinical Reporting) ---
def writer_node(state: AgentState):
    current_iter = state.get('iteration_count', 0) + 1
    print(f"✍️ [V1.1] WRITER: Drafting Clinical Summary (Attempt #{current_iter})...")
    
    feedback_context = f"\nPREVIOUS SAFETY AUDIT NOTES: {state['critic_feedback']}" if state['critic_feedback'] else ""
    
    prompt = f"""Write an official Clinical Summary for a physician in Macedonian language.
    
    CLINICAL RULES:
    {state.get('master_rules', '')}
    
    PAST MEDICAL LESSONS:
    {state.get('lessons_learned', '')}
    
    PATIENT DATA:
    - History: {state['pdf_data']}
    - Lab Results: {state['vision_description']}
    - Research Found: {state['research_data']}
    
    {feedback_context}
    
    Format: Start with 'SAFETY AUDIT VERDICT'. Focus on accuracy and risk prevention. No exclamation marks!"""
    
    response = client.models.generate_content(model="gemini-3.1-flash-lite-preview", contents=prompt)
    state = track_usage(response, state)
    return {"draft": response.text, "iteration_count": current_iter, "total_tokens": state["total_tokens"], "estimated_cost": state["estimated_cost"]}

# --- 5. PATIENT EDUCATION NODE (Social/Infographic) ---
def instagram_node(state: AgentState):
    print("📱 [V1.1] MEDICAL EDUCATOR: Creating patient-friendly info...")
    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=f"Summarize this medical report for a PATIENT in simple ENGLISH. What should they do and what should they avoid? Text: '{state['draft']}'"
        )
        state = track_usage(response, state)
        return {"insta_post": response.text, "total_tokens": state["total_tokens"], "estimated_cost": state["estimated_cost"]}
    except:
        return {"insta_post": "Patient summary unavailable.", "total_tokens": state["total_tokens"], "estimated_cost": state["estimated_cost"]}

# --- 6. CRITIC NODE (The Safety Auditor) ---
def critic_node(state: AgentState):
    print("⚖️ [V1.1] SAFETY AUDITOR: Checking for medical errors...")
    check_prompt = f"Evaluate this clinical summary for SAFETY:\n{state.get('master_rules', '')}\n\nREPORT:\n{state['draft']}\n\nReply ONLY with 'APPROVED' or provide REJECTION reasons."
    
    check_response = client.models.generate_content(model="gemini-3.1-flash-lite-preview", contents=check_prompt)
    
    # Update state usage
    state = track_usage(check_response, state)
    feedback = check_response.text
    
    with open("evolution_log.txt", "a", encoding="utf-8") as log:
        log.write(f"\n--- MEDICAL REVISION #{state['iteration_count']} [{datetime.datetime.now()}] ---\n")
        log.write(f"📊 Tokens: {state['total_tokens']} | 💰 Cost: ${state['estimated_cost']:.4f}\n")
        log.write(f"SAFETY FEEDBACK: {feedback}\n")
        
    return {"critic_feedback": feedback, "total_tokens": state["total_tokens"], "estimated_cost": state["estimated_cost"]}

# --- 7. MASTERPIECE NODE (Final Exports) ---
def masterpiece_node(state: AgentState):
    print("🎙️ [V1.1] VOICE & GLOBAL AGENT: Creating Patient Instructions and Global Report...")
    
    # 1. Voice Script
    vo_prompt = f"Create a professional, calm, and empathetic medical audio script in English for a patient, based on this report: {state['draft']}. Focus on clarity and safety instructions."
    vo_response = client.models.generate_content(model="gemini-3.1-flash-lite-preview", contents=vo_prompt)
    state = track_usage(vo_response, state)
    
    # 2. English Adaptation
    en_prompt = f"Translate and adapt this clinical summary into a professional Global Medical Report in English. Text: {state['draft']}"
    en_response = client.models.generate_content(model="gemini-3.1-flash-lite-preview", contents=en_prompt)
    state = track_usage(en_response, state)
    
    return {
        "voice_script": vo_response.text, 
        "english_version": en_response.text,
        "total_tokens": state["total_tokens"], 
        "estimated_cost": state["estimated_cost"]
    }

# --- 8. EXECUTIVE ROUTING (Streamlit Compatible) ---
def should_continue(state: AgentState):
    # Check if the Safety Auditor provided approval
    if "APPROVED" in state["critic_feedback"].upper():
        print(f"\n✅ CLINICAL SAFETY APPROVED. (Cost: ${state['estimated_cost']:.4f})")
        # Direct bypass of terminal input for Streamlit UI compatibility
        return "final"
    
    # If not approved and we have remaining attempts, trigger a rewrite
    if state["iteration_count"] < 3:
        print(f"⚠️ SAFETY REJECTED. Retrying adjustment {state['iteration_count'] + 1}/3...")
        return "rewrite"
    
    # Finalize after 3 attempts to prevent infinite token usage
    return "final"

# --- 9. GRAPH CONSTRUCTION ---
workflow = StateGraph(AgentState)
workflow.add_node("vision", vision_node); workflow.add_node("research", research_node)
workflow.add_node("writer", writer_node); workflow.add_node("instagram", instagram_node)
workflow.add_node("critic", critic_node); workflow.add_node("masterpiece", masterpiece_node)

workflow.set_entry_point("vision")
workflow.add_edge("vision", "research"); workflow.add_edge("research", "writer")
workflow.add_edge("writer", "instagram"); workflow.add_edge("instagram", "critic")
workflow.add_conditional_edges("critic", should_continue, {"rewrite": "writer", "final": "masterpiece"})
workflow.add_edge("masterpiece", END)
app = workflow.compile()

# --- 10. EXECUTION ---
if __name__ == "__main__":
    print("🛡️ STARTING CLINICAL REVISOR MCP V1.1...")
    
    def load_file(fn):
        try:
            with open(fn, "r", encoding="utf-8") as f: return f.read()
        except: return ""

    rules_text = load_file("clinical_rules.txt")
    lessons_text = load_file("lessons_learned.txt")
    
    try:
        reader = PdfReader("patient_record.pdf")
        pdf_text = "".join([p.extract_text() for p in reader.pages])
    except:
        pdf_text = load_file("patient_record.txt")

    initial_state = {
        "pdf_data": pdf_text, "vision_description": "", "research_data": "", "draft": "", 
        "insta_post": "", "critic_feedback": "", "iteration_count": 0, "total_tokens": 0, 
        "estimated_cost": 0.0, "voice_script": "", "english_version": "",
        "master_rules": rules_text, "lessons_learned": lessons_text
    }
    
    # Invoke Agent
    final_output = app.invoke(initial_state)
    
    # Create output folder and save final files
    os.makedirs("FINAL_MEDICAL_OUTPUT", exist_ok=True)
    
    # 1. Macedonian Clinical Summary
    with open("FINAL_MEDICAL_OUTPUT/Clinical_Summary_MK.txt", "w", encoding="utf-8") as f:
        f.write(final_output["draft"])
        
    # 2. Global English Report
    with open("FINAL_MEDICAL_OUTPUT/Global_Report_EN.txt", "w", encoding="utf-8") as f:
        f.write(final_output["english_version"])
        
    # 3. Patient Voice-over Script
    with open("FINAL_MEDICAL_OUTPUT/Patient_Voice_Guide.txt", "w", encoding="utf-8") as f:
        f.write(final_output["voice_script"])
        
    # 4. Patient Educational Content
    with open("FINAL_MEDICAL_OUTPUT/Patient_Education_Social.txt", "w", encoding="utf-8") as f:
        f.write(final_output["insta_post"])
    
    print(f"\n🏆 CLINICAL AUDIT COMPLETED!")
    print(f"📊 Total tokens used: {final_output['total_tokens']}")
    print(f"💰 Final calculated cost: ${final_output['estimated_cost']:.4f}")
    print("📁 All reports are saved in the 'FINAL_MEDICAL_OUTPUT' folder.")