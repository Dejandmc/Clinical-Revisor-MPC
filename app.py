import streamlit as st
import os
from PIL import Image
from pypdf import PdfReader
from main import app  # Importing the compiled LangGraph from main.py

st.set_page_config(page_title="Medical AI Auditor", page_icon="🛡️", layout="wide")

st.title("🛡️ Clinical Revisor MCP V1.1")
st.markdown("### Agentic Workflow for Medical Safety")

# --- SIDEBAR FOR SETTINGS ---
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info("Model: Gemini 3.1 Flash Lite")
    st.write("Agents: Vision, Research, Writer, Critic, Masterpiece")
    if st.button("🗑️ Clear Results"):
        st.session_state.clear()
        st.rerun()

# --- INPUT DATA ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📂 Upload Documents")
    pdf_file = st.file_uploader("Medical Record (PDF)", type="pdf")
    img_file = st.file_uploader("Laboratory Findings (JPG)", type=["jpg", "jpeg", "png"])

    if st.button("🚀 START ANALYSIS", use_container_width=True):
        if pdf_file and img_file:
            with st.status("🧬 Agents are processing medical history...", expanded=True) as status:
                
                # 1. Data Preparation
                # Reading PDF
                reader = PdfReader(pdf_file)
                pdf_text = "".join([p.extract_text() for p in reader.pages])
                
                # Saving image for Vision Node to find it
                # Because your main.py looks for "patient_lab_result.jpg"
                img = Image.open(img_file)
                img.save("patient_lab_result.jpg") 
                
                # Loading rules and lessons (if they exist)
                def load_local_txt(fn):
                    try:
                        with open(fn, "r", encoding="utf-8") as f: return f.read()
                    except: return ""

                # 2. Initial State (Same as in main.py)
                initial_state = {
                    "pdf_data": pdf_text, 
                    "vision_description": "", 
                    "research_data": "", 
                    "draft": "", 
                    "insta_post": "", 
                    "critic_feedback": "", 
                    "iteration_count": 0, 
                    "total_tokens": 0, 
                    "estimated_cost": 0.0, 
                    "voice_script": "", 
                    "english_version": "",
                    "master_rules": load_local_txt("clinical_rules.txt"), 
                    "lessons_learned": load_local_txt("lessons_learned.txt")
                }

                # 3. Executing LangGraph
                # NOTE: This will execute all nodes. 
                # For the demo, it goes automatically to masterpiece.
                final_output = app.invoke(initial_state)
                
                st.session_state['final_output'] = final_output
                status.update(label="✅ Review is complete!", state="complete")
        else:
            st.error("⚠️ You must upload both PDF and Image!")

# --- DISPLAY RESULTS ---
with col2:
    st.subheader("📊 Reports and Safety")
    
    if 'final_output' in st.session_state:
        out = st.session_state['final_output']
        
        # Price Indicator
        st.metric("Estimated processing cost", f"${out['estimated_cost']:.4f}")

        # Tabs for different reports
        tab1, tab2, tab3 = st.tabs(["📋 Clinical Summary", "🌍 Global Report", "📢 Patient Education"])
        
        with tab1:
            st.text_area("Physician Report:", out['draft'], height=350)
        
        with tab2:
            st.text_area("English Medical Version:", out['english_version'], height=350)
            
        with tab3:
            st.markdown(out['insta_post'])
            
        # Button to download the main report
        st.download_button("💾 Download EN Report", out['draft'], file_name="Summary_EN.txt")
    else:
        st.info("Upload the files and click 'Start Analysis' to see the results.")

st.divider()
st.caption("Clinical Revisor MCP v1.1 | Powered by Gemini 3.1 Flash Lite")
