🛡️ Clinical Revisor MCP V1.1
Multi-Modal Self-Correcting Agentic RAG System
A sophisticated medical safety auditing system powered by LangGraph and Gemini 3.1 Flash Lite. This agent performs clinical record analysis, visual lab report interpretation, and automated medical research with a built-in safety critic and self-correction loop.

🌟 Key Features
Multi-Modal Analysis: Integrates Vision Node for analyzing medical images (scans/lab results) and PDF/Text processing for patient records.

Self-Correcting Loop: Features a Safety Auditor (Critic) node that evaluates drafts against clinical rules and triggers recursive rewrites if safety standards aren't met.

Agentic RAG: Uses Tavily Search to perform real-time clinical research on drug interactions and medical contraindications.

LLMOps & Evaluation: Built-in tracking for:

Token consumption.

Estimated API costs per run.

Iterative revision logging in evolution_log.txt.

Multi-Output Generation: Produces professional Macedonian clinical summaries, Global English reports, patient voice-over scripts, and educational social media content.

🏗️ System Architecture
The agent operates on a directed acyclic graph (DAG) structure:

Vision Node: Extracts data from images using Gemini Vision.

Research Node: Performs web-retrieval via Tavily for clinical context.

Writer Node: Synthesizes data into a formal clinical summary.

Instagram Node: Adapts technical data for patient-friendly education.

Critic Node: Safety audit based on master_rules.

Router: Determines if the draft is APPROVED or needs a REWRITE (up to 3 iterations).

Masterpiece Node: Finalizes all global and media exports.

🚀 Getting Started
1. Prerequisites
Python 3.10+

Google GenAI API Key

Tavily API Key

2. Installation
Bash
git clone https://github.com/Dejandmc/Clinical-Revisor-MPC.git
cd Clinical-Revisor-MPC
pip install -r requirements.txt
3. Configuration
Create a .env file in the root directory:

Code snippet
GOOGLE_API_KEY=your_gemini_key_here
TAVILY_API_KEY=your_tavily_key_here
4. Required Files
Ensure the following files are present for the RAG system to function:

clinical_rules.txt: Global medical safety guidelines.

lessons_learned.txt: Database of past clinical insights.

patient_record.pdf: The primary patient history.

patient_lab_result.jpg: (Optional) Lab scan for vision analysis.

5. Usage
Bash
python main.py
📊 LLMOps & Safety
Every execution is logged for evaluation. The system calculates cost based on a pricing model of $0.0001 per 1k tokens, ensuring transparency in resource management during agentic iterations.
