import streamlit as st
from PyPDF2 import PdfReader
import requests
import os

# Sidebar for API key input
st.sidebar.title("🔑 API Configuration")
api_key_input = st.sidebar.text_input("Enter your Hugging Face API Key", type="password")
# Fallback to environment token if not provided via sidebar
if api_key_input:
    # Store temporarily in environment variables
    os.environ["HF_TOKEN"] = api_key_input
    st.success("API key is set for this session!")
api_key = os.getenv("HF_TOKEN")

page = st.sidebar.radio("Select Page", ["Resume Details", "Resume Matching", "Chat with Resume and Job Description", "About"],index=0)

def call_backend(endpoint, file, data):
    files = {"file": (file.name, file.getvalue(), "application/pdf")}
    response = requests.post(f"http://localhost:8000/{endpoint}", files=files, data=data)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error: {response.text}")
        return None
    
# App title and description
st.title("📄 Resume Analyzer Pro")
st.markdown("""
Upload your resume below to analyze its content, extract sections, and compare it with job descriptions using AI.  
---
""")



# Job description input


# ------------------------- Page 1: Resume Details -------------------------

if page == "Resume Details":
    st.header("Resume Details Analysis")
    # File uploader (Resume upload)
    st.session_state.uploaded_file = st.file_uploader("📤 Upload your Resume (PDF only)", type=["pdf"])
    if st.session_state.uploaded_file:
        st.success("✅ Resume uploaded successfully!")
    else:
        st.error("Please upload a resume PDF.")
        st.stop()
    
    if api_key:
        st.session_state.api_key = api_key
    else:
        st.error("Please enter your Hugging Face API key in the sidebar.")
        st.stop()
    
    if st.session_state.uploaded_file and st.session_state.api_key:
        if st.button("Analyze Resume"):
            with st.spinner("Analyzing resume..."):
                data = {"api_key": st.session_state.api_key}
                result = call_backend("resume_details", st.session_state.uploaded_file, data)
            if result:
                st.subheader("Extracted Resume Sections")
                feedback = result.get("llm_feedback", "")
                for line in feedback.split("\n"):
                    if line.strip():
                        st.write("•", line.strip())
    else:
        st.info("Please upload a PDF, provide your Hugging Face API key and enter a job description.")
    
    

# ------------------------- Page 2: Resume Matching -------------------------

elif page == "Resume Matching":
    st.header("Resume-Job Matching")
    if not "uploaded_file" in st.session_state:
        st.warning("⚠️ Please upload your resume on the Resume Details page first.")
        st.stop()

    st.session_state.job_description = st.text_area("Enter Job Description", value=st.session_state.get("job_description", ""), height=150)
    
    if "job_description" not in st.session_state:
        st.info(" Please enter a job description.")
        st.stop()

    if "api_key" not in st.session_state:
        st.info("⚠️ Please enter your Hugging Face API key in the sidebar.")
        st.stop()

    if st.session_state.uploaded_file and st.session_state.job_description and st.session_state.api_key:
        if st.button("Match Resume to Job Description"):
            with st.spinner("Matching ..."):
                data = {"api_key": st.session_state.api_key, "job_description": st.session_state.job_description}
                result = call_backend("resume_matching", st.session_state.uploaded_file, data)
            if result:
                feedback = result.get("llm_feedback", "")
                # Example parsing: first line = score, rest = bullet points
                lines = feedback.split("\n")
                if lines:
                    st.subheader("Match Score")
                    st.write(lines[0].strip())
                    st.subheader("Feedback / Missing Skills")
                    for line in lines[1:]:
                        if line.strip():
                            st.write("•", line.strip())
    else:
        st.info("Please upload a PDF, provide your Hugging Face API key and enter a job description.")

# ------------------------- Page 3: Chat with Resume and Job Description -------------------------
elif page == "Chat with Resume and Job Description":
    st.header("💬 Chat with Resume and Job Description")
    st.markdown("Ask questions about your resume or get career advice based on your profile and job description.")
    
    if "uploaded_file" not in st.session_state:
        st.warning("⚠️ Please upload your resume on the Resume Details page first.")
        st.stop()
    
    uploaded_file = st.session_state.uploaded_file
    st.write("File available for chat:", uploaded_file.name)
    
    # Display chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Job description context
    st.markdown("### Job Description Context")
    if "job_description" in st.session_state and st.session_state.job_description:
        st.text_area("Current Job Description:", value=st.session_state.job_description, height=400, disabled=True)
    else:
        st.warning("No job description provided. Upload or enter one on the previous page.")
    
    # Chat interface
    st.markdown("### Ask Questions")
    user_question = st.text_input("Type your question here:", 
                                placeholder="E.g., What skills should I learn? How can I improve my resume? What projects should I build?")
    
    if not user_question.strip():
        st.error("Please enter a query.")
    elif st.button("Ask") and user_question and api_key:
        if "api_key" not in st.session_state:
            st.session_state.api_key = api_key
        with st.spinner("Thinking..."):
            data = {"api_key": api_key, "query": user_question, "job_description": st.session_state.get("job_description", "")}
            result = call_backend("chat_with_resume", uploaded_file, data)
            response = result.get("llm_feedback", "") if result else "No response received."
        
        # Add to chat history
        st.session_state.chat_history.append({
            "question": user_question,
            "response": response
        })
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### Chat History")
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            st.markdown(f"""
                <div style="background-color: #2d3748; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                    <strong>👤 You:</strong> {chat['question']}
                </div>
                <div style="background-color: #2d3748; padding: 1rem; border-radius: 10px; margin: 1rem 0; border-left: 5px solid #1890ff;">
                    <strong>🤖 Assistant:</strong> {chat['response']}
                </div>
                """, unsafe_allow_html=True)

# ------------------------- Page 4: About Page -------------------------
elif page == "About":
    st.header("ℹ️ About Resume Analyzer Pro")
    st.markdown("""
    ### Welcome to Resume Analyzer Pro!
    
    This AI-powered application helps you analyze your resume against job descriptions and provides personalized recommendations to improve your career prospects.
    
    ### Features:
    - **Intelligent Resume Processing**: Upload PDF resumes with section-wise parsing and content visualization
    - **LLM-Powered Extraction**: AI-based extraction of skills, education, projects, work experience, and certifications via the Hugging Face Inference API
    - **Smart Job Matching**: Semantic matching using sentence-transformer embeddings and FAISS
    - **Enhanced Chat Interface**: Context-aware conversations powered by a Retrieval-Augmented Generation (RAG) flow
    - **Structured Content Display**: View clean resume sections with organized information
    - **Comprehensive Analysis**: Deep insights into personal info, education details, work experience, and more
    
    ### Technology Stack:
    - **RAG Pipeline**: Retrieval-Augmented Generation using LangChain text splitting and context assembly
    - **Sentence Transformers**: `sentence-transformers/all-MiniLM-L6-v2` via `HuggingFaceEmbeddings` for semantic embeddings
    - **FAISS**: Local vector stores for efficient similarity search (`vector_store/`)
    - **LLM Inference**: Hugging Face Inference API with `mistralai/Mistral-7B-Instruct-v0.2`
    - **FastAPI**: Backend service exposing analysis and chat endpoints
    - **Streamlit**: Interactive web interface
    - **PyPDF2**: PDF text extraction
    
    ### How It Works:
    1. **Upload & Process**: Upload your PDF resume; text is extracted locally with PyPDF2 and cleaned
    2. **Embedding & Retrieval**: Resume/job text is split into chunks, embedded with a sentence transformer, and searched in local FAISS stores
    3. **Smart Matching**: The LLM evaluates resume vs. job description with retrieved context to generate a score and feedback
    4. **Chat**: Ask questions; the system augments your query with relevant retrieved context and responds via the LLM
    5. **Contextual Advice**: Get personalized recommendations based on retrieved evidence and LLM reasoning
    
    ### Privacy Notice:
    Your resume data is processed locally and stored only for your session. We do not share your personal information with third parties.
    
    ### Contact:
    For support or feedback, please contact: @garvity
    """)

    # Footer
    st.markdown("---")
    st.markdown("Resume Analyzer Pro © 2025 | Powered by AI and Machine Learning")
