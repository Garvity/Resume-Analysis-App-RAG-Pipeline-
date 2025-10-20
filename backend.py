import os
import re
import numpy as np
from PyPDF2 import PdfReader
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
from transformers import pipeline
import uvicorn
import traceback
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("HF_TOKEN")

app = FastAPI()

# CORS middleware to allow requests from Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to your Streamlit port if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
    )

# Load embeddings model and both vector stores once
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
job_vectorstore = FAISS.load_local(
    "vector_store/job_faiss", embeddings, allow_dangerous_deserialization=True
)
resume_vectorstore = FAISS.load_local(
    "vector_store/resume_faiss", embeddings, allow_dangerous_deserialization=True
)

#  Extract text from uploaded PDF resume
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Clean the text data
def clean_text(text):
    """Cleans text by converting to lowercase, removing special characters,
    and handling whitespace."""
    try:
        text = str(text)
    except Exception as e:
        print(f"Error converting text to string: {e}")
        return text
    text = text.lower()
    text = re.sub(r"[^\x00-\x7f]", r"", text)
    text = re.sub(r"\t", r"", text).strip()
    text = re.sub(r"(\n|\r)+", r"\n", text).strip()
    text = re.sub(r" +", r" ", text).strip()
    return text


def get_llm_response(api_key, prompt, model="mistralai/Mistral-7B-Instruct-v0.2"):
    """
    Sends a prompt to Hugging Face Inference API and returns the generated text.
    """
    client = InferenceClient(provider= "featherless-ai", token=api_key)
    
    try:
        completion = client.chat.completions.create(
            model="mistralai/Mistral-7B-Instruct-v0.2",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )
        response = completion.choices[0].message.content.strip()
        return response

    except Exception as e:
        return f"❌ Error generating response: {str(e)}"

# Page 1: Resume Details
@app.post("/resume_details")
async def resume_details(file: UploadFile = File(...), api_key: str = Form(...)):
    resume_text = extract_text_from_pdf(file.file)
    resume_text = clean_text(resume_text)

    prompt = f"""
    You are a resume parser. Extract and label ONLY these sections from the resume text below:
    - Name
    - Email
    - LinkedIn Profile
    - GitHub Profile
    - Portfolio
    - Phone Number
    - Education
    - Skills
    - Experience
    - Projects
    - Achievements
    - Certifications
    - Extra-curricular Activities

    Rules:
    - Output in bullet points: **Section Name**: Extracted content (keep concise).
    - Include ONLY sections present in the text—skip missing ones.
    - If no relevant text, output nothing for that section.
    - Use bold for section names.
    - Do NOT add any extra commentary or information.
    - Ensure the output is clean and easy to read.
    - If the resume is empty or unreadable, respond with "No content found in the resume."
    - The section headings should be in bold and big compared to the rest of the text.

    Resume Text:
    {resume_text[:4000]}  # Slightly longer limit

    Start output directly with bullets—no intro text.
    """

    feedback = get_llm_response(api_key, prompt)
    print(feedback)
    if "Error" in feedback:
        return {"error": feedback}
    return {"llm_feedback": feedback}

# Page 2: Resume Matching
@app.post("/resume_matching")
async def resume_matching(
    file: UploadFile = File(...), 
    job_description: str = Form(...), 
    api_key: str = Form(...)
):
    resume_text = extract_text_from_pdf(file.file)
    resume_text = clean_text(resume_text)

    prompt = f"""
    Job Description:
    {job_description}

    Resume:
    {resume_text[:3000]}

    Please do the following:
    1. Provide a numeric match score (0-100) based on skills, experience, and qualifications.
    2. Explain in bullet points which skills, experiences, or qualifications are missing.
    3. Highlight areas where the candidate is weaker.
    4. Highlight any areas where the candidate is stronger or has relevant expertise.
    5. Suggest specific improvements or additions to the resume that would increase the match score.
    6. The response should be concise and professional.
    7. The headings should be in bold and big compared to the rest of the text.

    Keep the explanation clear and easy to read.
    """
    feedback = get_llm_response(api_key, prompt)
    if "Error" in feedback:
        return {"error": feedback}
    return {"llm_feedback": feedback}

# Page 3: Chat with Resume and Job Description
@app.post("/chat_with_resume")
async def chat_with_resume(
    query: str = Form(...),
    api_key: str = Form(...),
    job_description: str = Form(...),
    file: UploadFile = File(...)
):  
    resume_text = extract_text_from_pdf(file.file)
    resume_text = clean_text(resume_text)
    resume_chunks = splitter.split_text(resume_text)
    job_description_chunks = splitter.split_text(job_description)

    # Embed the query
    query_embedding = embeddings.embed_query(query) if query.strip() else np.zeros(embeddings.embedding_dim)  # Fallback zero vector

    # Safe avg embeddings with fallback
    avg_resume_embedding = np.mean([embeddings.embed_query(chunk) for chunk in resume_chunks], axis=0) if resume_chunks else np.zeros(embeddings.embedding_dim)
    avg_job_embedding = np.mean([embeddings.embed_query(chunk) for chunk in job_description_chunks], axis=0) if job_description_chunks else np.zeros(embeddings.embedding_dim)

    #  Perform similarity search on both stores
    top_jobs = job_vectorstore.similarity_search_by_vector(avg_resume_embedding, k=3)
    top_resumes = resume_vectorstore.similarity_search_by_vector(avg_job_embedding, k=3)
    job_results = job_vectorstore.similarity_search_by_vector(query_embedding, k=3)
    resume_results = resume_vectorstore.similarity_search_by_vector(query_embedding, k=3)

    #  Combine retrieved context
    job_context = "\n\n".join([doc.page_content for doc in job_results])
    resume_context = "\n\n".join([doc.page_content for doc in resume_results])
    top_jobs_context = "\n\n".join([doc.page_content for doc in top_jobs])
    top_resumes_context = "\n\n".join([doc.page_content for doc in top_resumes])

    #  Create the LLM prompt
    prompt = f"""
    You are an intelligent assistant helping a user understand resume-job fit.

    User's Query: {query}

    --- Uploaded Resume ---
    {resume_text[:3000]}

    --- Job Description Context ---
    {job_context}

    --- Resume Context ---
    {resume_context}

    --- Additional Relevant Matches ---
    Jobs: {top_jobs_context}
    Resumes: {top_resumes_context}

    Based on this data, provide a concise, professional answer.
    When asked about candidate whose resume is provided, refer to them as "the candidate".
    Focus on resume-job fit, skills, experience, and qualifications.
    """
    feedback = get_llm_response(api_key, prompt)
    if "Error" in feedback:
        return {"error": feedback}
    return {"llm_feedback": feedback}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=port)
    
# --- IGNORE ---
# # Code to create and save vector stores (run once)
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# vectorstore = FAISS.load_local("vector_store/job_faiss", embeddings, allow_dangerous_deserialization=True)
# resume_docs = [Document(page_content=text, metadata={"source": file.filename}) for text in chunks]
# vectorstore.add_documents(resume_docs)
# vectorstore.save_local("vector_store/job_faiss")
