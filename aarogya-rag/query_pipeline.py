import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
from uuid import UUID
import numpy as np

load_dotenv()

# Config
DB_URL = os.getenv("DATABASE_URL")
engine = create_engine(DB_URL)
Session = sessionmaker(bind=engine)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embedding_model = SentenceTransformer("abhinand/MedEmbed-base-v0.1")
llm_ollama = OllamaLLM(model="llama2")

def analyze_query_for_params(query: str, patient_id: str):
    """Use GPT-4o-mini to extract required params (e.g., 'Glucose')."""
    system_prompt = """
    Analyze the user's health question and identify required lab parameters (e.g., 'Glucose, SGOT').
    Output only a comma-separated list of parameter names.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Patient ID: {patient_id}. Question: {query}"}
        ],
        temperature=0.0
    )
    params_str = response.choices[0].message.content.strip()
    params = [p.strip() for p in params_str.split(",") if p.strip()]
    return params if params else []  # If none, proceed without

def fetch_params_from_db(patient_id: str, params: list):
    """Fetch param values from DB by patient_id and param names."""
    session = Session()
    try:
        # First, get report_ids for patient
        result = session.execute(text("""
            SELECT id FROM lab_reports WHERE patient_id = :patient_id
        """), {"patient_id": patient_id}).fetchall()
        report_ids = [row[0] for row in result]

        if not report_ids:
            return {}

        # Fetch params
        placeholders = ",".join([":report_id_%d" % i for i in range(len(report_ids))])
        param_query = text(f"""
            SELECT parameter, value FROM lab_params 
            WHERE report_id IN ({placeholders})
        """)
        result = session.execute(param_query, {f"report_id_{i}": rid for i, rid in enumerate(report_ids)}).fetchall()
        fetched = {row[0]: row[1] for row in result}
        return {p: fetched.get(p, "Not found") for p in params}
    finally:
        session.close()

def enhance_query(query: str, fetched_params: dict):
    """Enhance query with fetched params."""
    if fetched_params:
        params_str = ", ".join([f"{k}: {v}" for k, v in fetched_params.items()])
        return f"{query} (Contextual lab values: {params_str})"
    return query

def retrieve_chunks(query: str, patient_id: str, k=3):
    """Embed query, retrieve top-k similar chunks via pgvector cosine search."""
    session = Session()
    try:
        query_embedding = embedding_model.encode(query).tolist()

        # Vector search query
        search_query = text("""
            SELECT chunk_text, 1 - (embedding <=> :query_emb::vector) AS similarity
            FROM lab_chunks lc
            JOIN lab_reports lr ON lc.report_id = lr.id
            WHERE lr.patient_id = :patient_id
            ORDER BY embedding <=> :query_emb::vector
            LIMIT :k
        """)
        result = session.execute(search_query, {
            "query_emb": query_embedding,
            "patient_id": patient_id,
            "k": k
        }).fetchall()

        chunks = [{"text": row[0], "similarity": row[1]} for row in result]
        return [c["text"] for c in chunks]
    finally:
        session.close()

def generate_answer(query: str, chunks: list):
    """Generate answer using Ollama RAG."""
    context = "\n\n".join(chunks)
    prompt = f"""
    Answer the question based ONLY on the provided lab context.
    CONTEXT: {context}
    QUESTION: {query}
    ANSWER:
    """
    answer = llm_ollama.invoke(prompt)
    return answer

# Run the pipeline (example)
if __name__ == "__main__":
    patient_id = "your_patient_uuid_here"  # From extraction or input
    query = "My SGOT 54 and SGPT 64 are high—what could cause this?"
    
    params = analyze_query_for_params(query, patient_id)
    print(f"Identified params: {params}")
    
    fetched = fetch_params_from_db(patient_id, params)
    print(f"Fetched values: {fetched}")
    
    enhanced = enhance_query(query, fetched)
    print(f"Enhanced query: {enhanced}")
    
    chunks = retrieve_chunks(enhanced, patient_id)
    print(f"Retrieved {len(chunks)} chunks")
    
    answer = generate_answer(enhanced, chunks)
    print(f"Answer: {answer}")