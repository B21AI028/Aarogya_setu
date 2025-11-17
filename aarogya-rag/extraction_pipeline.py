import os
import json
import argparse
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from uuid import uuid4
import datetime

load_dotenv()

# Config
DB_URL = os.getenv("DATABASE_URL")
API_KEY = os.getenv("GOOGLE_API_KEY")  # Fallback to your provided key

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY must be set in .env or provided.")

engine = create_engine(DB_URL)
Session = sessionmaker(bind=engine)
embedding_model = SentenceTransformer("abhinand/MedEmbed-base-v0.1")  # MedEmbed
llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # Updated to latest Gemini 2.5 Flash
    google_api_key=API_KEY,    # Use the key directly
    temperature=0
)
llm_ollama = OllamaLLM(model="llama2")

def pdf_extractor(pdf_path: str):
    """Extract JSON from PDF using Gemini 2.5 Flash (renamed from extract_from_pdf)."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    full_text = "\n\n".join([page.page_content for page in pages])

    prompt = f"""
    Extract relevant information from the following lab report text and return as JSON with these fields:
    - patient_name (str)
    - patient_id (str, or generate UUID if missing)
    - date_of_collection (str, YYYY-MM-DD)
    - date_of_report (str, YYYY-MM-DD)
    - test_type (str)
    - doctor_name (str)
    - lab_name (str)
    - lab_values (list of dicts: {{"parameter": str, "value": str, "reference_range": str}})
    - abnormal_findings (list of str)
    - overall_interpretation (str)

    LAB REPORT TEXT:
    {full_text}

    Return only valid JSON.
    """

    try:
        response = llm_gemini.invoke(prompt)
        response_text = response.content.strip()

        # Parse JSON (robust)
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            json_text = response_text[start:end].strip()
        elif response_text.startswith("{"):
            json_text = response_text
        else:
            raise ValueError("No valid JSON in response")

        result = json.loads(json_text)
        if not result.get("patient_id"):
            result["patient_id"] = str(uuid4())  # Generate if missing
        return result, full_text
    except Exception as e:
        raise RuntimeError(f"Extraction failed: {e}")

def generate_summary(full_text: str, interpretation: str):
    """Generate lab summary using Ollama."""
    prompt = f"""
    Summarize the following lab report text and interpretation into a concise medical summary (200-400 words).
    Focus on key findings, abnormalities, and clinical implications.
    TEXT: {full_text}
    INTERPRETATION: {interpretation}
    """
    summary = llm_ollama.invoke(prompt)
    return summary

def chunk_summary(summary: str, chunk_size=300, overlap=50):
    """Chunk summary into tokens."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size * 4,  # Approx tokens
        chunk_overlap=overlap * 4,
        separators=["\n\n", "\n", ". "]
    )
    chunks = splitter.create_documents([summary])
    return [chunk.page_content for chunk in chunks]

def store_to_db(extracted: dict, full_text: str, chunks: list):
    """Store to Postgres: reports, params, chunks with embeddings."""
    session = Session()
    try:
        report_id = uuid4()
        created_at = datetime.datetime.now()

        # Store lab_report
        session.execute(text("""
            INSERT INTO lab_reports (id, patient_id, patient_name, date_of_collection, date_of_report,
                                    test_type, doctor_name, lab_name, overall_interpretation, abnormal_findings, raw_text)
            VALUES (:id, :patient_id, :patient_name, :date_of_collection, :date_of_report,
                    :test_type, :doctor_name, :lab_name, :overall_interpretation, :abnormal_findings, :raw_text)
        """), {
            "id": report_id,
            "patient_id": extracted["patient_id"],
            "patient_name": extracted["patient_name"],
            "date_of_collection": extracted.get("date_of_collection"),
            "date_of_report": extracted.get("date_of_report"),
            "test_type": extracted.get("test_type"),
            "doctor_name": extracted.get("doctor_name"),
            "lab_name": extracted.get("lab_name"),
            "overall_interpretation": extracted["overall_interpretation"],
            "abnormal_findings": extracted["abnormal_findings"],
            "raw_text": full_text
        })

        # Store lab_params
        for param in extracted["lab_values"]:
            is_abnormal = param["parameter"] in [f.strip() for af in extracted["abnormal_findings"] for f in af.split()]  # Simple check
            session.execute(text("""
                INSERT INTO lab_params (report_id, parameter, value, reference_range, is_abnormal)
                VALUES (:report_id, :parameter, :value, :reference_range, :is_abnormal)
            """), {
                "report_id": report_id,
                "parameter": param["parameter"],
                "value": param["value"],
                "reference_range": param["reference_range"],
                "is_abnormal": is_abnormal
            })

        # Embed and store chunks
        for i, chunk_text in enumerate(chunks):
            embedding = embedding_model.encode(chunk_text).tolist()
            session.execute(text("""
                INSERT INTO lab_chunks (report_id, chunk_text, embedding, metadata)
                VALUES (:report_id, :chunk_text, :embedding::vector, :metadata)
            """), {
                "report_id": report_id,
                "chunk_text": chunk_text,
                "embedding": embedding,
                "metadata": json.dumps({"chunk_id": i, "summary_type": "lab_summary"})
            })

        session.commit()
        print(f"Stored report {report_id} with {len(extracted['lab_values'])} params and {len(chunks)} chunks.")
        return report_id
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

# Run the pipeline
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and store lab report from PDF")
    parser.add_argument("--pdf", default="uploads/Pragati_1.pdf", help="Path to PDF file")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        raise FileNotFoundError(f"PDF not found: {args.pdf}")

    extracted, full_text = pdf_extractor(args.pdf)
    print(f"Extracted patient: {extracted.get('patient_name', 'N/A')} (ID: {extracted['patient_id']})")

    summary = generate_summary(full_text, extracted["overall_interpretation"])
    print(f"Generated summary length: {len(summary)} chars")

    chunks = chunk_summary(summary)
    print(f"Created {len(chunks)} chunks")

    store_to_db(extracted, full_text, chunks)
