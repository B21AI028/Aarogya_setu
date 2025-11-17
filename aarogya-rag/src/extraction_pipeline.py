#!/usr/bin/env python3
"""
lab_pipeline.py

Run with:
  python lab_pipeline.py --manual
  python lab_pipeline.py --pdf uploads/Nimi.pdf

This script:
- can use provided manual extracted data (your pasted lab_values) when PDF extraction isn't available
- creates a simple textual summary (no LLM required when using --manual)
- chunks the summary
- embeds chunks using SentenceTransformer
- stores report, params, and chunks into Postgres (or SQLite for testing)

Make sure to set DATABASE_URL in .env (e.g. export DATABASE_URL="postgresql://user:pass@host:5432/dbname")
"""

import os
import json
import argparse
import datetime
from uuid import uuid4
from dotenv import load_dotenv

# Optional LLM imports left in case you want to re-enable later
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_ollama import OllamaLLM
    LLM_AVAILABLE = True
except Exception:
    LLM_AVAILABLE = False

# Text splitter and embedding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# DB
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

load_dotenv()

DB_URL = os.getenv("DATABASE_URL", "")  # e.g. postgresql://user:pass@host:5432/db
if not DB_URL:
    # Use a local sqlite file for quick testing if DATABASE_URL not provided
    DB_URL = "sqlite:///lab_pipeline_test.db"
    print("WARNING: DATABASE_URL not set. Falling back to local sqlite at lab_pipeline_test.db")

# Initialize DB engine and sessionmaker
engine = create_engine(DB_URL, future=True)
Session = sessionmaker(bind=engine)

# Embedding model: change to your own if needed
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # small & fast; swap to MedEmbed when available

# Optional LLM clients (not used for --manual mode)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if LLM_AVAILABLE and GOOGLE_API_KEY:
    llm_gemini = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0)
    llm_ollama = OllamaLLM(model="gemma:2b")
else:
    llm_gemini = None
    llm_ollama = None

# --------------- Your provided lab_values (manual fallback) ---------------
MANUAL_EXTRACTED = {
    "patient_name": "Unknown Patient",
    "patient_id": str(uuid4()),
    "date_of_collection": None,
    "date_of_report": None,
    "test_type": "Basic Metabolic Panel",
    "doctor_name": None,
    "lab_name": None,
    "lab_values": [
        {"parameter": "Urea", "value": "22 mg/dL", "reference_range": "17-43 mg/dL"},
        {"parameter": "Creatinine - Serum", "value": "0.62 mg/dL", "reference_range": "0.51-0.95 mg/dL"},
        {"parameter": "Uric Acid", "value": "6.7 mg/dL", "reference_range": "2.6-6.0 mg/dL"},
        {"parameter": "Sodium - serum", "value": "133 mmol/L", "reference_range": "136-146 mmol/L"},
        {"parameter": "Potassium - Serum", "value": "4.6 mmol/L", "reference_range": "3.5-5.1 mmol/L"},
        {"parameter": "Chloride - Serum", "value": "100 mmol/L", "reference_range": "101-109 mmol/L"},
        {"parameter": "Calcium - Serum", "value": "9.1 mg/dL", "reference_range": "8.8-10.6 mg/dL"}
    ],
    "abnormal_findings": [],  # will compute below
    "overall_interpretation": ""
}

# Populate abnormal_findings and interpretation automatically for manual mode
def auto_interpret(extracted):
    abnormalities = []
    for p in extracted["lab_values"]:
        try:
            # parse numeric part of value and range simply (best-effort)
            val_str = p["value"].split()[0].replace(",", "")
            val = float(val_str)
            # try to parse reference range (low-high)
            rr = p.get("reference_range") or ""
            if "-" in rr:
                parts = rr.split("-")
                try:
                    low = float(parts[0])
                    high = float(parts[1].split()[0])
                    if val < low or val > high:
                        abnormalities.append(f"{p['parameter']}: {p['value']} (ref {p['reference_range']})")
                except Exception:
                    # skip if parsing fails
                    pass
        except Exception:
            pass

    extracted["abnormal_findings"] = abnormalities
    if abnormalities:
        extracted["overall_interpretation"] = "Abnormal values detected: " + "; ".join(abnormalities)
    else:
        extracted["overall_interpretation"] = "No values outside reference ranges detected."

    return extracted

# ------------------- PDF extractor (optional) -------------------
def pdf_extractor(pdf_path: str):
    """
    Try to extract JSON from PDF using LLM (original approach).
    If LLMs are not configured this will raise.
    Keep this function so you can re-enable LLM-based extraction later.
    """
    if not llm_gemini:
        raise RuntimeError("LLM pdf_extractor is not available. Set GOOGLE_API_KEY and install langchain_google_genai.")
    from langchain_community.document_loaders import PyPDFLoader
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
    response = llm_gemini.invoke(prompt)
    response_text = getattr(response, "content", str(response)).strip()

    # Robust parsing
    if "```json" in response_text:
        start = response_text.find("```json") + 7
        end = response_text.find("```", start)
        json_text = response_text[start:end].strip()
    elif response_text.startswith("{"):
        json_text = response_text
    else:
        raise ValueError("No valid JSON in LLM response")

    result = json.loads(json_text)
    if not result.get("patient_id"):
        result["patient_id"] = str(uuid4())
    return result, full_text

# ------------------- Summary generation -------------------
def generate_summary_simple(full_text_or_values: str, interpretation: str, extracted: dict):
    """
    Create a short 200-400 word summary from the lab values and interpretation without LLM.
    Uses the structured extracted dict to craft a concise clinical summary.
    """
    lines = []
    lines.append(f"Patient: {extracted.get('patient_name','N/A')} (ID: {extracted.get('patient_id')})")
    if extracted.get("date_of_report"):
        lines.append(f"Report date: {extracted.get('date_of_report')}")
    lines.append("")
    lines.append("Key results:")
    for p in extracted["lab_values"]:
        lines.append(f"- {p['parameter']}: {p['value']} (Ref: {p.get('reference_range','N/A')})")
    lines.append("")
    if extracted.get("abnormal_findings"):
        lines.append("Abnormal findings:")
        for a in extracted["abnormal_findings"]:
            lines.append(f"- {a}")
    else:
        lines.append("No abnormal findings detected based on provided reference ranges.")
    lines.append("")
    lines.append("Interpretation:")
    lines.append(interpretation or extracted.get("overall_interpretation",""))
    # join and limit length
    summary = "\n".join(lines)
    # crude length normalization
    if len(summary) < 800:
        return summary
    else:
        return summary[:800]

# ------------------- Chunking -------------------
def chunk_summary(summary: str, chunk_size=300, overlap=50):
    """
    Chunk text summary into list of chunks (strings).
    chunk_size and overlap are approx character targets here.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size * 4,  # approximate multiplier to get reasonable chunking
        chunk_overlap=overlap * 4,
        separators=["\n\n", "\n", ". "]
    )
    docs = splitter.create_documents([summary])
    return [d.page_content for d in docs]

# ------------------- DB storage -------------------
def store_to_db(extracted: dict, full_text: str, chunks: list):
    """
    Stores:
    - lab_reports (id, patient_id, patient_name, date_of_collection, date_of_report,
                  test_type, doctor_name, lab_name, overall_interpretation, abnormal_findings, raw_text, created_at)
    - lab_params  (report_id, parameter, value, reference_range, is_abnormal)
    - lab_chunks  (report_id, chunk_text, embedding_json, metadata_json, created_at)

    NOTE: This implementation stores embeddings as JSON (embedding_json) to be DB-agnostic.
    If you want to store as vector type in Postgres, adapt the SQL to use the vector extension.
    """
    session = Session()
    report_id = str(uuid4())
    created_at = datetime.datetime.utcnow().isoformat()

    try:
        # Ensure abnormal_findings and overall_interpretation are JSON-serializable
        abnormal_json = json.dumps(extracted.get("abnormal_findings", []))

        # Insert into lab_reports
        insert_report_sql = text("""
            INSERT INTO lab_reports
            (id, patient_id, patient_name, date_of_collection, date_of_report, test_type,
             doctor_name, lab_name, overall_interpretation, abnormal_findings, raw_text, created_at)
            VALUES (:id, :patient_id, :patient_name, :date_of_collection, :date_of_report, :test_type,
                    :doctor_name, :lab_name, :overall_interpretation, :abnormal_findings, :raw_text, :created_at)
        """)
        session.execute(insert_report_sql, {
            "id": report_id,
            "patient_id": extracted["patient_id"],
            "patient_name": extracted.get("patient_name"),
            "date_of_collection": extracted.get("date_of_collection"),
            "date_of_report": extracted.get("date_of_report"),
            "test_type": extracted.get("test_type"),
            "doctor_name": extracted.get("doctor_name"),
            "lab_name": extracted.get("lab_name"),
            "overall_interpretation": extracted.get("overall_interpretation"),
            "abnormal_findings": abnormal_json,
            "raw_text": full_text,
            "created_at": created_at
        })

        # Insert lab_params
        for param in extracted["lab_values"]:
            param_json = json.dumps(param)
            # simple abnormal check (exact match in abnormal_findings strings)
            is_abnormal = any(param["parameter"].lower() in af.lower() for af in extracted.get("abnormal_findings", []))
            insert_param_sql = text("""
                INSERT INTO lab_params (report_id, parameter, value, reference_range, is_abnormal, raw_json)
                VALUES (:report_id, :parameter, :value, :reference_range, :is_abnormal, :raw_json)
            """)
            session.execute(insert_param_sql, {
                "report_id": report_id,
                "parameter": param["parameter"],
                "value": param["value"],
                "reference_range": param.get("reference_range"),
                "is_abnormal": bool(is_abnormal),
                "raw_json": param_json
            })

        # Insert chunks with embeddings stored as JSON (embedding_json)
        for i, chunk_text in enumerate(chunks):
            embedding = embedding_model.encode(chunk_text).tolist()
            insert_chunk_sql = text("""
                INSERT INTO lab_chunks (report_id, chunk_text, embedding_json, metadata_json, created_at)
                VALUES (:report_id, :chunk_text, :embedding_json, :metadata_json, :created_at)
            """)
            session.execute(insert_chunk_sql, {
                "report_id": report_id,
                "chunk_text": chunk_text,
                "embedding_json": json.dumps(embedding),
                "metadata_json": json.dumps({"chunk_id": i, "summary_type": "lab_summary"}),
                "created_at": created_at
            })

        session.commit()
        print(f"Stored report {report_id} with {len(extracted['lab_values'])} params and {len(chunks)} chunks.")
        return report_id

    except Exception as e:
        session.rollback()
        raise
    finally:
        session.close()

# ------------------- main -------------------
def main():
    parser = argparse.ArgumentParser(description="Extract and store lab report from PDF or manual JSON fallback")
    parser.add_argument("--pdf", default=None, help="Path to PDF file")
    parser.add_argument("--manual", action="store_true", help="Use embedded manual extracted data (provided lab_values)")
    parser.add_argument("--no-db", action="store_true", help="Do not write to DB; just run pipeline and print results")
    args = parser.parse_args()

    if args.manual:
        extracted = MANUAL_EXTRACTED.copy()
        extracted = auto_interpret(extracted)
        full_text = "Manual input - structured lab_values provided by user."
    elif args.pdf:
        if not os.path.exists(args.pdf):
            raise FileNotFoundError(f"PDF not found: {args.pdf}")
        extracted, full_text = pdf_extractor(args.pdf)
        # ensure abnormal findings
        if not extracted.get("abnormal_findings"):
            extracted = auto_interpret(extracted)
    else:
        parser.error("Either --manual or --pdf <path> must be provided.")

    print("EXTRACTED (preview):")
    print(json.dumps({
        "patient_name": extracted.get("patient_name"),
        "patient_id": extracted.get("patient_id"),
        "lab_values_count": len(extracted["lab_values"]),
        "abnormal_findings": extracted.get("abnormal_findings")
    }, indent=2))

    # Generate summary (simple) — for manual mode we avoid LLM
    summary = generate_summary_simple(full_text, extracted.get("overall_interpretation", ""), extracted)
    print("\nSUMMARY (first 800 chars):")
    print(summary[:800])

    # Chunk
    chunks = chunk_summary(summary)
    print(f"\nCreated {len(chunks)} chunks (showing first chunk):\n")
    if chunks:
        print(chunks[0][:800])

    # Store
    if args.no_db:
        print("\n-- no-db set; skipping DB write --")
    else:
        # Ensure DB tables exist (very lightweight guidance)
        # Note: For production, use proper migrations. Here we try to create simple tables in sqlite/postgres if they don't exist.
        with engine.begin() as conn:
            if engine.dialect.name == "sqlite":
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS lab_reports (
                        id TEXT PRIMARY KEY,
                        patient_id TEXT,
                        patient_name TEXT,
                        date_of_collection TEXT,
                        date_of_report TEXT,
                        test_type TEXT,
                        doctor_name TEXT,
                        lab_name TEXT,
                        overall_interpretation TEXT,
                        abnormal_findings TEXT,
                        raw_text TEXT,
                        created_at TEXT
                    );
                """))
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS lab_params (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        report_id TEXT,
                        parameter TEXT,
                        value TEXT,
                        reference_range TEXT,
                        is_abnormal BOOLEAN,
                        raw_json TEXT
                    );
                """))
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS lab_chunks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        report_id TEXT,
                        chunk_text TEXT,
                        embedding_json TEXT,
                        metadata_json TEXT,
                        created_at TEXT
                    );
                """))
            else:
                # For Postgres, create JSON columns; if you have vector extension and want vector type, change accordingly.
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS lab_reports (
                        id TEXT PRIMARY KEY,
                        patient_id TEXT,
                        patient_name TEXT,
                        date_of_collection TEXT,
                        date_of_report TEXT,
                        test_type TEXT,
                        doctor_name TEXT,
                        lab_name TEXT,
                        overall_interpretation TEXT,
                        abnormal_findings JSONB,
                        raw_text TEXT,
                        created_at TIMESTAMP
                    );
                """))
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS lab_params (
                        id SERIAL PRIMARY KEY,
                        report_id TEXT,
                        parameter TEXT,
                        value TEXT,
                        reference_range TEXT,
                        is_abnormal BOOLEAN,
                        raw_json JSONB
                    );
                """))
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS lab_chunks (
                        id SERIAL PRIMARY KEY,
                        report_id TEXT,
                        chunk_text TEXT,
                        embedding_json JSONB,
                        metadata_json JSONB,
                        created_at TIMESTAMP
                    );
                """))

        # Store to DB
        report_id = store_to_db(extracted, full_text, chunks)
        print(f"\nSuccess: report stored with id {report_id}")

if __name__ == "__main__":
    main()
