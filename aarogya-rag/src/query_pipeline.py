
import os
import sys
import json
import argparse
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sentence_transformers import SentenceTransformer
import numpy as np
import warnings

load_dotenv()

# ----------------- Config -----------------
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/aarogya")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", None)
USE_OPENAI = bool(OPENAI_KEY)

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

# Embedding models (may auto-switch to match stored dim)
preferred_model = "abhinand/MedEmbed-base-v0.1"
candidate_models = [
    "abhinand/MedEmbed-base-v0.1",  # 768-dim
    "all-MiniLM-L6-v2",             # 384-dim
    "sentence-transformers/all-mpnet-base-v2"
]
embedding_model = SentenceTransformer(preferred_model)

# OpenAI client (if available)
client = OpenAI(api_key=OPENAI_KEY) if USE_OPENAI else None

# ----------------- Helpers -----------------
def _fetch_one(query: str, params: dict):
    """Utility to run a single-row query safely."""
    session = Session()
    try:
        return session.execute(text(query), params).fetchone()
    finally:
        session.close()


def detect_stored_dim_for_patient(patient_id: str):
    session = Session()
    try:
        # Try JSON-based schema first
        row = None
        try:
            row = session.execute(text("""
                SELECT lc.embedding_json
                FROM lab_chunks lc
                JOIN lab_reports lr ON lc.report_id = lr.id
                WHERE lr.patient_id = :patient_id
                LIMIT 1
            """), {"patient_id": patient_id}).fetchone()
        except Exception:
            row = None

        # Fallback to vector-based schema
        if not row:
            try:
                row = session.execute(text("""
                    SELECT lc.embedding
                    FROM lab_chunks lc
                    JOIN lab_reports lr ON lc.report_id = lr.id
                    WHERE lr.patient_id = :patient_id
                    LIMIT 1
                """), {"patient_id": patient_id}).fetchone()
            except Exception:
                row = None
        if not row:
            return None
        emb_json = row[0]
        if emb_json is None:
            return None
        # emb_json may be JSON text, Python list, memoryview, or vector type
        try:
            if isinstance(emb_json, (bytes, bytearray)):
                # unlikely, but guard
                return None
            if isinstance(emb_json, str):
                try:
                    emb = json.loads(emb_json)
                except Exception:
                    emb = eval(emb_json)
            else:
                emb = list(emb_json) if not hasattr(emb_json, "__len__") else emb_json
            return len(emb) if hasattr(emb, "__len__") else None
        except Exception:
            return None
    finally:
        session.close()

def find_matching_model_for_dim(dim: int):
    global embedding_model
    if dim is None:
        return None
    for m in candidate_models:
        try:
            tmp = SentenceTransformer(m)
            vec = tmp.encode("test vector for dimension check")
            if len(vec) == dim:
                embedding_model = tmp
                return m
        except Exception:
            continue
    return None

def cosine_sim(a: np.ndarray, b: np.ndarray):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        warnings.warn(f"Embedding shapes differ: query {a.shape} vs chunk {b.shape}. Aligning by trunc/pad.")
        minlen = min(a.shape[0], b.shape[0])
        a = a[:minlen]
        b = b[:minlen]
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def retrieve_chunks_python(patient_id: str, query: str, k=3):
    stored_dim = detect_stored_dim_for_patient(patient_id)
    if stored_dim:
        test_vec = embedding_model.encode("dimension test")
        if len(test_vec) != stored_dim:
            matched = find_matching_model_for_dim(stored_dim)
            if matched:
                print(f"[info] switched embedding model to '{matched}' to match stored dim {stored_dim}.")
            else:
                warnings.warn(f"No candidate model matched stored dim {stored_dim}. Proceeding with current model (may reduce accuracy).")

    session = Session()
    try:
        rows = []
        # Try JSON-based schema
        try:
            rows = session.execute(text("""
                SELECT lc.id, lc.chunk_text, lc.embedding_json AS emb_col, lc.metadata_json AS meta_col
                FROM lab_chunks lc
                JOIN lab_reports lr ON lc.report_id = lr.id
                WHERE lr.patient_id = :patient_id
            """), {"patient_id": patient_id}).fetchall()
            mode = "json"
        except Exception:
            rows = []
            mode = None

        # Fallback to vector-based schema
        if not rows:
            try:
                rows = session.execute(text("""
                    SELECT lc.id, lc.chunk_text, lc.embedding AS emb_col, lc.metadata AS meta_col
                    FROM lab_chunks lc
                    JOIN lab_reports lr ON lc.report_id = lr.id
                    WHERE lr.patient_id = :patient_id
                """), {"patient_id": patient_id}).fetchall()
                mode = "vector"
            except Exception:
                rows = []
                mode = None

        if not rows:
            return []

        q_emb = embedding_model.encode(query)
        candidates = []
        for r in rows:
            chunk_id = r[0]
            chunk_text = r[1]
            emb_json = r[2]
            # Parse embedding based on mode/shape
            if mode == "json":
                if isinstance(emb_json, str):
                    try:
                        emb = json.loads(emb_json)
                    except Exception:
                        try:
                            emb = eval(emb_json)
                        except Exception:
                            continue
                else:
                    emb = emb_json
            else:
                # vector column or other iterable
                try:
                    emb = list(emb_json) if not isinstance(emb_json, (list, tuple, np.ndarray)) else emb_json
                except Exception:
                    continue
            try:
                emb_arr = np.array(emb, dtype=float)
            except Exception:
                continue
            sim = cosine_sim(q_emb, emb_arr)
            candidates.append({"id": chunk_id, "text": chunk_text, "sim": sim})
        candidates.sort(key=lambda x: x["sim"], reverse=True)
        return candidates[:k]
    finally:
        session.close()

# ----------------- Answer generation using OpenAI -----------------
def generate_answer(chunks: list, query: str):
    """
    Use OpenAI chat completion if available. Otherwise fall back to a simple chunk-summary response.
    """
    if not chunks:
        return "No context available for this patient."

    # Build compact context (trim very long chunks to avoid token bloat)
    max_chars_per_chunk = 2000
    context = "\n\n".join([ (c["text"][:max_chars_per_chunk] + ("..." if len(c["text"]) > max_chars_per_chunk else "")) for c in chunks ])

    system_msg = (
        "You are a helpful and cautious medical assistant. "
        "Answer concisely and reference only the provided lab report context. "
        "If you are uncertain, advise consulting a clinician."
    )
    user_msg = f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\n we have give ans point to point no extra info"

    # Use OpenAI if configured
    if USE_OPENAI and client:
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.0,
                max_tokens=300
            )
            text_out = resp.choices[0].message.content.strip()
            return text_out
        except Exception as e:
            # Log but fall back
            print("[warning] OpenAI call failed:", str(e))

    # Fallback deterministic response if OpenAI not available or fails
    top_excerpt = "\n\n---\n\n".join([c["text"][:400] for c in chunks])
    return f"Based on the patient's lab report excerpts:\n\n{top_excerpt}\n\n(Consult clinician for interpretation.)"

# ----------------- Run example -----------------
def lookup_latest_patient_id():
    """Return most recent patient_id from lab_reports."""
    session = Session()
    try:
        row = session.execute(text("""
            SELECT patient_id
            FROM lab_reports
            ORDER BY created_at DESC
            LIMIT 1
        """), {}).fetchone()
        return row[0] if row else None
    finally:
        session.close()


def lookup_patient_id_by_name(name: str):
    session = Session()
    try:
        row = session.execute(text("""
            SELECT patient_id
            FROM lab_reports
            WHERE LOWER(patient_name) = LOWER(:name)
            ORDER BY created_at DESC
            LIMIT 1
        """), {"name": name}).fetchone()
        return row[0] if row else None
    finally:
        session.close()


def list_recent_patients(limit: int = 5):
    session = Session()
    try:
        rows = session.execute(text("""
            SELECT DISTINCT patient_id, patient_name, created_at
            FROM lab_reports
            ORDER BY created_at DESC
            LIMIT :lim
        """), {"lim": limit}).fetchall()
        return [(r[0], r[1]) for r in rows]
    finally:
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query RAG over lab reports")
    parser.add_argument("--patient-id", dest="patient_id", default=None, help="Patient UUID to query")
    parser.add_argument("--patient-name", dest="patient_name", default=None, help="Patient name to resolve to latest ID")
    parser.add_argument("--latest", action="store_true", help="Use most recent patient by created_at (default if nothing provided)")
    parser.add_argument("--question", dest="question", default="My uric acid value is?", help="User question")
    parser.add_argument("-k", dest="k", type=int, default=3, help="Top-K chunks to retrieve")
    args = parser.parse_args()

    print("Running RAG query pipeline...")

    # Resolve patient_id
    patient_id = args.patient_id
    if not patient_id and args.patient_name:
        patient_id = lookup_patient_id_by_name(args.patient_name)
    if not patient_id:
        patient_id = lookup_latest_patient_id()

    if not patient_id:
        print("No patient data found in DB.\n\nNext steps:\n- Run the extraction pipeline to ingest a report, e.g.:\n  python aarogya-rag/src/extraction_pipeline.py --manual\n- Ensure the same DATABASE_URL is used by both extraction and query pipelines (see .env).")
        sys.exit(1)

    print("Detecting stored embedding dimension for patient...")
    dim = detect_stored_dim_for_patient(patient_id)
    print("Stored embedding dim:", dim)

    print("Retrieving chunks (with automatic model selection)...")
    topk = retrieve_chunks_python(patient_id, args.question, k=args.k)
    print("Retrieved top chunks (id, sim):", [(c['id'], round(c['sim'], 4)) for c in topk])

    if not topk:
        print("\nNo chunks found for this patient.\n\nTroubleshooting:\n- Confirm the extraction pipeline inserted chunks for this patient_id.\n- Verify both scripts point to the same DB via DATABASE_URL (.env).\n- List recent patients to cross-check IDs:")
        recents = list_recent_patients(5)
        for pid, pname in recents:
            print(f"  - {pname or 'Unknown'}: {pid}")

    print("\nGenerating answer (OpenAI or fallback)...")
    answer = generate_answer(topk, args.question)
    print("\nANSWER:\n", answer)
