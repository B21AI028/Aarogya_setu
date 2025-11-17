
import os
import json
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
def detect_stored_dim_for_patient(patient_id: str):
    session = Session()
    try:
        row = session.execute(text("""
            SELECT lc.embedding_json
            FROM lab_chunks lc
            JOIN lab_reports lr ON lc.report_id = lr.id
            WHERE lr.patient_id = :patient_id
            LIMIT 1
        """), {"patient_id": patient_id}).fetchone()
        if not row:
            return None
        emb_json = row[0]
        if emb_json is None:
            return None
        if isinstance(emb_json, str):
            try:
                emb = json.loads(emb_json)
            except Exception:
                try:
                    emb = eval(emb_json)
                except Exception:
                    return None
        else:
            emb = emb_json
        return len(emb) if hasattr(emb, "__len__") else None
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
        rows = session.execute(text("""
            SELECT lc.id, lc.chunk_text, lc.embedding_json, lc.metadata_json
            FROM lab_chunks lc
            JOIN lab_reports lr ON lc.report_id = lr.id
            WHERE lr.patient_id = :patient_id
        """), {"patient_id": patient_id}).fetchall()

        if not rows:
            return []

        q_emb = embedding_model.encode(query)
        candidates = []
        for r in rows:
            chunk_id = r[0]
            chunk_text = r[1]
            emb_json = r[2]
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
if __name__ == "__main__":
    print("Running RAG test pipeline...")

    patient_id = "7e0da30e-8a99-450c-9908-c0e2c95ab939"
    query = "My uric acid value is?"

    print("Detecting stored embedding dimension for patient...")
    dim = detect_stored_dim_for_patient(patient_id)
    print("Stored embedding dim:", dim)

    print("Retrieving chunks (with automatic model selection)...")
    topk = retrieve_chunks_python(patient_id, query, k=3)
    print("Retrieved top chunks (id, sim):", [(c['id'], round(c['sim'], 4)) for c in topk])

    print("\nGenerating answer (OpenAI or fallback)...")
    answer = generate_answer(topk, query)
    print("\nANSWER:\n", answer)
