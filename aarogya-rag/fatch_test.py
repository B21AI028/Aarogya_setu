# fetch_lab_report.py
import os
import pandas as pd
from sqlalchemy import create_engine, text

# update DATABASE_URL to match your DB; example:
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/aarogya")

engine = create_engine(DATABASE_URL)

# 1) latest reports
reports_q = "SELECT id, patient_id, patient_name, created_at, overall_interpretation FROM lab_reports ORDER BY created_at DESC LIMIT 10;"
reports_df = pd.read_sql(reports_q, engine)
print("=== latest lab_reports ===")
print(reports_df.to_string(index=False))

# If you want to inspect one report:
if not reports_df.empty:
    report_id = reports_df.loc[0, "id"]
    print("\nUsing report_id:", report_id)

    params_q = text("SELECT parameter, value, reference_range, is_abnormal, raw_json FROM lab_params WHERE report_id = :rid;")
    params_df = pd.read_sql(params_q, engine, params={"rid": report_id})
    print("\n=== lab_params ===")
    print(params_df.to_string(index=False))

    chunks_q = text("SELECT id, chunk_text, metadata_json FROM lab_chunks WHERE report_id = :rid;")
    chunks_df = pd.read_sql(chunks_q, engine, params={"rid": report_id})
    print("\n=== lab_chunks ===")
    print(chunks_df.to_string(index=False))
