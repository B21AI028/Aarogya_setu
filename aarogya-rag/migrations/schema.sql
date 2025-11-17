-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Lab Reports Table (stores metadata like patient_id, report_date, etc.)
CREATE TABLE IF NOT EXISTS lab_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,  -- Link to profiles
    patient_name VARCHAR(255) NOT NULL,
    date_of_collection DATE,
    date_of_report DATE,
    test_type VARCHAR(255),
    doctor_name VARCHAR(255),
    lab_name VARCHAR(255),
    overall_interpretation TEXT,
    abnormal_findings TEXT[],  -- Array of strings
    raw_text TEXT,  -- Store extracted raw text here (instead of S3)
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_lab_reports_patient_id ON lab_reports(patient_id);

-- Lab Params Table (extracted values)
CREATE TABLE IF NOT EXISTS lab_params (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id UUID NOT NULL REFERENCES lab_reports(id) ON DELETE CASCADE,
    parameter VARCHAR(255) NOT NULL,
    value VARCHAR(255) NOT NULL,
    reference_range VARCHAR(255),
    is_abnormal BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_lab_params_report_id ON lab_params(report_id);
CREATE INDEX idx_lab_params_parameter ON lab_params(parameter);

-- Lab Chunks Table (for RAG: chunked summaries with embeddings)
CREATE TABLE IF NOT EXISTS lab_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id UUID NOT NULL REFERENCES lab_reports(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    embedding VECTOR(768),  -- MedEmbed dimension
    metadata JSONB,  -- e.g., {"chunk_id": 1, "summary_type": "lab_summary"}
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for vector search (cosine similarity)
CREATE INDEX idx_lab_chunks_embedding ON lab_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);