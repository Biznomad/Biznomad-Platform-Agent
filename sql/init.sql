CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS courses(
  id SERIAL PRIMARY KEY,
  title TEXT,
  url TEXT,
  created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS lessons(
  id SERIAL PRIMARY KEY,
  course_id T REFERENCES courses(id),
  title TEXT,
  url TEXT,
  html_s3_key TEXT,
  video_url TEXT,
  transcript_s3_key TEXT,
  created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks(
  id SERIAL PRIMARY KEY,
  lesson_id INT REFERENCES lessons(id),
  content TEXT,
  embedding vector(3072),
  meta JSONB
);

CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING GIN (to_tsvector('english', content));
CREATE INDEX IF NOT EXISTS idx_chunks_vec ON chunks USING ivfflat (embedding) WITH (lists=100);
