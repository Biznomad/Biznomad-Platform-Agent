import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)

def keyword_then_vector(conn, qtext: str, qvec: list[float], k: int = 10):
    """
    Hybrid search: perform keyword filter, then vector similarity ranking.
    Returns top-k results as mappings with columns id, lesson_id, content, dist.
    """
    # Keyword pre-filter: find candidate chunks that match the query terms using full-text search.
    rows = conn.execute(text(
        """
        SELECT id, lesson_id, content
        FROM chunks
        WHERE to_tsvector('english', content) @@ plainto_tsquery(:q)
        LIMIT 200
        """
    ), {"q": qtext}).mappings().all()
    ids = [r["id"] for r in rows]
    # If no keyword matches, fall back to vector search across all chunks.
    if not ids:
        return conn.execute(text(
            """
            SELECT id, lesson_id, content, (embedding <-> :qvec) AS dist
            FROM chunks
            ORDER BY embedding <-> :qvec
            LIMIT :k
            """
        ), {"qvec": qvec, "k": k}).mappings().all()
    # Vector similarity ranking on the keyword-filtered set.
    return conn.execute(text(
        """
        SELECT id, lesson_id, content, (embedding <-> :qvec) AS dist
        FROM chunks
        WHERE id = ANY(:ids)
        ORDER BY embedding <-> :qvec
        LIMIT :k
        """
    ), {"qvec": qvec, "ids": ids, "k": k}).mappings().all()
