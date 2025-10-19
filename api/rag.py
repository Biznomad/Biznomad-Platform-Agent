import os
from openai import OpenAI
from sqlalchemy import text
from .db import engine, keyword_then_vector

# Initialize the OpenAI client using the API key from environment variables.
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Create embeddings for a list of texts using OpenAI's embedding model.
    Returns a list of embedding vectors corresponding to the input texts.
    """
    resp = client.embeddings.create(model="text-embedding-3-large", input=texts)
    return [d.embedding for d in resp.data]

def answer(query: str) -> tuple[str, list[str]]:
    """
    Perform RAG: embed the query, search relevant chunks, then generate an answer.
    Returns a tuple of (answer text, list of citation strings).
    """
    # Embed the user query into a vector.
    qvec = embed_texts([query])[0]
    with engine.connect() as conn:
        # Hybrid search: keyword filter then vector similarity ranking.
        hits = keyword_then_vector(conn, query, qvec, k=10)
        # Build the context and citation list from the search hits.
        context = []
        citations = []
        for h in hits:
            # Retrieve course and lesson titles for citations.
            meta = conn.execute(text(
                """
                SELECT c.title AS course, l.title AS lesson
                FROM lessons l JOIN courses c ON l.course_id = c.id
                WHERE l.id = :lid
                """
            ), {"lid": h["lesson_id"]}).mappings().first()
            context.append(h["content"])
            citations.append(f"{meta['course']} \u25b8 {meta['lesson']} [chunk {h['id']}]")
    # Prepare messages for the chat completion API.
    system = "You are a concise assistant. Cite chunks like [chunk N] in-line."
    user_message = f"Q: {query}\n\nContext:\n" + "\n---\n".join(context)
    chat = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user_message}],
        temperature=0.2
    )
    answer_text = chat.choices[0].message.content
    return answer_text, citations
