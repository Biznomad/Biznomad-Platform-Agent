import json
import re
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import text

from .db import engine
from .rag import embed_texts, answer
from .ingest.thinkific import ingest_thinkific
from .ingest.util_media import (
    download_bytes,
    video_to_mp3,
    whisper_transcribe,
    put_transcript,
)

app = FastAPI(title="Course Agent")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """
    Render the home page with a chat interface and ingestion form.
    """
    return templates.TemplateResponse("chat.html", {"request": request})


@app.post("/chat")
def chat(prompt: str = Form(...)):
    """
    Receive a question from the user, perform RAG search, and return an answer with citations.
    """
    ans, cites = answer(prompt)
    return {"answer": ans, "citations": cites}


@app.post("/ingest/thinkific")
def run_thinkific(base_url: str = Form(...), email: str = Form(...), password: str = Form(...)):
    """
    Ingest Thinkific content by logging in and crawling courses and lessons.
    Inserts course and lesson metadata into the database.
    """
    data = ingest_thinkific(base_url, email, password)
    ingested = 0
    with engine.begin() as conn:
        for L in data:
            # Upsert course by title.
            c = conn.execute(
                text("SELECT id FROM courses WHERE title=:t"),
                {"t": L["course_title"]},
            ).mappings().first()
            if not c:
                conn.execute(
                    text("INSERT INTO courses(title, url) VALUES (:t, :u)"),
                    {"t": L["course_title"], "u": base_url},
                )
                c = conn.execute(
                    text("SELECT id FROM courses WHERE title=:t"),
                    {"t": L["course_title"]},
                ).mappings().first()
            cid = c["id"]
            # Insert lesson metadata.
            conn.execute(
                text(
                    """
                    INSERT INTO lessons(course_id, title, url, html_s3_key, video_url)
                    VALUES (:cid, :lt, :url, :hk, :vid)
                    """
                ),
                {
                    "cid": cid,
                    "lt": L["lesson_title"],
                    "url": L["url"],
                    "hk": L["html_key"],
                    "vid": L["video_url"],
                },
            )
            ingested += 1
    return {"ingested": ingested}


@app.post("/index_html")
def index_html(lesson_id: int = Form(...), html_text: str = Form(...)):
    """
    Extract plain text from HTML, embed, and insert chunks into the database.
    """
    # Remove HTML tags to get plain text.
    text_only = re.sub("<[^>]+>", " ", html_text)
    text_only = re.sub(r"\s+", " ", text_only).strip()
    parts = [text_only[i:i+1600] for i in range(0, len(text_only), 1600)]
    vecs = embed_texts(parts)
    with engine.begin() as conn:
        for p, e in zip(parts, vecs):
            conn.execute(
                text(
                    """
                    INSERT INTO chunks(lesson_id, content, embedding, meta)
                    VALUES (:lid, :c, :e, :m)
                    """
                ),
                {
                    "lid": lesson_id,
                    "c": p,
                    "e": e,
                    "m": json.dumps({"source": "html"}),
                },
            )
    return {"indexed_chunks": len(parts)}


@app.post("/transcribe_lesson")
def transcribe_lesson(lesson_id: int = Form(...)):
    """
    Download a lesson's video, convert to audio, transcribe, embed, and insert chunks.
    """
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT video_url FROM lessons WHERE id=:i"), {"i": lesson_id}
        ).mappings().first()
        if not row or not row["video_url"]:
            return JSONResponse({"error": "No video_url"}, status_code=400)
        video_bytes = download_bytes(row["video_url"])
        audio_bytes = video_to_mp3(video_bytes)
        transcript = whisper_transcribe(audio_bytes)
        # Store transcript in S3.
        key = f"transcripts/{lesson_id}.txt"
        put_transcript(key, transcript)
        # Update the lesson record with transcript key.
        conn.execute(
            text("UPDATE lessons SET transcript_s3_key=:k WHERE id=:i"),
            {"k": key, "i": lesson_id},
        )
        # Chunk and embed the transcript.
        parts = [transcript[i:i+1600] for i in range(0, len(transcript), 1600)]
        vecs = embed_texts(parts)
        for p, e in zip(parts, vecs):
            conn.execute(
                text(
                    """
                    INSERT INTO chunks(lesson_id, content, embedding, meta)
                    VALUES (:lid, :c, :e, :m)
                    """
                ),
                {
                    "lid": lesson_id,
                    "c": p,
                    "e": e,
                    "m": json.dumps({"source": "whisper"}),
                },
            )
    return {"transcribed": True, "chunks": len(parts)}
