import os
import boto3
import httpx
import tempfile
import subprocess
from openai import OpenAI

# Configure S3 client using environment variables. Works with MinIO or other S3-compatible stores.
S3_ENDPOINT = os.environ.get("S3_ENDPOINT")
S3_ACCESS_KEY_ID = os.environ.get("S3_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.environ.get("S3_SECRET_ACCESS_KEY")
BUCKET_TRANS = os.environ.get("S3_BUCKET_TRANSCRIPTS")

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY_ID,
    aws_secret_access_key=S3_SECRET_ACCESS_KEY,
)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def download_bytes(url: str) -> bytes:
    """
    Download content from a given URL and return as bytes.
    """
    with httpx.stream("GET", url, timeout=120) as response:
        response.raise_for_status()
        return response.read()

def video_to_mp3(video_bytes: bytes) -> bytes:
    """
    Convert video bytes to MP3 audio using ffmpeg. Returns MP3 bytes.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f_in, \
         tempfile.NamedTemporaryFile(suffix=".mp3") as f_out:
        f_in.write(video_bytes)
        f_in.flush()
        # Use ffmpeg to extract audio and encode as MP3.
        subprocess.run(
            ["ffmpeg", "-y", "-i", f_in.name, "-vn", "-acodec", "libmp3lame", f_out.name],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return f_out.read()

def whisper_transcribe(audio_bytes: bytes) -> str:
    """
    Transcribe audio bytes to text using OpenAI Whisper API (GPT-4o-transcribe).
    """
    with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
        f.write(audio_bytes)
        f.flush()
        transcription = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=open(f.name, "rb"),
        )
    return transcription.text

def put_transcript(key: str, text: str) -> None:
    """
    Upload transcript text to S3 (MinIO) at the given key.
    """
    s3.put_object(
        Bucket=BUCKET_TRANS,
        Key=key,
        Body=text.encode("utf-8"),
        ContentType="text/plain; charset=utf-8",
    )
