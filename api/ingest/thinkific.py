import os
import re
import time
import boto3
from playwright.sync_api import sync_playwright

# Configure S3 client using environment variables.
S3_ENDPOINT = os.environ.get("S3_ENDPOINT")
S3_ACCESS_KEY_ID = os.environ.get("S3_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.environ.get("S3_SECRET_ACCESS_KEY")
BUCKET_RAW = os.environ.get("S3_BUCKET_RAW")
PLAYWRIGHT_STATE_DIR = os.environ.get("PLAYWRIGHT_STATE_DIR", "/app/playwright_state")

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=S3_ACCESS_KEY_ID,
    aws_secret_access_key=S3_SECRET_ACCESS_KEY,
)

def _safe_key(s: str) -> str:
    """
    Generate a safe S3 key by replacing non-alphanumeric characters with underscores and limiting length.
    """
    return re.sub(r"[^a-zA-Z0-9._/-]+", "_", s)[:200]

def ingest_thinkific(base_url: str, email: str, password: str):
    """
    Log in to Thinkific, crawl courses and lessons, and upload lesson HTML to S3.
    Returns a list of lessons with metadata.
    """
    lessons = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(storage_state=f"{PLAYWRIGHT_STATE_DIR}/thinkific.json")
        page = ctx.new_page()

        # Navigate to courses page; handle login if necessary.
        page.goto(f"{base_url}/courses")
        if "sign_in" in page.url or "login" in page.url:
            page.goto(f"{base_url}/users/sign_in")
            page.fill('input[type="email"]', email)
            page.fill('input[type="password"]', password)
            page.click('button[type="submit"]')
            page.wait_for_load_state("networkidle")
            # Save the authenticated storage state for future sessions.
            ctx.storage_state(path=f"{PLAYWRIGHT_STATE_DIR}/thinkific.json")

        # Load the courses page and collect unique course links.
        page.goto(f"{base_url}/courses")
        time.sleep(1)
        course_links = list({
            a.get_attribute("href")
            for a in page.query_selector_all('a[href*="/courses/"]')
            if a.get_attribute("href")
        })
        for href in course_links:
            # Normalize relative URLs.
            if not href.startswith("http"):
                href = base_url.rstrip("/") + href
            page.goto(href, wait_until="networkidle")
            course_title = (page.text_content("h1") or "Course").strip()
            # Collect unique lesson links within the course.
            lesson_links = list({
                a.get_attribute("href")
                for a in page.query_selector_all('a[href*="/lessons/"]')
                if a.get_attribute("href")
            })
            for l in lesson_links:
                if not l.startswith("http"):
                    l = base_url.rstrip("/") + l
                page.goto(l, wait_until="networkidle")
                lesson_title = (page.text_content("h1") or "Lesson").strip()
                html = page.content()
                key = f"thinkific/{_safe_key(course_title)}/{_safe_key(lesson_title)}.html"
                s3.put_object(
                    Bucket=BUCKET_RAW,
                    Key=key,
                    Body=html.encode("utf-8"),
                    ContentType="text/html; charset=utf-8",
                )
                # Attempt to retrieve a video URL from video or source tags.
                video_url = page.get_attribute('video', 'src') or page.get_attribute('source[type="video/mp4"]', 'src')
                lessons.append({
                    "course_title": course_title,
                    "lesson_title": lesson_title,
                    "url": page.url,
                    "html_key": key,
                    "video_url": video_url,
                })
        ctx.close()
        browser.close()
    return lessons
