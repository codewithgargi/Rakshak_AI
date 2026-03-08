"""
Rakshak AI — FastAPI Backend (v3.0)
====================================
Endpoints:
  POST /api/triage        — Medical document triage (file or text)
  POST /api/symptoms      — Symptom analysis (with full conversation history)
  POST /api/transcribe    — Audio → text (Whisper)
  GET  /                  — Serve frontend

New in v3.0:
  - Full multi-turn conversation history support (ChatGPT-style)
  - Session ID based chat management
  - Simple in-memory rate limiting (10 req/min per IP)
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI, APIError, RateLimitError, APIConnectionError
from dotenv import load_dotenv
import base64, json, re, io, os, uuid, traceback, logging, time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from collections import defaultdict
import logging.handlers

load_dotenv()

# ─── Logging Setup ────────────────────────────────────────────────
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("rakshak")
logger.setLevel(logging.INFO)


# IST timezone set karo logs ke liye
class ISTFormatter(logging.Formatter):
    def converter(self, timestamp):
        import datetime
        dt = datetime.datetime.utcfromtimestamp(timestamp)
        ist = dt + datetime.timedelta(hours=5, minutes=30)
        return ist.timetuple()

formatter = ISTFormatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

file_handler = RotatingFileHandler(LOG_DIR / "app.log", maxBytes=5_000_000, backupCount=3)
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# ─── Config ───────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = 10
MAX_FILE_BYTES   = MAX_FILE_SIZE_MB * 1024 * 1024

ALLOWED_TYPES = {
    "application/pdf",
    "image/jpeg", "image/jpg", "image/png",
    "image/webp", "image/tiff",
}

ALLOWED_AUDIO = {
    "audio/wav", "audio/mpeg", "audio/mp4",
    "audio/webm", "audio/ogg", "audio/flac",
    "audio/x-wav", "audio/x-m4a",
}

# ─── Rate Limiter (Simple In-Memory) ─────────────────────────────
# Stores: { ip_address: [timestamp1, timestamp2, ...] }
RATE_LIMIT_REQUESTS = 15          # max requests
RATE_LIMIT_WINDOW   = 60          # per 60 seconds
_rate_store: dict[str, list] = defaultdict(list)

def check_rate_limit(ip: str):
    """
    Allows max RATE_LIMIT_REQUESTS per RATE_LIMIT_WINDOW seconds per IP.
    Raises HTTP 429 if exceeded.
    """
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW

    # Keep only timestamps within current window
    _rate_store[ip] = [t for t in _rate_store[ip] if t > window_start]

    if len(_rate_store[ip]) >= RATE_LIMIT_REQUESTS:
        logger.warning(f"Rate limit hit for IP: {ip}")
        raise HTTPException(429, detail={
            "error": "rate_limit_exceeded",
            "message": f"Too many requests. Max {RATE_LIMIT_REQUESTS} per minute. Please wait."
        })

    _rate_store[ip].append(now)

# ─── App Setup ────────────────────────────────────────────────────
app = FastAPI(title="Rakshak AI", version="3.0")

# CORS — currently open for development.
# For production, replace "*" with your actual domain e.g. ["https://rakshak.yourdomain.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC = Path(__file__).parent / "static"
if STATIC.exists():
    app.mount("/static", StaticFiles(directory=STATIC), name="static")


# ─── Middleware: Request ID ───────────────────────────────────────
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    rid = str(uuid.uuid4())[:8]
    request.state.rid = rid
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


# ─── Root ─────────────────────────────────────────────────────────
@app.get("/")
def root():
    idx = STATIC / "index.html"
    if idx.exists():
        return FileResponse(idx)
    return {"status": "Rakshak AI running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "version": "3.0"}


# ─── Helpers ─────────────────────────────────────────────────────
def get_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise HTTPException(400, detail={
            "error": "missing_api_key",
            "message": "OPENAI_API_KEY not set. Add to .env file."
        })
    return OpenAI(api_key=key)


def get_client_ip(request: Request) -> str:
    """Extract real client IP, handling proxies."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def validate_file(file: UploadFile, file_bytes: bytes, allowed_types: set):
    size = len(file_bytes)
    if size > MAX_FILE_BYTES:
        raise HTTPException(413, detail={
            "error": "file_too_large",
            "message": f"File exceeds {MAX_FILE_SIZE_MB}MB limit. Got {size / 1024 / 1024:.1f}MB."
        })
    ct = (file.content_type or "").split(";")[0].strip().lower()
    if ct not in allowed_types:
        raise HTTPException(415, detail={
            "error": "unsupported_media_type",
            "message": f"File type '{ct}' not allowed. Accepted: {', '.join(sorted(allowed_types))}"
        })
    return ct


def ocr_image_bytes(image_bytes: bytes) -> str:
    results = []
    try:
        import pytesseract
        import numpy as np
        import cv2
        from PIL import Image

        orig = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(orig)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        try:
            scaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            denoised = cv2.fastNlMeansDenoising(scaled, h=10)
            binary = cv2.adaptiveThreshold(
                denoised, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 31, 10
            )
            t1 = pytesseract.image_to_string(
                Image.fromarray(binary), config="--oem 3 --psm 6"
            ).strip()
            if t1: results.append(t1)
        except Exception:
            pass

        try:
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            t2 = pytesseract.image_to_string(
                Image.fromarray(otsu), config="--oem 3 --psm 6"
            ).strip()
            if t2: results.append(t2)
        except Exception:
            pass

        try:
            t3 = pytesseract.image_to_string(orig, config="--oem 3 --psm 6").strip()
            if t3: results.append(t3)
        except Exception:
            pass

    except ImportError:
        logger.warning("pytesseract / opencv not installed — OCR skipped")
    except Exception as e:
        logger.warning(f"OCR error: {e}")

    return max(results, key=len, default="")


def process_pdf(file_bytes: bytes) -> tuple[str, str | None, str | None]:
    text_parts = []

    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            total = len(pdf.pages)
            logger.info(f"PDF has {total} pages")
            for i, pg in enumerate(pdf.pages):
                tx = pg.extract_text()
                if tx and tx.strip():
                    text_parts.append(f"[Page {i+1}/{total}]\n{tx}")
    except ImportError:
        logger.warning("pdfplumber not installed")
    except Exception as e:
        logger.warning(f"pdfplumber extraction failed: {e}")

    image_b64 = None
    media_type = None
    try:
        from pdf2image import convert_from_bytes
        pages = convert_from_bytes(file_bytes, dpi=300, first_page=1, last_page=3)
        if pages:
            for page_idx, page in enumerate(pages):
                buf = io.BytesIO()
                page.save(buf, format="PNG")
                img_bytes = buf.getvalue()

                if page_idx == 0:
                    image_b64 = base64.standard_b64encode(img_bytes).decode()
                    media_type = "image/png"

                ocr_text = ocr_image_bytes(img_bytes)
                if ocr_text and ocr_text.strip():
                    text_parts.append(f"[OCR Page {page_idx+1}]\n{ocr_text}")
    except ImportError:
        logger.warning("pdf2image not installed — vision disabled for PDFs")
    except Exception as e:
        logger.warning(f"PDF to image failed: {e}")

    return "\n\n".join(text_parts), image_b64, media_type


def process_upload(file_bytes: bytes, content_type: str) -> tuple[str, str | None, str | None]:
    if content_type == "application/pdf":
        return process_pdf(file_bytes)
    elif content_type.startswith("image/"):
        image_b64 = base64.standard_b64encode(file_bytes).decode()
        ocr_text = ocr_image_bytes(file_bytes)
        return ocr_text, image_b64, content_type
    return "", None, None


def safe_parse_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except Exception:
        pass
    try:
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        return json.loads(cleaned)
    except Exception:
        pass
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass
    logger.error(f"JSON parse failed. Raw: {raw[:500]}")
    return {"parseError": True, "rawResponse": raw[:1000]}


def call_openai(client: OpenAI, messages: list, max_tokens: int = 1500, rid: str = "") -> dict:
    try:
        logger.info(f"[{rid}] OpenAI call | messages={len(messages)} | max_tokens={max_tokens}")
        resp = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=max_tokens,
            messages=messages,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
        result = safe_parse_json(raw)
        usage = resp.usage
        logger.info(f"[{rid}] Tokens used: prompt={usage.prompt_tokens} completion={usage.completion_tokens}")
        return result

    except RateLimitError:
        raise HTTPException(429, detail={
            "error": "rate_limit",
            "message": "OpenAI rate limit hit. Please wait a moment and try again."
        })
    except APIConnectionError:
        raise HTTPException(503, detail={
            "error": "connection_error",
            "message": "Could not connect to OpenAI. Check your internet connection."
        })
    except APIError as e:
        raise HTTPException(502, detail={
            "error": "openai_api_error",
            "message": str(e)
        })


# ─── Triage System Prompt ─────────────────────────────────────────
TRIAGE_SYSTEM = """You are a medical document triage assistant. Analyze the document and respond ONLY with valid JSON.

{
  "documentType": "Prescription | Lab Report | Referral | Discharge Summary | Unknown",
  "patientName": null, "patientAge": null, "gender": null,
  "doctorName": null, "hospital": null, "date": null,
  "symptoms": [], "diagnosis": [], "medicines": [],
  "testValues": [{"name":"","value":"","unit":"","flag":"normal|high|low|critical"}],
  "urgency": "Normal | Attention Needed | Critical",
  "urgencyReason": "1-2 sentence reason",
  "keyFindings": [],
  "summary": "3-4 sentence plain English summary",
  "summaryHindi": "3-4 sentence Hindi summary",
  "recommendedActions": []
}

URGENCY RULES:
Critical: glucose>400/<50, BP>180/120, Hb<7, O2sat<90, K<2.5/>6.5, Na<120/>155, sepsis/stroke/MI signs
Attention Needed: abnormal labs, high-risk meds, specialist referral, pregnancy complications
Normal: routine visit, stable chronic, normal labs

IMPORTANT:
- Extract ALL lab values from ALL pages, not just the first
- If test values are cut off or partial, note them with flag="unknown"
- For medicines, include name + dose + frequency if available

CRITICAL FOR MEDICINES:
- Copy medicine names EXACTLY character by character as written in the document
- NEVER auto-correct spellings — "Braceness" stays "Braceness", "Proflo Ortho" stays "Proflo Ortho"
- These may be brand names, regional products, or uncommon spellings — preserve them exactly
- If handwriting is unclear, write closest match with [unclear] tag
- Do NOT replace with similar known medicine names"""


# ─── Triage Endpoint ──────────────────────────────────────────────
@app.post("/api/triage")
async def triage(
    request: Request,
    file: UploadFile = File(None),
    text: str = Form(""),
):
    rid = getattr(request.state, "rid", "?")
    ip  = get_client_ip(request)
    check_rate_limit(ip)   # ← Rate limit check
    logger.info(f"[{rid}] /api/triage called | file={file.filename if file else None} | ip={ip}")

    try:
        client = get_client()
        msg_content = []
        extracted_text = text.strip()

        if file and file.filename:
            file_bytes = await file.read()
            ct = validate_file(file, file_bytes, ALLOWED_TYPES)
            logger.info(f"[{rid}] Processing upload: {file.filename} | {len(file_bytes)/1024:.1f}KB | {ct}")
            doc_text, image_b64, media_type = process_upload(file_bytes, ct)

            if doc_text:
                extracted_text = (doc_text + "\n\n" + extracted_text).strip()
                logger.info(f"[{rid}] Extracted text: {len(extracted_text)} chars")

            if image_b64 and media_type:
                msg_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{image_b64}",
                        "detail": "high"
                    }
                })

        if not extracted_text and not msg_content:
            raise HTTPException(400, detail={
                "error": "no_content",
                "message": "Please upload a file or provide document text."
            })

        prompt = "Triage this medical document.\n\n"
        if extracted_text:
            prompt += f"DIGITALLY EXTRACTED TEXT (high accuracy, prefer this):\n{extracted_text}\n\n"
        prompt += "The image is also attached if available. For medicine names, preserve exact spelling."

        msg_content.append({"type": "text", "text": prompt})

        messages = [
            {"role": "system", "content": TRIAGE_SYSTEM},
            {"role": "user",   "content": msg_content},
        ]

        result = call_openai(client, messages, max_tokens=1800, rid=rid)
        logger.info(f"[{rid}] Triage complete | urgency={result.get('urgency','?')}")
        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{rid}] Triage unexpected error: {traceback.format_exc()}")
        raise HTTPException(500, detail={"error": "internal_error", "message": str(e)})


# ─── Symptom System Prompt ────────────────────────────────────────
SYMPTOM_SYSTEM = """You are HealthMate — a warm, private health assistant. Users describe symptoms in Hindi, English, or Hinglish.

RULES:
- NOT a doctor. Always mention this.
- Chest pain/breathlessness/severe bleeding/unconsciousness → emergencyAlert=true
- Women's health (periods, PCOS, pregnancy) → dignity and privacy
- Mental health → empathetic, not clinical
- Give DETAILED remedies (e.g. "Garam paani mein adrak aur tulsi 10 min ubaalein, din mein 3 baar piyein")
- OTC medicines → include exact dose and timing
- If user says only one word or query is too vague → needsMoreInfo=true
- IMPORTANT: You have access to the full conversation history. Use context from previous messages.
  If user mentions "woh fever" or "that pain", refer back to earlier messages.

RESPOND ONLY IN THIS JSON:
{
  "severity": "low|medium|high|emergency",
  "needsMoreInfo": false,
  "followUpQuestions": [],
  "possibleCauses": [], "possibleCausesHindi": [],
  "homeRemedies": [], "homeRemediesHindi": [],
  "yogaExercise": [], "yogaExerciseHindi": [],
  "dietAdvice": [], "dietAdviceHindi": [],
  "otcMedicines": [{"name":"","dose":"","note":""}],
  "whenToSeeDoctor": "", "whenToSeeDoctorHindi": "",
  "responseEnglish": "warm 4-5 sentence response that acknowledges context from previous messages if relevant",
  "responseHindi": "warm 4-5 sentence Hindi response",
  "emergencyAlert": false
}"""


# ─── Symptoms Endpoint ────────────────────────────────────────────
@app.post("/api/symptoms")
async def symptoms(
    request: Request,
    message: str = Form(...),
    history: str = Form("[]"),
):
    """
    history format (JSON array):
    [
      {"role": "user", "content": "mujhe bukhaar hai"},
      {"role": "assistant", "content": "I understand you have fever..."},
      ...
    ]
    Frontend sends the FULL conversation as flat role/content pairs.
    Last entry should NOT be the current message — that's passed separately.
    """
    rid = getattr(request.state, "rid", "?")
    ip  = get_client_ip(request)
    check_rate_limit(ip)   # ← Rate limit check
    logger.info(f"[{rid}] /api/symptoms | message_len={len(message)} | ip={ip}")

    if not message.strip():
        raise HTTPException(400, detail={"error": "empty_message", "message": "Message cannot be empty."})

    try:
        client = get_client()

        # Parse history safely
        try:
            hist = json.loads(history) if history else []
            if not isinstance(hist, list):
                hist = []
        except json.JSONDecodeError:
            hist = []

        # Build messages for GPT
        # System prompt first
        msgs = [{"role": "system", "content": SYMPTOM_SYSTEM}]

        # Add conversation history (last 10 messages = 5 turns)
        # History format: [{"role": "user"|"assistant", "content": "..."}]
        for entry in hist[-10:]:
            role = entry.get("role", "")
            content = entry.get("content", "").strip()
            if role in ("user", "assistant") and content:
                msgs.append({"role": role, "content": content})

        # Add current user message
        msgs.append({"role": "user", "content": message.strip()})

        result = call_openai(client, msgs, max_tokens=1500, rid=rid)
        logger.info(f"[{rid}] Symptoms complete | severity={result.get('severity','?')}")
        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{rid}] Symptoms error: {traceback.format_exc()}")
        raise HTTPException(500, detail={"error": "internal_error", "message": str(e)})


# ─── Transcribe Endpoint ──────────────────────────────────────────
@app.post("/api/transcribe")
async def transcribe(
    request: Request,
    audio: UploadFile = File(...),
):
    rid = getattr(request.state, "rid", "?")
    ip  = get_client_ip(request)
    check_rate_limit(ip)
    logger.info(f"[{rid}] /api/transcribe | file={audio.filename}")

    try:
        client = get_client()
        audio_bytes = await audio.read()

        if len(audio_bytes) > MAX_FILE_BYTES:
            raise HTTPException(413, detail={
                "error": "file_too_large",
                "message": f"Audio exceeds {MAX_FILE_SIZE_MB}MB limit."
            })

        ct = (audio.content_type or "").split(";")[0].strip().lower()
        logger.info(f"[{rid}] Audio: {len(audio_bytes)/1024:.1f}KB | {ct}")

        resp = client.audio.transcriptions.create(
            model="whisper-1",
            file=(audio.filename or "audio.wav", audio_bytes, ct or "audio/wav"),
            language=None,
            prompt="Medical terms, symptoms, medicines in Hindi or English",
        )
        logger.info(f"[{rid}] Transcription done: {len(resp.text)} chars")
        return {"text": resp.text, "requestId": rid}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{rid}] Transcribe error: {traceback.format_exc()}")
        raise HTTPException(500, detail={"error": "transcription_failed", "message": str(e)})

# ─── Admin Stats Endpoint ─────────────────────────────────────────
from collections import Counter
import re as _re

ADMIN_KEY = os.getenv("ADMIN_KEY", "rakshak2024")  # .env mein ADMIN_KEY set karo

@app.get("/admin/stats")
async def admin_stats(request: Request):
    key = request.headers.get("X-Admin-Key", "")
    if key != ADMIN_KEY:
        raise HTTPException(403, detail="Forbidden")

    log_file = LOG_DIR / "app.log"
    if not log_file.exists():
        return {"total_requests": 0, "today_requests": 0,
                "unique_ips": 0, "rate_limited": 0,
                "endpoints": {}, "top_ips": [], "recent_logs": []}

    lines = log_file.read_text().splitlines()

    total = 0
    today_count = 0
    ips = []
    rate_limited = 0
    endpoints = Counter()
    recent = []

    today_str = time.strftime("%Y-%m-%d")

    for line in lines:
        # Triage calls
        if "/api/triage called" in line:
            total += 1
            endpoints["triage"] += 1
            ip = _re.search(r"ip=([\d\.]+)", line)
            fname = _re.search(r"file=(.+?) \|", line)
            t = line[:19]
            if today_str in line: today_count += 1
            if ip: ips.append(ip.group(1))
            recent.append({
                "time": t, "endpoint": "/api/triage",
                "ip": ip.group(1) if ip else "?",
                "detail": fname.group(1) if fname else "text input"
            })

        elif "/api/symptoms" in line:
            total += 1
            endpoints["symptoms"] += 1
            ip = _re.search(r"ip=([\d\.]+)", line)
            t = line[:19]
            if today_str in line: today_count += 1
            if ip: ips.append(ip.group(1))
            recent.append({
                "time": t, "endpoint": "/api/symptoms",
                "ip": ip.group(1) if ip else "?",
                "detail": "symptom query"
            })

        elif "/api/transcribe" in line and "Transcription done" not in line:
            total += 1
            endpoints["transcribe"] += 1
            ip = _re.search(r"ip=([\d\.]+)", line)
            t = line[:19]
            if today_str in line: today_count += 1
            if ip: ips.append(ip.group(1))
            recent.append({
                "time": t, "endpoint": "/api/transcribe",
                "ip": ip.group(1) if ip else "?",
                "detail": "audio upload"
            })

        elif "Rate limit hit" in line:
            rate_limited += 1

    ip_counts = Counter(ips)

    return {
        "total_requests": total,
        "today_requests": today_count,
        "unique_ips": len(set(ips)),
        "rate_limited": rate_limited,
        "endpoints": dict(endpoints),
        "top_ips": ip_counts.most_common(8),
        "recent_logs": list(reversed(recent))[:20]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
