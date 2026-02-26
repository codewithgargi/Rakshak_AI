"""
HealthTriage - Rural Healthcare Document Triage System
======================================================
AI-powered document triage for rural healthcare workers.
Upload prescriptions, lab reports, or referrals and get
instant urgency assessment and key information extraction.
"""

import streamlit as st
from openai import OpenAI
import base64
import json
import re
from pathlib import Path
from datetime import datetime
from PIL import Image
import io
from dotenv import load_dotenv
load_dotenv()

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HealthTriage · Rural Healthcare Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

/* ── Root & Body ── */
:root {
    --bg:       #0a1628;
    --surface:  #0f2044;
    --card:     #132752;
    --border:   #1e3a6e;
    --accent:   #00d4aa;
    --warn:     #f59e0b;
    --critical: #ef4444;
    --normal:   #10b981;
    --text:     #e2eaf7;
    --muted:    #6b8ab0;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse 80% 50% at 50% -10%, #0d2a5e 0%, #0a1628 60%) !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }

/* ── Hide Streamlit Branding ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Headings ── */
h1, h2, h3 { color: var(--text) !important; font-family: 'Plus Jakarta Sans', sans-serif !important; }

/* ── Upload widget ── */
[data-testid="stFileUploader"] {
    background: var(--card) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
    padding: 8px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}
[data-testid="stFileUploader"] label { color: var(--text) !important; }
[data-testid="stFileUploadDropzone"] { background: transparent !important; }

/* ── Buttons ── */
.stButton > button {
    background: var(--accent) !important;
    color: #0a1628 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 15px !important;
    padding: 12px 28px !important;
    width: 100% !important;
    transition: all 0.2s !important;
    letter-spacing: -0.2px !important;
}
.stButton > button:hover {
    background: #00f0c0 !important;
    box-shadow: 0 4px 20px rgba(0,212,170,0.35) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:disabled {
    opacity: 0.4 !important;
    cursor: not-allowed !important;
    transform: none !important;
}

/* ── Text area ── */
.stTextArea > div > div > textarea {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    line-height: 1.7 !important;
}
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(0,212,170,0.15) !important;
}

/* ── Select box / radio ── */
.stSelectbox > div > div {
    background: var(--card) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
}
.stRadio > div { gap: 8px; }
.stRadio > div > label { color: var(--text) !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 20px 0 !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-weight: 600 !important;
}
.streamlit-expanderContent {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 14px !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.5px; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-size: 20px !important; font-weight: 700 !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)


# ─── Helper: HTML components ─────────────────────────────────────────────────
def header_html():
    return """
    <div style="
        background: rgba(15,32,68,0.9);
        border-bottom: 1px solid #1e3a6e;
        padding: 16px 32px;
        display: flex;
        align-items: center;
        gap: 14px;
        margin: -1rem -1rem 2rem -1rem;
        backdrop-filter: blur(10px);
    ">
        <div style="
            width:42px; height:42px; border-radius:12px;
            background: linear-gradient(135deg, #00d4aa, #0099ff);
            display:flex; align-items:center; justify-content:center;
            font-size:20px; font-weight:900; color:#0a1628;
            font-family: serif; flex-shrink:0;
        ">H</div>
        <div>
            <div style="font-size:18px; font-weight:700; color:#e2eaf7; letter-spacing:-0.3px;">HealthTriage</div>
            <div style="font-size:11px; color:#6b8ab0; font-family:'DM Mono',monospace; letter-spacing:0.5px; text-transform:uppercase;">
                Rural Healthcare Document Assistant
            </div>
        </div>
        <div style="margin-left:auto; display:flex; gap:8px; align-items:center;">
            <span style="
                background:rgba(0,212,170,0.1); border:1px solid rgba(0,212,170,0.3);
                color:#00d4aa; font-size:10px; font-family:'DM Mono',monospace;
                padding:4px 12px; border-radius:20px; letter-spacing:0.5px; text-transform:uppercase;
            ">AI-Powered</span>
            <span style="
                background:rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.3);
                color:#ef4444; font-size:10px; font-family:'DM Mono',monospace;
                padding:4px 12px; border-radius:20px; letter-spacing:0.5px; text-transform:uppercase;
            ">Decision Support Only</span>
        </div>
    </div>
    """

def panel_card(title, icon, content_html):
    return f"""
    <div style="
        background:#0f2044; border:1px solid #1e3a6e;
        border-radius:14px; overflow:hidden; margin-bottom:16px;
    ">
        <div style="
            padding:14px 18px; border-bottom:1px solid #1e3a6e;
            display:flex; align-items:center; gap:10px;
        ">
            <div style="
                width:30px; height:30px; border-radius:8px;
                background:rgba(0,212,170,0.12);
                display:flex; align-items:center; justify-content:center; font-size:14px;
            ">{icon}</div>
            <span style="font-size:13px; font-weight:600; color:#e2eaf7;">{title}</span>
        </div>
        <div style="padding:18px;">{content_html}</div>
    </div>
    """

def urgency_banner(urgency, reason):
    cfg = {
        "Critical":         ("#ef4444", "rgba(239,68,68,0.12)",  "rgba(239,68,68,0.3)",  "🔴"),
        "Attention Needed": ("#f59e0b", "rgba(245,158,11,0.12)", "rgba(245,158,11,0.3)", "🟡"),
        "Normal":           ("#10b981", "rgba(16,185,129,0.12)", "rgba(16,185,129,0.3)", "🟢"),
    }
    color, bg, border, emoji = cfg.get(urgency, cfg["Normal"])
    return f"""
    <div style="
        background:{bg}; border:1px solid {border};
        border-radius:10px; padding:14px 18px;
        display:flex; align-items:flex-start; gap:12px; margin-bottom:16px;
    ">
        <div style="width:10px;height:10px;border-radius:50%;background:{color};margin-top:4px;flex-shrink:0;"></div>
        <div>
            <div style="font-size:14px;font-weight:700;color:{color};font-family:'DM Mono',monospace;letter-spacing:0.5px;">
                {emoji} {urgency.upper()}
            </div>
            <div style="font-size:12px;color:#a0b4cc;margin-top:4px;line-height:1.5;">{reason}</div>
        </div>
    </div>
    """

def info_grid(items):
    """items = list of (label, value) tuples"""
    cells = ""
    for label, value in items:
        cells += f"""
        <div style="background:#132752;border:1px solid #1e3a6e;border-radius:8px;padding:12px;">
            <div style="font-size:10px;color:#6b8ab0;font-family:'DM Mono',monospace;letter-spacing:0.5px;text-transform:uppercase;margin-bottom:4px;">{label}</div>
            <div style="font-size:13px;font-weight:600;color:#e2eaf7;">{value or "—"}</div>
        </div>
        """
    return f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:16px;">{cells}</div>'

def tag(text, style="default"):
    styles = {
        "default": ("rgba(0,212,170,0.08)", "rgba(0,212,170,0.2)", "#00d4aa"),
        "warn":    ("rgba(245,158,11,0.08)", "rgba(245,158,11,0.2)", "#f59e0b"),
        "crit":    ("rgba(239,68,68,0.08)",  "rgba(239,68,68,0.2)",  "#ef4444"),
        "blue":    ("rgba(0,153,255,0.08)",  "rgba(0,153,255,0.2)",  "#4db8ff"),
    }
    bg, border, color = styles.get(style, styles["default"])
    return f"""<span style="background:{bg};border:1px solid {border};color:{color};
        font-size:11px;padding:3px 9px;border-radius:5px;
        font-family:'DM Mono',monospace;display:inline-block;margin:2px;">{text}</span>"""

def tag_section(label, items, style="default"):
    if not items:
        return ""
    tags_html = "".join(tag(i, style) for i in items)
    return f"""
    <div style="margin-bottom:16px;">
        <div style="font-size:10px;color:#6b8ab0;font-family:'DM Mono',monospace;letter-spacing:0.5px;
            text-transform:uppercase;margin-bottom:8px;">{label}</div>
        <div style="display:flex;flex-wrap:wrap;gap:4px;">{tags_html}</div>
    </div>
    """

def summary_box(text):
    return f"""
    <div style="background:#132752;border:1px solid #1e3a6e;border-radius:10px;padding:16px;
        font-size:13px;line-height:1.75;color:#e2eaf7;">{text}</div>
    """

def doc_type_chip(doc_type):
    return f"""<span style="background:rgba(0,153,255,0.1);border:1px solid rgba(0,153,255,0.3);
        color:#4db8ff;font-size:11px;font-family:'DM Mono',monospace;padding:4px 12px;
        border-radius:5px;display:inline-block;margin-bottom:14px;letter-spacing:0.3px;
        text-transform:uppercase;">📄 {doc_type}</span>"""

def footer_note():
    return """
    <div style="text-align:center;font-size:11px;color:#6b8ab0;margin-top:32px;
        font-family:'DM Mono',monospace;opacity:0.7;letter-spacing:0.3px;padding:16px;">
        ⚠️ Decision support only · Not a substitute for professional medical judgment ·
        Always consult a qualified healthcare provider
    </div>
    """


# ─── OCR helpers ─────────────────────────────────────────────────────────────
def extract_text_from_image_bytes(image_bytes: bytes) -> str:
    """OCR with preprocessing for better accuracy on scanned/handwritten docs."""
    try:
        import pytesseract
        from PIL import ImageEnhance, ImageFilter
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("L")                          # Grayscale
        img = ImageEnhance.Contrast(img).enhance(2.5)  # Boost contrast
        img = img.filter(ImageFilter.SHARPEN)           # Sharpen
        return pytesseract.image_to_string(img, config="--psm 6")
    except Exception:
        return ""


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract text from PDF using pdfplumber."""
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        return "\n".join(text_parts)
    except Exception:
        return ""


def image_bytes_to_base64(image_bytes: bytes, media_type: str) -> str:
    return base64.standard_b64encode(image_bytes).decode("utf-8")


# ─── AI Analysis ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a medical document triage assistant for rural healthcare workers in India and similar low-resource settings.

Analyze the provided document (image, OCR text, or pasted text) and respond ONLY with a valid JSON object — no markdown, no explanation, no preamble.

Required JSON format:
{
  "documentType": "Prescription | Lab Report | Referral | Discharge Summary | Unknown",
  "patientName": "string or null",
  "patientAge": "string or null",
  "gender": "string or null",
  "doctorName": "string or null",
  "hospital": "string or null",
  "date": "string or null",
  "symptoms": ["list of symptoms mentioned"],
  "diagnosis": ["list of diagnoses or suspected conditions"],
  "medicines": ["medicine name + dosage if available"],
  "testValues": [
    {"name": "test name", "value": "result value", "unit": "unit if present", "flag": "normal | high | low | critical"}
  ],
  "urgency": "Normal | Attention Needed | Critical",
  "urgencyReason": "1-2 sentence plain English reason for the urgency level",
  "keyFindings": ["3-5 most important findings a healthcare worker must know"],
  "summary": "3-4 sentence plain-language summary a rural healthcare worker can act on immediately",
  "recommendedActions": ["list of suggested next steps based on document content"]
}

URGENCY RULES (apply strictly):
Critical — any ONE of:
  • Blood glucose > 400 or < 50 mg/dL
  • BP > 180/120 mmHg
  • Hemoglobin < 7 g/dL
  • O2 saturation < 90%
  • Potassium < 2.5 or > 6.5 mEq/L
  • Sodium < 120 or > 155 mEq/L
  • Creatinine > 10 mg/dL
  • Signs of sepsis, stroke, MI, or severe infection
  • Emergency referral with urgent language
  • Unconscious / altered consciousness noted
  • Severe allergic reaction / anaphylaxis

Attention Needed — any ONE of:
  • Abnormal lab values not meeting Critical thresholds
  • New or worsening chronic disease
  • Prescription for high-risk medications (warfarin, insulin, chemotherapy)
  • Referral to specialist needed
  • Follow-up in < 1 week recommended
  • Pregnancy-related complications

Normal — routine visit, stable chronic condition, normal lab values, regular follow-up.

Be precise. If information is missing use null or []. Numbers only in value fields, no units in value (put units in unit field)."""


def analyze_with_claude(
    text: str,
    image_bytes: bytes | None = None,
    media_type: str | None = None,
) -> dict:
    """Send document to OpenAI GPT-4o and return parsed triage JSON."""
    client = OpenAI()

    # Build message content
    user_content = []

    # Include image if provided (GPT-4o vision format)
    if image_bytes and media_type:
        b64 = image_bytes_to_base64(image_bytes, media_type)
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{media_type};base64,{b64}",
                "detail": "high",
            },
        })

    # Text payload
    msg = f"Please triage this medical document:\n\n{text.strip()}" if text.strip() else "Please triage this medical document image."
    user_content.append({"type": "text", "text": msg})

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        response_format={"type": "json_object"},  # Force JSON output
    )

    raw = response.choices[0].message.content or ""
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    return json.loads(clean)


# ─── Session State ────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "history" not in st.session_state:
    st.session_state.history = []
if "analyzing" not in st.session_state:
    st.session_state.analyzing = False


# ─── Layout ──────────────────────────────────────────────────────────────────
st.markdown(header_html(), unsafe_allow_html=True)

left_col, right_col = st.columns([1, 1], gap="large")


# ══════════════════════════════════════════════════════════════════
# LEFT COLUMN — Input
# ══════════════════════════════════════════════════════════════════
with left_col:
    st.markdown("""
    <div style="font-size:13px;font-weight:600;color:#e2eaf7;margin-bottom:12px;letter-spacing:-0.2px;">
        📤 Upload Document
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop file here or click to browse",
        type=["pdf", "png", "jpg", "jpeg", "webp", "bmp", "tiff"],
        label_visibility="collapsed",
        help="Supported: PDF, JPG, PNG, WebP, BMP, TIFF",
    )

    # Show file info
    if uploaded_file:
        size_kb = len(uploaded_file.getvalue()) / 1024
        file_icon = "🖼️" if uploaded_file.type.startswith("image") else "📄"
        st.markdown(f"""
        <div style="background:#132752;border:1px solid #1e3a6e;border-radius:8px;
            padding:10px 14px;display:flex;align-items:center;gap:10px;margin-top:8px;">
            <span style="font-size:20px;">{file_icon}</span>
            <div>
                <div style="font-size:12px;color:#e2eaf7;font-family:'DM Mono',monospace;">
                    {uploaded_file.name}
                </div>
                <div style="font-size:10px;color:#6b8ab0;">
                    {size_kb:.1f} KB · {uploaded_file.type}
                </div>
            </div>
            <span style="margin-left:auto;background:rgba(0,212,170,0.1);border:1px solid rgba(0,212,170,0.3);
                color:#00d4aa;font-size:10px;padding:2px 8px;border-radius:4px;font-family:'DM Mono',monospace;">
                READY
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Show image preview
        if uploaded_file.type.startswith("image"):
            with st.expander("👁️ Preview Image"):
                img = Image.open(io.BytesIO(uploaded_file.getvalue()))
                st.image(img, use_column_width=True)

    # Divider
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin:16px 0;color:#6b8ab0;
        font-size:11px;font-family:'DM Mono',monospace;letter-spacing:0.5px;">
        <div style="flex:1;height:1px;background:#1e3a6e;"></div>
        OR PASTE TEXT
        <div style="flex:1;height:1px;background:#1e3a6e;"></div>
    </div>
    """, unsafe_allow_html=True)

    pasted_text = st.text_area(
        "Paste document text",
        placeholder="Paste prescription, lab report values, referral notes, or any medical document text...\n\nExample:\nPatient: Ram Kumar, 58M\nBP: 190/115 mmHg\nBlood Glucose (fasting): 420 mg/dL\nDiagnosis: Hypertensive crisis + Uncontrolled DM Type 2\nReferred to District Hospital immediately.",
        height=200,
        label_visibility="collapsed",
    )

    # API Key (optional override)
    with st.expander("🔑 API Key (optional — uses env var by default)"):
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Leave blank to use OPENAI_API_KEY environment variable",
            label_visibility="collapsed",
        )

    # Analyze button
    can_analyze = bool(uploaded_file or pasted_text.strip())
    analyze_clicked = st.button(
        "🔍  Analyze Document",
        disabled=not can_analyze,
        use_container_width=True,
    )

    # Info boxes
    st.markdown("""
    <div style="background:#0f2044;border:1px solid #1e3a6e;border-radius:10px;
        padding:12px 16px;margin-top:12px;">
        <div style="font-size:11px;color:#6b8ab0;font-family:'DM Mono',monospace;line-height:1.8;">
            ⚡ Real OCR via Tesseract (images)<br>
            📄 PDF text extraction via pdfplumber<br>
            🧠 Claude AI for classification &amp; triage<br>
            🔒 No data stored — session only
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# ANALYSIS LOGIC
# ══════════════════════════════════════════════════════════════════
if analyze_clicked and can_analyze:
    # Override API key if provided
    if api_key_input.strip():
        import os
        os.environ["OPENAI_API_KEY"] = api_key_input.strip()

    with right_col:
        with st.spinner("🧠  Analyzing document with AI..."):
            try:
                image_bytes = None
                media_type = None
                extracted_text = pasted_text.strip()

                if uploaded_file:
                    raw_bytes = uploaded_file.getvalue()
                    file_type = uploaded_file.type

                    if file_type == "application/pdf":
                        # Step 1: Try pdfplumber for text-based PDFs
                        pdf_text = extract_text_from_pdf_bytes(raw_bytes)
                        if pdf_text.strip():
                            extracted_text = pdf_text + ("\n\n" + extracted_text if extracted_text else "")
                        else:
                            # Step 2: Scanned PDF — convert to images, OCR + send to Claude vision
                            try:
                                from pdf2image import convert_from_bytes
                                pages = convert_from_bytes(raw_bytes, dpi=300)
                                ocr_parts = []
                                for i, pg in enumerate(pages):
                                    buf = io.BytesIO()
                                    pg.save(buf, format="PNG")
                                    page_bytes = buf.getvalue()
                                    # Send first page as image to Claude vision
                                    if i == 0:
                                        image_bytes = page_bytes
                                        media_type = "image/png"
                                    # Also run OCR as supplementary text
                                    ocr_result = extract_text_from_image_bytes(page_bytes)
                                    if ocr_result.strip():
                                        ocr_parts.append(ocr_result)
                                if ocr_parts:
                                    extracted_text = "\n".join(ocr_parts) + ("\n\n" + extracted_text if extracted_text else "")
                                if not ocr_parts and image_bytes is None:
                                    st.warning("⚠️ Could not extract text. PDF sent to AI vision directly.")
                            except Exception as pdf_err:
                                st.error(f"❌ PDF OCR failed: {pdf_err}")
                                st.info("Fix: `pip install pdf2image` + `sudo apt-get install poppler-utils` (Linux) or `brew install poppler` (Mac)")

                    elif file_type.startswith("image/"):
                        # Use image directly for vision + run OCR as supplementary text
                        image_bytes = raw_bytes
                        media_type = file_type
                        ocr_text = extract_text_from_image_bytes(raw_bytes)
                        if ocr_text.strip():
                            extracted_text = ocr_text + ("\n\n" + extracted_text if extracted_text else "")

                if not extracted_text.strip() and image_bytes is None:
                    st.error("❌ No content to analyze. Please upload a file or paste text.")
                    st.stop()

                result = analyze_with_claude(extracted_text, image_bytes, media_type)
                st.session_state.result = result

                # Add to history
                st.session_state.history.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "filename": uploaded_file.name if uploaded_file else "Pasted Text",
                    "urgency": result.get("urgency", "Unknown"),
                    "docType": result.get("documentType", "Unknown"),
                    "patient": result.get("patientName") or "Unknown",
                })

            except json.JSONDecodeError:
                st.error("❌ AI returned malformed response. Please try again.")
                st.stop()
            except Exception as e:
                err = str(e).lower()
                if "auth" in err or "api key" in err or "401" in err or "incorrect api key" in err:
                    st.error("❌ Invalid API key. Please check your OPENAI_API_KEY.")
                else:
                    st.error(f"❌ Error: {str(e)}")
                st.stop()


# ══════════════════════════════════════════════════════════════════
# RIGHT COLUMN — Results (native Streamlit widgets, no CSS grid)
# ══════════════════════════════════════════════════════════════════
with right_col:
    st.markdown("#### 📊 Triage Results")

    result = st.session_state.result

    if result is None:
        st.info("🩺 No document analyzed yet. Upload a file or paste text, then click **Analyze**.")

    else:
        urgency = result.get("urgency", "Normal")
        doc_type = result.get("documentType", "Unknown")

        # ── Urgency banner via st.success / warning / error ──
        urgency_reason = result.get("urgencyReason", "")
        urgency_msg = f"**{doc_type}** · {urgency_reason}"
        if urgency == "Critical":
            st.error(f"🔴 CRITICAL — {urgency_msg}")
        elif urgency == "Attention Needed":
            st.warning(f"🟡 ATTENTION NEEDED — {urgency_msg}")
        else:
            st.success(f"🟢 NORMAL — {urgency_msg}")

        # ── Patient info — 3 cols ──
        c1, c2, c3 = st.columns(3)
        c1.metric("Patient", result.get("patientName") or "—")
        c2.metric("Age", result.get("patientAge") or "—")
        c3.metric("Gender", result.get("gender") or "—")

        c4, c5, c6 = st.columns(3)
        c4.metric("Doctor", result.get("doctorName") or "—")
        c5.metric("Hospital / Clinic", result.get("hospital") or "—")
        c6.metric("Date", result.get("date") or "—")

        st.divider()

        # ── Symptoms ──
        symptoms = result.get("symptoms", [])
        if symptoms:
            st.markdown("**🤒 Symptoms**")
            st.markdown("  ".join(f"`{s}`" for s in symptoms))

        # ── Diagnosis ──
        diagnosis = result.get("diagnosis", [])
        if diagnosis:
            st.markdown("**🔬 Diagnosis / Suspected Conditions**")
            st.markdown("  ".join(f"`{d}`" for d in diagnosis))

        # ── Medicines ──
        medicines = result.get("medicines", [])
        if medicines:
            st.markdown("**💊 Medicines Prescribed**")
            st.markdown("  ".join(f"`{m}`" for m in medicines))

        # ── Lab Values — dataframe ──
        test_values = result.get("testValues", [])
        if test_values:
            import pandas as pd
            st.markdown("**🧪 Lab / Test Values**")
            flag_emoji = {"critical": "🔴 CRITICAL", "high": "🟡 HIGH", "low": "🟡 LOW", "normal": "🟢 Normal"}
            rows = []
            for tv in test_values:
                flag = (tv.get("flag") or "normal").lower()
                rows.append({
                    "Test":   tv.get("name", ""),
                    "Value":  tv.get("value", ""),
                    "Unit":   tv.get("unit", ""),
                    "Status": flag_emoji.get(flag, flag),
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

        st.divider()

        # ── Key Findings ──
        findings = result.get("keyFindings", [])
        if findings:
            st.markdown("**⚡ Key Findings**")
            for f in findings:
                if urgency == "Critical":
                    st.markdown(f"🔴 {f}")
                elif urgency == "Attention Needed":
                    st.markdown(f"🟡 {f}")
                else:
                    st.markdown(f"✅ {f}")

        # ── Recommended Actions ──
        actions = result.get("recommendedActions", [])
        if actions:
            st.markdown("**✅ Recommended Actions**")
            for i, a in enumerate(actions, 1):
                st.markdown(f"{i}. {a}")

        st.divider()

        # ── AI Summary ──
        st.markdown("**🤖 AI Summary for Healthcare Worker**")
        st.info(result.get("summary", ""))

        # ── Export ──
        st.download_button(
            label="⬇️  Export Full Report (JSON)",
            data=json.dumps(result, indent=2),
            file_name=f"triage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════
# HISTORY PANEL
# ══════════════════════════════════════════════════════════════════
if st.session_state.history:
    st.divider()
    st.markdown("#### 🕑 Session History")
    import pandas as pd
    urgency_emoji = {"Critical": "🔴", "Attention Needed": "🟡", "Normal": "🟢"}
    hist_rows = []
    for entry in reversed(st.session_state.history):
        hist_rows.append({
            "Time":     entry["timestamp"],
            "File":     entry["filename"],
            "Patient":  entry["patient"],
            "Type":     entry["docType"],
            "Urgency":  urgency_emoji.get(entry["urgency"], "") + " " + entry["urgency"],
        })
    st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)


# ── Footer ──
st.markdown(footer_note(), unsafe_allow_html=True)