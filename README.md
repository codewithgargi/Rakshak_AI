# Rakshak AI — Your Health Guardian

> **AI for Bharat Hackathon** | Bilingual AI-powered health assistant for India

[![Live Demo](https://img.shields.io/badge/Live%20Demo-rakshakai.duckdns.org-00C8B4?style=for-the-badge)](https://rakshakai.duckdns.org)
[![AWS EC2](https://img.shields.io/badge/Deployed%20on-AWS%20EC2-FF9900?style=for-the-badge&logo=amazonaws)](https://aws.amazon.com)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)](https://python.org)

---

## 🩺 What is Rakshak AI?

Rakshak AI is a bilingual, AI-powered health assistant built specifically for India. It bridges the healthcare gap for millions by making medical information accessible in Hindi, English, and Hinglish.

It consists of two powerful modules:

- **MediScan** — Analyzes medical documents (prescriptions, lab reports, discharge summaries) and delivers structured triage recommendations with urgency levels in English and Hindi
- **HealthMate** — A conversational symptom checker that understands Hindi, English, and Hinglish, providing home remedies, OTC medicine suggestions, diet advice, and emergency alerts

---

## ✨ Features

### 📄 MediScan
- Upload PDF, JPG, PNG via drag & drop
- OCR + Vision AI (Tesseract + OpenCV) for text extraction
- Urgency detection — Normal, Attention Needed, Critical
- Medicine names preserved exactly as written
- Triage results in English and Hindi
- Export triage report as PDF

### 💬 HealthMate
- Conversational AI in Hindi, English, Hinglish
- Voice input via Whisper AI
- Home remedies, OTC medicines, diet & yoga advice
- Emergency alert for critical symptoms
- Multi-turn conversation with context memory

### 📊 User Dashboard
- Real-time session stats — scans, chats, critical flags
- Activity timeline & urgency breakdown
- Export session data & clear data option

### 🔒 Admin Dashboard
- Password-protected server-side monitoring
- Total requests, unique IPs, rate limit events
- Endpoint usage stats & recent logs

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, FastAPI, Gunicorn |
| AI | GPT-4o (OpenAI), Whisper AI |
| OCR | Tesseract, OpenCV, Pillow |
| PDF Processing | pdfplumber, pdf2image, PyMuPDF |
| Frontend | HTML, CSS, JavaScript |
| Deployment | AWS EC2, Docker, Nginx, Let's Encrypt |
| DNS | DuckDNS |

---

## 🚀 Setup & Deployment

### Prerequisites
- Python 3.11+
- Docker
- OpenAI API Key

### Local Setup

```bash
# Clone the repository
git clone https://github.com/codewithgargi/Rakshak_AI.git
cd Rakshak_AI

# Create .env file
cp .env .env.local
# Add your keys to .env

# Run with Docker
docker build -t rakshak .
docker run -d -p 8000:8000 --env-file .env --name rakshak-app rakshak
```

### Environment Variables

```env
OPENAI_API_KEY=your_openai_api_key
ADMIN_KEY=your_admin_password
ALLOWED_ORIGINS=http://localhost:8000
```

---

## 🌐 Live Demo

**[https://rakshakai.duckdns.org](https://rakshakai.duckdns.org)**

Deployed on AWS EC2 (eu-north-1) with HTTPS via Let's Encrypt.

---

## 📁 Project Structure

```
Rakshak_AI/
├── main.py              # FastAPI backend
├── Dockerfile           # Docker configuration
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (template)
├── .gitignore
└── static/
    ├── index.html       # Frontend UI
    ├── admin.html       # Admin dashboard
    └── logo1.png        # App logo
```

---

## 🔮 Future Roadmap

- Amazon Bedrock integration for AWS-native AI inference
- Amazon Transcribe for speech-to-text
- Multi-language support (Tamil, Telugu, Bengali)
- Mobile app (React Native)
- Doctor connect & telemedicine booking
- ABDM (Ayushman Bharat Digital Mission) integration

---

## 👩‍💻 Built by

**GARGI GUPTA** | AI for Bharat Hackathon 2026

---

*Making intelligent healthcare accessible to every Indian, in their own language.* 🇮🇳
