# INCOGNITOVISION - AI Security Camera System

AI-powered security camera system with person detection and smart alerting.

## Hardware
- **Board**: RDK X5 Development Board (8GB)
- **Camera**: Raspberry Pi 8MP Camera Module V2

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Frontend      │────▶│    Backend      │◀────│   AI Service    │
│   (React)       │     │   (Node.js)     │     │   (Flask/YOLO)  │
│   Port: 5173    │     │   Port: 3000    │     │   Port: 5000    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                              │
                              ▼
                        Firebase RTDB
```

## Features
- 🔍 Person detection (known: Aum, Prem, Auto)
- 🚨 Unknown person alerts with smart cooldown
- 📹 Live video streaming
- 📋 Incident logging to Firebase
- 🖥️ React dashboard with alerts panel

## Quick Start

### AI Service (Python)
```bash
cd ai_service
pip install -r requirements.txt
python ai_server.py
```

### Backend (Node.js)
```bash
cd backend
npm install
node server.js
```

### Frontend (React)
```bash
cd frontend
npm install
npm run dev
```

## RDK X5 Deployment

1. Connect via SSH:
   ```bash
   ssh sunrise@<board-ip>
   ```

2. Clone repository:
   ```bash
   git clone https://github.com/<your-repo>/securitycam-py-repo.git
   ```

3. Install dependencies and run AI service:
   ```bash
   cd securitycam-py-repo/ai_service
   pip3 install -r requirements.txt
   python3 ai_server.py
   ```

## License
Private Project - INCOGNITOVISION
