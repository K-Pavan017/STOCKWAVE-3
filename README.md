# StockWave 🚀

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green.svg)](https://www.python.org/)
[![Node.js](https://img.shields.io/badge/Node.js-18%2B-blue.svg)](https://nodejs.org/)

---

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Architecture Diagram](#architecture-diagram)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
- [Running the Application](#running-the-application)
- [API Reference](#api-reference)
- [Environment Variables](#environment-variables)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview
StockWave is a full‑stack web application that provides real‑time stock data retrieval, interactive visualisations, and AI‑driven price predictions. The backend, built with **Flask**, exposes a RESTful API, while the frontend, built with **React** and **Vite**, delivers a responsive UI.

---

## Key Features
- **Live stock data** – fetches latest quotes from Yahoo Finance.
- **Historical data visualisation** – interactive charts for up to 12 months of data.
- **AI predictions** – LSTM model generates short‑term price forecasts.
- **User authentication** – secure signup/login with JWT.
- **Rate limiting & caching** – Redis‑backed rate limiter and response cache for high performance.
- **Docker‑ready** – can be containerised for production deployment.

---

## Technology Stack
| Layer | Technology | Reason |
|-------|------------|--------|
| **Backend** | Python 3.9+, Flask, Flask‑RESTful, SQLAlchemy, Alembic, Redis, Waitress | Lightweight, extensible REST API framework with robust ORM support. |
| **Frontend** | React 18, Vite, TailwindCSS, Chart.js | Modern reactive UI with fast hot‑module replacement. |
| **Database** | SQLite (development) / PostgreSQL (production) | Simple file‑based DB for dev; scalable relational DB for prod. |
| **Caching & Rate Limiting** | Redis | In‑memory store for low‑latency caching and distributed rate limiting. |
| **AI Model** | TensorFlow/Keras (LSTM) | Proven time‑series forecasting model. |

---

## Architecture Diagram
> **⚠️ Replace the placeholder below with an actual diagram**

```
[Architecture Diagram Placeholder]
```

The diagram should illustrate the interaction between the client (React), the API gateway (Flask), the database, Redis cache, and the prediction service.

---

## Getting Started
### Prerequisites
- **Python 3.9+**
- **Node.js 18+** and **npm**
- **Redis** server (optional for caching; the app falls back to in‑memory if unavailable)
- **Git** (to clone the repository)

### Backend Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/stockwave-3.git
cd stockwave-3/backend

# Create a virtual environment
python -m venv venv
# Activate (Windows)
venv\Scripts\activate
# Or (Linux/macOS)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Initialize the database
flask db upgrade   # Applies Alembic migrations
```

### Frontend Setup
```bash
cd ../frontend
npm install
```

---

## Running the Application
### Development Mode
```bash
# In one terminal – start Redis (if you have it installed)
redis-server

# Backend (run with hot‑reload using Waitress’s dev mode)
cd ../backend
python app.py   # runs on http://localhost:8080

# Frontend (Vite dev server)
cd ../frontend
npm run dev   # runs on http://localhost:5173
```

### Production (Docker)
```bash
# Build Docker images
docker compose build
# Run containers
docker compose up -d
```

---

## API Reference
All endpoints are prefixed with `/api` and return JSON.

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| `POST` | `/signup` | Register a new user. | ❌ |
| `POST` | `/login` | Authenticate and receive a JWT. | ❌ |
| `POST` | `/stock/fetch` | Pull latest stock data for a symbol. | ✅ |
| `GET` | `/stock/data/<symbol>` | Retrieve historical records (cached). | ✅ |
| `GET` | `/stock/predict/<symbol>` | Get AI‑generated prediction. Supports `horizon` query (`day`, `week`, `month`, `3month`). | ✅ |

**Error handling** – All responses include a `success` boolean and a `message` field.

---

## Environment Variables
Create a `.env` file in the `backend` directory (see `.env.example` for reference).
```
FLASK_ENV=development
SECRET_KEY=your_secret_key
SQLALCHEMY_DATABASE_URI=sqlite:///site.db   # or PostgreSQL URI
REDIS_URL=redis://localhost:6379/0
JWT_SECRET_KEY=your_jwt_secret
```

---

## Screenshots
> Replace the placeholders below with actual screenshots of the UI.

![Home page placeholder](assets/home_placeholder.png)
![Stock chart placeholder](assets/chart_placeholder.png)

---

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/awesome-feature`).
3. Write tests for new functionality.
4. Ensure linting passes (`npm run lint` & `flake8`).
5. Submit a pull request with a clear description of changes.

Please adhere to the **Code of Conduct** located in `CODE_OF_CONDUCT.md`.

---

## License
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## Contact
**Pavan** – [GitHub](https://github.com/yourusername) – pavan@example.com

Feel free to open an issue for bugs or feature requests.

---

*Last updated: $(date -u +'%Y-%m-%d')*
