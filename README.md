# 🛠️ PatchOps – AI-Powered Incident Resolution Hub

PatchOps is an AI-powered web application that helps engineering teams analyze production incidents, identify root causes, generate remediation suggestions, and automate incident response workflows.

🔗 **Live Demo:** http://patchops-env.eba-d4azdp3d.ap-south-1.elasticbeanstalk.com/

> **Note:** This is a public demo. Please register a new account for testing and avoid using real production credentials.

---

## 🚀 Features

- 🤖 AI-powered Root Cause Analysis using Google Gemini
- ⚡ Real-time streaming responses (Server-Sent Events)
- 📊 Interactive architecture diagrams using Mermaid.js
- 💻 AI-generated Git diff patches
- 💬 Slack integration
- 🎫 Jira ticket creation
- 🔀 GitHub Pull Request generation
- 📄 Incident post-mortem generation (Markdown & PDF)
- 📈 Incident analytics dashboard
- 🔐 JWT-based user authentication

---

## 🛠️ Tech Stack

**Frontend**
- HTML
- CSS
- JavaScript
- Tailwind CSS

**Backend**
- Python
- FastAPI

**Database**
- SQLite
- SQLAlchemy

**Authentication**
- JWT
- bcrypt

**AI**
- Google Gemini API

**Deployment**
- Docker
- AWS Elastic Beanstalk

---

## 📂 Project Structure

```text
PatchOps/
│── templates/
│── main.py
│── auth.py
│── incident_store.py
│── requirements.txt
│── Dockerfile
└── README.md
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/your-username/patchops.git
cd patchops
```

### Create a virtual environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux/macOS**

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Create a `.env` file

```env
GEMINI_API_KEY=YOUR_API_KEY
GEMINI_MODEL=gemini-3.5-flash
JWT_SECRET_KEY=YOUR_SECRET_KEY
```

### Run the application

```bash
python main.py
```

or

```bash
uvicorn main:app --reload
```

Open:

```
http://localhost:8000
```

---

## 🐳 Run with Docker

Build the image:

```bash
docker build -t patchops .
```

Run the container:

```bash
docker run -p 8000:8000 patchops
```

Open:

```
http://localhost:8000
```

---

## ☁️ Deployment

The application is containerized using Docker and deployed on **AWS Elastic Beanstalk**.



 licensed under the MIT License.
