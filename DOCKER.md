# Docker — Build, Run, and Deploy

🐳 [Docker Hub](https://hub.docker.com/repository/docker/shivamrajput130/enterprise-rag-assistant/general)

---

## Prerequisites

Two credentials are required before running:

| Variable | Required | Where to get it |
|---|---|---|
| `GROQ_API_KEY` | Yes — all LLM calls fail without it | [console.groq.com](https://console.groq.com) |
| `SUPABASE_URL` | Yes for feedback analytics | Supabase project settings → API |
| `SUPABASE_KEY` | Yes for feedback analytics | Supabase project settings → API |

If Supabase is not configured, the RAG pipeline and Pandas Analytics Engine still work. Only the feedback buttons will show an error.

---

## Quickstart — Pull from Docker Hub

```bash
# Pull the image
docker pull shivamrajput130/enterprise-rag-assistant:latest

# Create your .env file
cp .env.example .env
# Open .env and fill in your credentials

# Run the container
docker run \
  --name enterprise-rag-assistant \
  --env-file .env \
  -p 8000:8000 \
  -p 8501:8501 \
  shivamrajput130/enterprise-rag-assistant:latest
```

**Access:**
- Streamlit UI: `http://localhost:8501`
- FastAPI backend: `http://localhost:8000`
- API docs (Swagger): `http://localhost:8000/docs`

**Verify the container is running:**
```bash
curl http://localhost:8000/health
```
Expected response:
```json
{"status": "healthy"}
```

---

## .env File

```bash
# Copy the example file
cp .env.example .env
```

`.env.example` (already in repo — copy and fill in your values):
```
GROQ_API_KEY=
SUPABASE_URL=
SUPABASE_KEY=
```
```
GROQ_API_KEY=your_groq_api_key_here
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_key_here
```

> **Note:** `SUPABASE_URL` and `SUPABASE_KEY` are used via the Supabase Python client. If your `feedback_db.py` connects via `psycopg2` directly, replace the above with:
> ```
> DB_HOST=your-project.supabase.co
> DB_PORT=5432
> DB_NAME=postgres
> DB_USER=postgres
> DB_PASSWORD=your_db_password
> ```
> Verify against your actual `.env.example` in the repo.

The `.env` file is in `.gitignore` — it will not be pushed to GitHub. Never put credentials directly in the Dockerfile or in the Docker run command as plain arguments.

---

## Persist Uploaded Documents

By default, documents uploaded inside the container are lost when the container stops. To persist them across restarts:

```bash
docker run --env-file .env \
  -p 8000:8000 \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  shivamrajput130/enterprise-rag-assistant:latest
```

This mounts your local `data/` directory into the container. ChromaDB and uploaded documents survive container restarts.

---

## Build Locally

To build the image from source instead of pulling from Docker Hub:

> **Note:** First build may take several minutes — PyTorch, Transformers, and Sentence Transformers are large dependencies (~4.3 GB total image size).

```bash
git clone https://github.com/shivamrajput-ds/enterprise-rag-assistant
cd enterprise-rag-assistant

docker build -t enterprise-rag-assistant .
```

Run the locally built image:
```bash
docker run --env-file .env \
  -p 8000:8000 \
  -p 8501:8501 \
  enterprise-rag-assistant
```

---

## Dockerfile Overview

```dockerfile
# Base image
FROM python:3.11-slim

# Install uv for fast dependency installation
RUN pip install uv

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN uv pip install --system -r requirements.txt

# Copy source
COPY . .

# Expose ports
EXPOSE 8000 8501

# Start both FastAPI and Streamlit via start.sh
CMD ["bash", "start.sh"]
```

`start.sh` starts both the FastAPI backend and the Streamlit UI in the same container. FastAPI runs on port 8000, Streamlit on port 8501.

---

## Common Issues

**Container starts but UI is not accessible:**
Confirm both ports are mapped with `-p 8000:8000 -p 8501:8501`. If running on a remote server or VM, check firewall rules for those ports.

**`GROQ_API_KEY not set` error:**
Verify your `.env` file has the key set correctly and you are passing `--env-file .env` to the run command. Check for extra spaces or quotes around the value.

**Supabase connection error:**
Verify `SUPABASE_URL` and `SUPABASE_KEY` in `.env`. The URL should start with `https://`. The key should be the `anon` key from Supabase project settings, not the `service_role` key.

**ChromaDB data not persisting:**
Add the `-v $(pwd)/data:/app/data` volume mount to your run command.

**`vertexai` import error during build:**
This surfaced from RAGAS internal dependencies. Resolved by pinning `ragas==0.2.10` in `requirements.txt`. If the error reappears, verify the pin is in place.

**Port 8000 or 8501 already in use:**
Find and stop the process using the port:
```bash
# See what's running
docker ps

# Stop a specific container
docker stop <container_id>

# Or on Windows, check which process is using the port
netstat -ano | findstr :8000
```
Then re-run the container.

---

## Image Information

| Property | Value |
|---|---|
| Docker Image Size | ~4.3 GB |
| Base Image | `python:3.11-slim` |
| Exposed Ports | 8000 (FastAPI), 8501 (Streamlit) |

First build may take several minutes — PyTorch, Transformers, and Sentence Transformers are large dependencies.

---

## Docker Hub

Image repository: [hub.docker.com/r/shivamrajput130/enterprise-rag-assistant](https://hub.docker.com/repository/docker/shivamrajput130/enterprise-rag-assistant/general)

Tags:
- `latest` — most recent stable build

To push a new version (maintainer only):
```bash
docker build -t shivamrajput130/enterprise-rag-assistant:latest .
docker push shivamrajput130/enterprise-rag-assistant:latest
```

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | Yes | None | Groq API key for all LLM calls |
| `SUPABASE_URL` | For feedback | None | Supabase project URL |
| `SUPABASE_KEY` | For feedback | None | Supabase anon key |

No other environment variables are required. All pipeline parameters (chunk size, retrieval top-k, model names) are configured via `config.yaml` inside the image.