# Lex Bot v2 - Indian Law Research Assistant

A production-ready RAG (Retrieval-Augmented Generation) bot for Indian Law, built with FastAPI, LangGraph, and Google Gemini.

## Prerequisites

- **Python 3.10+**
- **PostgreSQL 15+** (with `pgvector` extension)

## Setup

1.  **Clone the repository**
    ```bash
    git clone <repository-url>
    cd law_chat
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Configuration**
    Create a `.env` file in `lex_bot/` (or root) with your API keys and database URL:
    ```env
    GOOGLE_API_KEY=your_google_api_key
    DATABASE_URL=postgresql://postgres:password@localhost:5432/rag_db
    # ... other keys as needed
    ```

## Database Setup

You need a running PostgreSQL instance with the `pgvector` extension.

### Option A: Docker (Recommended)
Use the provided `docker-compose.yml` to start a pre-configured database:
```bash
docker compose up -d
```

### Option B: Manual / Local Postgres
1.  Install PostgreSQL.
2.  Create a database (e.g., `rag_db`).
3.  Enable the extension:
    ```sql
    CREATE EXTENSION IF NOT EXISTS vector;
    ```
4.  Update `DATABASE_URL` in your `.env` to point to your local instance.

## Running the Application

Run the application directly with Python:

```bash
python -m lex_bot.app
```

The API will be available at `http://localhost:8000`.
Docs are available at `http://localhost:8000/docs`.

## Deployment / Hosting

Since the app requires `pgvector`, you have two main strategies for hosting the database without Docker:

### 1. Managed Database (Recommended)
Use a cloud provider that supports `pgvector` out of the box. You simply provide the connection string (`DATABASE_URL`).
- **Supabase**: Supports `pgvector` by default.
- **Neon**: Supports `pgvector`.
- **AWS RDS / Google Cloud SQL / Azure**: Supported on newer Postgres versions.
- **Render / Railway**: Often supported in their managed Postgres plugins.

**Setup**:
1. Create a database instance on the provider.
2. Run `CREATE EXTENSION vector;` in the provider's SQL editor.
3. Copy the connection string to your production `.env`.

### 2. Self-Hosted (VPS)
If you rent a Linux server (e.g., EC2, DigitalOcean):
1. Install Postgres: `sudo apt install postgresql-15`
2. Install the extension: `sudo apt install postgresql-15-pgvector`
3. Configure the app as usual.
