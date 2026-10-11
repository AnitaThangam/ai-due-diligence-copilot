# AI Due Diligence Copilot

A beginner-friendly Retrieval-Augmented Generation (RAG) application that searches financial and business PDFs and generates answers grounded in retrieved passages.

> Educational portfolio prototype. It is not investment, legal, or accounting advice. Verify claims and calculations against original documents.

## Features
- Upload PDF annual reports, filings, and investor presentations
- Extract text with page-level source metadata
- Split text into overlapping chunks
- Create semantic embeddings with Sentence Transformers
- Store and retrieve chunks with ChromaDB
- Generate answers using Groq's chat API
- Display retrieved passages, source filenames, and page numbers
- Unit tests for document chunking

## Architecture
PDF → text extraction → chunking → embeddings → ChromaDB vector search → retrieved context → Groq LLM → answer with source references.

## Requirements
- Windows 10/11
- Python 3.11 recommended
- Internet access for package/model downloads and Groq API calls
- A Groq API key for answer generation

## Setup on Windows / VS Code
Open this folder in VS Code, then select Terminal → New Terminal.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and replace the placeholder with your Groq API key. Do not share or commit this file.

Run the app:

```powershell
streamlit run app.py
```

Upload one or more PDFs in the sidebar, click **Process uploaded PDFs**, then ask a question.

## Sample PDFs
The `data/sample_pdfs/` folder contains synthetic sample company reports created for this project. The data is fictional and should not be treated as actual company information.

## Run tests
```powershell
pytest -q
```

## GitHub preparation
```powershell
git init
git add .
git commit -m "Build AI Due Diligence Copilot RAG prototype"
```
Create an empty GitHub repository, then follow GitHub's instructions to add the remote and push. Never commit `.env`, API keys, uploaded confidential files, or `chroma_db/`.

## Current limitations
- Extracts selectable PDF text only; scanned PDFs require OCR
- Embedding model downloads the first time it runs
- Uses semantic retrieval, not a specialized financial table parser
- LLM responses may be wrong; citations and source passages must be checked
- Financial calculations should be computed from structured figures, not guessed by an LLM
- No authentication or multi-user isolation; run locally with non-confidential documents

## Suggested next steps
1. Add a small manually verified question-answer evaluation set.
2. Evaluate retrieval relevance and citation support.
3. Add structured extraction and explicit calculations for financial metrics.
4. Add optional Azure/AWS deployment only after the local version is stable.
