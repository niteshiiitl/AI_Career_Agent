# Backend Patch Summary

## Date
Applied as part of workflow step `wf_b73106a1470c9ef7`.

## Files Changed

### `api.py`
1. **Imports** — Extended the `fastapi` import line to include `UploadFile` and `File`. Added `VECTOR_STORE_PATH` to the `core.retrieval` import.
2. **`POST /upload-resume`** — New endpoint that accepts a multipart file upload (`.pdf`, `.docx`, or `.txt`). Extracts text using `pypdf` for PDFs, `python-docx` for Word documents, and plain UTF-8 decoding for text files. Rejects files with fewer than 50 characters of extracted text. Passes extracted text to `run_agent` and returns a `JobMatchResponse`.
3. **`GET /vector-db/status`** — New endpoint that queries the live ChromaDB collection and returns its document count, embedding model name, vector store path, and a `"ready"` status flag.

### `requirements.txt`
Added two new package pins at the bottom:
- `pypdf>=4.0.0` — PDF text extraction for `/upload-resume`.
- `python-docx>=1.1.0` — DOCX text extraction for `/upload-resume`.

## What was NOT changed
- No existing endpoints were modified.
- No other files were touched.
- The server was not started.
