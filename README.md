# Sophos AI Sales Engineer Automator

A local-first, privacy-focused AI application designed to act as an automated Sales Engineer. It watches for raw, voice-to-text meeting transcripts, processes them using a local LLM (Llama 3.1), queries local Sophos/Secureworks PDF datasheets via Hybrid Search, and automatically generates context-aware, ready-to-send Microsoft Outlook `.eml` drafts.

## 🚀 Key Features

* **100% Local Privacy:** Utilizes local Ollama models and ChromaDB to ensure customer transcripts and sensitive datasheets never leave your hardware.
* **Hybrid RAG Architecture:** Combines dense Vector Search (ChromaDB) for conceptual understanding with sparse Keyword Search (BM25) to accurately identify specific Sophos acronyms and product SKUs.
* **Agentic Attachment Filtering (LLM-as-a-Judge):** The LLM autonomously evaluates retrieved PDFs. If a document is highly relevant, it explicitly commands the Python backend to attach it. If not, it suppresses the attachment to prevent spamming the customer.
* **Dynamic Phonetic Correction:** Intercepts raw voice-to-text transcripts and uses a customizable `glossary.yml` to instantly correct misheard terms (e.g., "Sofos" -> "Sophos", "manage thread response" -> "MDR") *before* the AI processes the text.
* **Live Web Enrichment:** Integrates with DuckDuckGo Search to pull live contextual data and public documentation links into the email draft.
* **Automated CI/CD Deployment:** Fully Dockerized and integrated with GitHub Actions and Watchtower for seamless, zero-downtime background updates.

## 📂 Directory Structure

The application dynamically manages the following directories mapped to your local machine:
* `/transcripts/` - Drop your `.txt` meeting notes here to trigger the pipeline.
* `/outputs/` - The generated `.eml` Outlook drafts will appear here.
* `/attachments/` - Drop your Sophos/Secureworks PDFs here. They are automatically ingested and chunked into Markdown on startup.
* `/seeds/` - Drop `.txt` files containing examples of your personal writing style. The LLM dynamically loads these for Few-Shot prompting.

## 🛠️ Architecture Workflow

1. **Ingestion:** A Watchdog observer detects a new transcript in `/transcripts`.
2. **Correction:** The text is scrubbed using `glossary.yml` to fix transcription errors.
3. **Retrieval:** The text is queried against local PDFs using a Reciprocal Rank Fusion of ChromaDB and BM25.
4. **Enrichment:** DuckDuckGo queries the web for up-to-date links.
5. **Generation:** Llama 3.1 synthesizes the context and outputs an email draft.
6. **Execution:** Python parses the LLM's attachment commands, creates an HTML-rich `.eml` file, attaches the approved PDFs, and drops it in `/outputs`.

## 💻 Deployment & Installation

This project utilizes a Continuous Deployment pipeline via GitHub Container Registry (GHCR) and Watchtower. 

**1. Clone the repository and configure your variables.**
Add your local PDFs to the `attachments` folder and your writing samples to the `seeds` folder.

**2. Launch the Stack:**
Use the provided `docker-compose.yml` to launch the application alongside ChromaDB and Watchtower. Ensure Watchtower is configured with your specific label environment variables.
```bash
docker compose up -d
```

**3. macOS Automator Integration (Optional but Recommended):**
To achieve a "magic" workflow, configure a macOS Folder Action on the `/outputs` directory to automatically open any new items using Microsoft Outlook.