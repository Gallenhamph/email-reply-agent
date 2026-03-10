# Sophos AI Sales Engineer Automator

A local-first, privacy-focused AI application designed to act as an automated Sales Engineer. It watches for raw, unstructured voice-to-text meeting transcripts (up to 90+ minutes long), processes them using a heavyweight local LLM (Mistral Small 24B), queries local Sophos/Secureworks PDF datasheets via a native Hybrid Search, and automatically generates context-aware, human-sounding Microsoft Outlook `.eml` drafts.



## 🚀 Key Features

* **100% Local Privacy:** Utilizes local Ollama models and ChromaDB. Customer transcripts and sensitive datasheets never leave your hardware.
* **Heavyweight LLM Power:** Powered by `mistral-small` (24B parameters) configured with an unlocked 32,000 token context window. It effortlessly digests raw, noisy, 90-minute meeting transcripts without losing context, while strictly adhering to XML-fenced prompt rules to mimic your unique writing style.
* **Native Hybrid RAG Architecture:** Bypasses fragile framework dependencies by using a custom-built Reciprocal Rank Fusion (RRF) algorithm. It mathematically combines dense Vector Search (ChromaDB) for conceptual understanding with sparse Keyword Search (BM25) to accurately identify specific Sophos acronyms and product SKUs.
* **Agentic Attachment Filtering (LLM-as-a-Judge):** The LLM autonomously evaluates retrieved PDFs. If a document directly addresses the customer's specific needs, it explicitly commands the backend to attach it. If not, it suppresses the attachment to prevent spam.
* **Intelligent Live Web Enrichment:** Integrates with DuckDuckGo Search to pull live contextual data. Uses a targeted extraction prompt and a human-mimicking delay loop to evade IP rate-limiting while fetching perfect documentation links.
* **Dynamic Phonetic Correction:** Intercepts raw voice-to-text transcripts and uses a customizable `glossary.yml` to instantly correct misheard terms (e.g., "Sofos" -> "Sophos", "manage thread response" -> "MDR") *before* the AI processes the text.
* **Debounced Event Handling:** Features a custom Object-Oriented Watchdog debouncer that gracefully ignores duplicate file-creation events triggered by macOS, preventing duplicate LLM runs and pipeline spam.
* **Automated CI/CD Deployment:** Fully Dockerized and integrated with GitHub Actions and Watchtower for seamless, zero-downtime background updates.

## 📂 Directory Structure

The application dynamically manages the following directories mapped to your local machine:
* `/transcripts/` - Drop your raw `.txt` meeting notes here to trigger the pipeline.
* `/outputs/` - The generated `.eml` Outlook drafts will appear here.
* `/attachments/` - Drop your Sophos/Secureworks PDFs here. They are automatically ingested and chunked into Markdown on startup.
* `/seeds/` - Drop 1-2 `.txt` files containing examples of your personal writing style. The LLM dynamically loads these for Few-Shot style transfer.
* `glossary.yml` - Your custom dictionary for fixing transcription typos.

## 🛠️ Architecture Workflow

1. **Ingestion:** A debounced Watchdog observer detects a new transcript in `/transcripts`.
2. **Correction:** The text is scrubbed using `glossary.yml` to fix transcription errors.
3. **Extraction & Search:** The LLM identifies the top 2 product topics and queries DuckDuckGo for live context.
4. **Retrieval:** The text is queried against local PDFs using Native RRF (ChromaDB + BM25), returning the top 8 highly technical document chunks.
5. **Generation:** Mistral Small synthesizes the transcript, web data, PDF context, and your style seeds to output a uniquely tailored, organic email draft.
6. **Execution:** Python parses the LLM's attachment commands, creates an HTML-rich `.eml` file, attaches the approved PDFs, and drops it in `/outputs`.

## 💻 Deployment & Installation

This project utilizes a Continuous Deployment pipeline via GitHub Container Registry (GHCR) and Watchtower. 

**1. Prepare Your Hardware:**
Ensure your host machine (Mac with Apple Silicon recommended, 36GB+ Unified Memory) has Ollama installed and the correct model pulled:
```bash
ollama pull mistral-small
ollama pull nomic-embed-text
```

**2. Clone the repository and configure your variables.**
Add your local PDFs to the `attachments` folder and your writing samples to the `seeds` folder.

**3. Launch the Stack:**
Use the provided `docker-compose.yml` to launch the application alongside ChromaDB and Watchtower.
```bash
docker compose up -d
```

**4. macOS Automator Integration (Optional but Recommended):**
To achieve a "magic" workflow, configure a macOS Folder Action on the `/outputs` directory to automatically open any new items using Microsoft Outlook.