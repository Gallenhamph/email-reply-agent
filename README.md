# Sophos AI Sales Engineer Automator

A local-first, privacy-focused AI application designed to act as an automated Sales Engineer. It watches for raw, unstructured voice-to-text meeting transcripts (up to 90+ minutes long), processes them using a heavyweight local LLM (Mistral Small 24B), queries local Sophos/Secureworks PDF datasheets via a native Hybrid Search, and automatically generates context-aware, human-sounding Microsoft Outlook `.eml` drafts.

## 🚀 Key Features

* **100% Local Privacy:** Utilizes local Ollama models and ChromaDB. Customer transcripts and sensitive datasheets never leave your hardware.
* **Heavyweight LLM Power:** Powered by `mistral-small` (24B parameters) configured with an unlocked 32,000 token context window. It effortlessly digests raw, noisy, 90-minute meeting transcripts without losing context, while strictly adhering to XML-fenced prompt rules to mimic your unique writing style.
* **Portable Identity Configuration:** Decoupled architecture allows any Sales Engineer to deploy this app and personalize the AI's identity simply by editing standard environment variables.
* **Native Hybrid RAG Architecture:** Bypasses fragile framework dependencies by using a custom-built Reciprocal Rank Fusion (RRF) algorithm. It mathematically combines dense Vector Search (ChromaDB) for conceptual understanding with sparse Keyword Search (BM25) to accurately identify specific Sophos acronyms and product SKUs.
* **Agentic Attachment Filtering (LLM-as-a-Judge):** The LLM autonomously evaluates retrieved PDFs. If a document directly addresses the customer's specific needs, it explicitly commands the backend to attach it. If not, it suppresses the attachment to prevent spam.
* **Intelligent Live Web Enrichment:** Integrates with DuckDuckGo Search to pull live contextual data. Uses a targeted extraction prompt and a human-mimicking delay loop to evade IP rate-limiting while fetching perfect documentation links.
* **Debounced Event Handling:** Features a custom Object-Oriented Watchdog debouncer that gracefully ignores duplicate file-creation events triggered by macOS, preventing duplicate LLM runs and pipeline spam.

## 📂 Directory Structure

The application dynamically manages the following directories mapped to your local machine:
* `/transcripts/` - Drop your raw `.txt` meeting notes here to trigger the pipeline.
* `/outputs/` - The generated `.eml` Outlook drafts will appear here.
* `/attachments/` - Drop your Sophos/Secureworks PDFs here. They are automatically ingested and chunked into Markdown on startup.
* `/seeds/` - Drop 1-2 `.txt` files containing examples of your personal writing style. 
* `glossary.yml` - Your custom dictionary for fixing transcription typos (e.g., "Sofos" -> "Sophos").

## 💻 Deployment & Installation

This project utilizes a Continuous Deployment pipeline via GitHub Container Registry (GHCR) and Watchtower, making it incredibly easy to distribute to other engineers.

**1. Prepare Your Hardware:**
Ensure your host machine (Mac with Apple Silicon recommended, 36GB+ Unified Memory) has [Ollama](https://ollama.com/) installed and the correct models pulled:
```bash
ollama pull mistral-small
ollama pull nomic-embed-text
```

**2. Clone the repository and configure your identity:**
Open the `docker-compose.yml` file and update the `Environment` section with your actual name and company:
```yaml
    environment:
      # --- USER CONFIGURATION ---
      - SE_NAME=Your First and Last Name
      - SE_COMPANY=Your Company Name
      # --------------------------
```

**3. Load your Knowledge Base:**
Add your local PDFs to the `attachments` folder and your writing samples to the `seeds` folder. (For best results, only use 1-2 highly refined seed emails so the AI doesn't over-index on them).

**4. Launch the Stack:**
Use the provided `docker-compose.yml` to launch the application alongside ChromaDB and Watchtower.
```bash
docker compose up -d
```