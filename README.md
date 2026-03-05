# AI Sales Engineer Email Automator (Local RAG)

This application automatically generates professional, context-aware follow-up emails based on meeting transcripts. It runs entirely locally for maximum data privacy, utilizing an M-series Mac, Ollama, Docker, and LangChain.

## Features
* **100% Local Processing:** Customer data never leaves the host machine.
* **Live Web Search:** Uses DuckDuckGo to search `site:sophos.com` for up-to-date product context and links.
* **Few-Shot Tone Matching:** Bypasses standard AI tropes by mimicking user-provided seed emails.
* **Automated Packaging:** Outputs ready-to-send `.eml` files with physical attachments.

## Prerequisites
* macOS (M-series Apple Silicon recommended)
* [Docker Desktop](https://www.docker.com/products/docker-desktop/)
* [Ollama](https://ollama.com/) (Running natively on macOS)

## Setup Instructions

1. **Start the local LLM:**
   Ensure Ollama is running natively on your Mac and pull the Llama 3.1 model. In your terminal run:
   
   `ollama run llama3.1`

2. **Clone this repository:**
   
   `git clone <your-repo-url>`
   `cd sophos-rag-app`

3. **Configure the application:**
   * Add your standard PDF attachments to the `/attachments` folder.
   * Edit `app.py` to insert your own "Few-Shot" email examples in the `EMAIL_PROMPT`.

4. **Build and start the Docker container:**
   
   `docker compose up --build`

## Usage
Simply drop a `.txt` file containing meeting notes or a transcript into the `transcripts/` folder. The application will detect the file, extract key topics, search the live web for context, generate an email, and drop a ready-to-open `.eml` draft into the `outputs/` folder.