# Watchtower CI/CD Test Successful!
import os
import time
import mimetypes
import logging
import re
import chromadb
import markdown
import pymupdf4llm
import yaml
from email.message import EmailMessage

from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler

# LangChain Imports
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever


# ==========================================
# CONFIGURATION & SETUP
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
BASE_DIR = "/app"
TRANSCRIPTS_DIR = os.path.join(BASE_DIR, "transcripts")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
ATTACHMENTS_DIR = os.path.join(BASE_DIR, "attachments")
SEEDS_DIR = os.path.join(BASE_DIR, "seeds")
GLOSSARY_FILE = os.path.join(BASE_DIR, "glossary.yml")

# Initialize Models and Tools
llm = OllamaLLM(model="llama3.1", base_url=OLLAMA_URL)
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_URL)
search_tool = DuckDuckGoSearchResults()

# Initialize ChromaDB Client
chroma_client = chromadb.HttpClient(host="chromadb", port=8000)

# Global variable to hold the BM25 Keyword Index
global_bm25_retriever = None

# ==========================================
# PROMPT TEMPLATES
# ==========================================

EXTRACTION_PROMPT = PromptTemplate.from_template("""
You are an expert cybersecurity architect. Read the following raw voice-to-text meeting transcript and identify the primary cybersecurity products, tools, or services discussed. 

CRITICAL RULE: The transcript contains phonetic spelling errors, misheard words, and broken acronyms (e.g., "Sofos" instead of "Sophos", "manage thread response" instead of "MDR", "secure works" instead of "Secureworks"). You MUST correct these to their proper industry names.

Return ONLY a comma-separated list of the properly spelled product names. Do not write a sentence. If none are found, return "Sophos and Secureworks".

TRANSCRIPT:
{transcript}
""")

EMAIL_PROMPT = PromptTemplate.from_template("""
You are an expert Sales Engineer representing Sophos and Secureworks. Your task is to write a short, punchy follow-up email based on the meeting transcript.

STRICT RULES:
1. PERSPECTIVE & PRONOUNS (CRITICAL): You are writing directly TO the customer. You MUST translate third-person meeting notes into direct address (e.g., "you asked"). Use "we/our" when referring to Sophos/Secureworks capabilities.
2. TRANSCRIPTION CORRECTION (CRITICAL): The meeting transcript contains phonetic errors. Use your expert cybersecurity domain knowledge to automatically correct terms (e.g., "Sofos" -> "Sophos").
3. BRIEF RECAP: Begin the email with a highly concise summary of the salient business problems discussed.
4. ACTION ITEMS: Focus strictly on deliverables, open questions, and next steps. 
5. Do NOT use words like: delve, robust, tailored, seamless, testament, crucial.
6. Use the LIVE WEB DATA to include accurate public context and hyperlink references.
7. RELEVANCE CHECK (CRITICAL): Evaluate the LOCAL PDF KNOWLEDGE. If a document is highly relevant to the customer's specific needs, mention you have attached it. If the documents provided are NOT relevant to the conversation, ignore them completely and do NOT mention any attachments.
8. You MUST adopt the tone and structure of the EXAMPLE EMAILS. Do NOT copy their exact content.
9. OUTPUT FORMAT: First, output the raw email text starting directly with the greeting. Then, skip a line after your signature and provide a strict list of the exact filenames you decided to attach, formatted like this:
ATTACHMENTS: file1.pdf, file2.pdf
If no documents were relevant, output:
ATTACHMENTS: NONE

<EXAMPLE_EMAILS>
{seed_emails}
</EXAMPLE_EMAILS>

<LIVE_WEB_DATA>
{web_data}
</LIVE_WEB_DATA>

<LOCAL_PDF_KNOWLEDGE>
{pdf_data}
</LOCAL_PDF_KNOWLEDGE>

<MEETING_TRANSCRIPT>
{transcript}
</MEETING_TRANSCRIPT>

Output the email draft immediately below this line, starting with the greeting:
""")

# ==========================================
# CORE FUNCTIONS
# ==========================================
def setup_directories():
    """Ensure all required directories exist."""
    for d in [TRANSCRIPTS_DIR, OUTPUTS_DIR, ATTACHMENTS_DIR, SEEDS_DIR]:
        os.makedirs(d, exist_ok=True)

def load_seed_emails() -> str:
    """Dynamically load few-shot examples from the seeds directory."""
    seeds_text = ""
    for filename in os.listdir(SEEDS_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(SEEDS_DIR, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                seeds_text += f"--- Example: {filename} ---\n{f.read().strip()}\n\n"
                
    return seeds_text or "No seed examples provided. Please use a professional, concise Sales Engineer tone."

def get_vector_store() -> Chroma:
    """Retrieves or creates the Chroma vector store collection."""
    return Chroma(client=chroma_client, collection_name="sophos_docs", embedding_function=embeddings)

def ingest_pdfs_on_startup():
    global global_bm25_retriever  # <--- Allow modifying the global variable
    
    logger.info("Checking for new PDFs in /attachments to index...")
    
    try:
        chroma_client.delete_collection("sophos_docs")
        logger.info("Cleared old database collection for fresh ingestion.")
    except Exception:
        pass 

    vector_store = get_vector_store()
    documents = []
    
    for filename in os.listdir(ATTACHMENTS_DIR):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(ATTACHMENTS_DIR, filename)
            logger.info(f"Parsing {filename} into Markdown...")
            try:
                md_text = pymupdf4llm.to_markdown(file_path)
                doc = Document(page_content=md_text, metadata={"source": file_path})
                documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to parse {filename}: {e}")

    if not documents:
        logger.info("No PDFs found or parsed. Skipping database insertion.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    splits = text_splitter.split_documents(documents)
    
    # 1. Build the Dense Vector Database (Chroma)
    vector_store.add_documents(splits)
    logger.info(f"Successfully indexed {len(splits)} Markdown chunks into ChromaDB!")
    
    # 2. Build the Sparse Keyword Database (BM25)
    logger.info("Building BM25 Keyword Index for Hybrid Search...")
    global_bm25_retriever = BM25Retriever.from_documents(splits)
    global_bm25_retriever.k = 3  # Tell BM25 to return its top 3 keyword matches

def execute_web_search(topics: str) -> str:
    """Queries DuckDuckGo for live data across specified domains."""
    web_results = ""
    for domain in ["sophos.com", "secureworks.com"]:
        try:
            res = search_tool.run(f"site:{domain} {topics}")
            web_results += f"{domain.upper()} RESULTS:\n{res}\n\n"
        except Exception as e:
            logger.warning(f"Search failed for {domain}: {e}")
            
    return web_results.strip() or "No live web data available at this time."

def package_eml_file(body: str, attachments: list, source_path: str):
    """Wraps the generated Markdown text into a rich HTML .eml file."""
    msg = EmailMessage()
    msg['Subject'] = "Follow-up regarding our recent discussion"
    msg['From'] = "youremail@yourcompany.com" 
    msg['To'] = "customer@example.com"
    msg['X-Unsent'] = '1' # Forces Outlook to open as a draft
    
    msg.set_content(body)
    
    html_body = markdown.markdown(body)
    full_html = f"<html><body>{html_body}</body></html>"
    msg.add_alternative(full_html, subtype='html')

    # --- ATTACHMENT LOOP ---
    for filename in attachments:
        filepath = os.path.join(ATTACHMENTS_DIR, filename)
        if os.path.exists(filepath):
            ctype, encoding = mimetypes.guess_type(filepath)
            if ctype is None or encoding is not None:
                ctype = 'application/octet-stream'
            maintype, subtype = ctype.split('/', 1)
            
            with open(filepath, 'rb') as f:
                msg.add_attachment(f.read(), maintype=maintype, subtype=subtype, filename=filename)
                logger.info(f"Attached document: {filename}")

    # --- FILE SAVING BLOCK ---
    base_name = os.path.basename(source_path).replace('.txt', '')
    output_file = os.path.join(OUTPUTS_DIR, f"Draft_{base_name}.eml")
    
    with open(output_file, 'wb') as f:
        f.write(bytes(msg))
    
    logger.info(f"Success! Ready to send: {output_file}")


def clean_transcript_with_glossary(text: str) -> str:
    if not os.path.exists(GLOSSARY_FILE):
        return text
        
    try:
        with open(GLOSSARY_FILE, 'r', encoding='utf-8') as f:
            glossary = yaml.safe_load(f)
            
        if glossary:
            # Sort keys by length descending to replace longer phrases first
            for key in sorted(glossary.keys(), key=len, reverse=True):
                # Regex \b ensures we only replace whole words, not parts of words
                pattern = re.compile(r'\b' + re.escape(key) + r'\b', re.IGNORECASE)
                text = pattern.sub(glossary[key], text)
                
    except Exception as e:
        logger.error(f"Failed to apply glossary: {e}")
        
    return text

# ==========================================
# EVENT HANDLER
# ==========================================

# Global state for debouncing duplicate macOS file events
processed_files = {}
DEBOUNCE_SECONDS = 15

class AttachmentHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory or not event.src_path.lower().endswith('.pdf'):
            return
            
        # --- DEBOUNCE LOGIC ---
        current_time = time.time()
        if current_time - processed_files.get(event.src_path, 0) < DEBOUNCE_SECONDS:
            return # Ignore duplicate macOS events
        processed_files[event.src_path] = current_time
        
        logger.info(f"New PDF detected: {event.src_path}")
        time.sleep(2) # Give the OS time to finish writing the file bytes
        
        logger.info("Rebuilding vector database with new attachments...")
        ingest_pdfs_on_startup()


class TranscriptHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith('.txt'):
            return
            
        # --- DEBOUNCE LOGIC ---
        current_time = time.time()
        if current_time - processed_files.get(event.src_path, 0) < DEBOUNCE_SECONDS:
            return # Ignore duplicate macOS events
        processed_files[event.src_path] = current_time
        
        logger.info(f"New transcript detected: {event.src_path}")
        time.sleep(1) # Ensure file system lock releases
        
        with open(event.src_path, 'r', encoding='utf-8') as f:
            raw_transcript = f.read()
            
        logger.info("Applying custom glossary corrections to transcript...")
        transcript = clean_transcript_with_glossary(raw_transcript)

        try:
            # 1. Topic Extraction
            logger.info("Extracting key topics...")
            topics = (EXTRACTION_PROMPT | llm).invoke({"transcript": transcript}).strip()
            
            # 2. Live Web Search
            logger.info(f"Searching web for topics: {topics}")
            web_data = execute_web_search(topics)

            # 3. Hybrid Search Retrieval (Vector + Keyword)
            logger.info("Running Native Hybrid Search (ChromaDB + BM25)...")
            vector_store = get_vector_store()
            chroma_retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            
            if global_bm25_retriever:
                # A. Query both engines independently
                bm25_docs = global_bm25_retriever.invoke(topics)
                chroma_docs = chroma_retriever.invoke(topics)
                
                # B. Native Reciprocal Rank Fusion (RRF) Algorithm
                fused_scores = {}
                doc_map = {}
                
                for docs in [bm25_docs, chroma_docs]:
                    for rank, doc in enumerate(docs):
                        content = doc.page_content
                        if content not in fused_scores:
                            fused_scores[content] = 0
                            doc_map[content] = doc
                        fused_scores[content] += 1 / (rank + 60)
                        
                # C. Sort by highest fused score and grab the top 3
                sorted_items = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
                pdf_results = [doc_map[content] for content, score in sorted_items][:3]
                
            else:
                pdf_results = chroma_retriever.invoke(topics)
            
            pdf_context = ""
            files_to_attach = set() 
            for doc in pdf_results:
                pdf_context += f"Source File: {doc.metadata['source']}\nContent: {doc.page_content}\n\n"
                files_to_attach.add(os.path.basename(doc.metadata['source']))
                
            # 4. Email Generation
            logger.info("Generating final email draft...")
            email_body = (EMAIL_PROMPT | llm).invoke({
                "seed_emails": load_seed_emails(),
                "web_data": web_data,
                "pdf_data": pdf_context,
                "transcript": transcript
            })
            
            # 5. Clean AI Preambles and Extract Attachments
            logger.info("Applying formatting cleanup and extracting attachment decisions...")
            
            final_attachments = []
            attach_match = re.search(r'ATTACHMENTS:\s*(.+)', email_body, re.IGNORECASE)
            
            if attach_match:
                attach_str = attach_match.group(1).strip()
                if "NONE" not in attach_str.upper():
                    for file in files_to_attach:
                        if file in attach_str:
                            final_attachments.append(file)
                email_body = email_body[:attach_match.start()].strip()

            match = re.search(r'^(Hi\s|Hello\s|Dear\s|Hey\s|Good\s)', email_body, re.MULTILINE | re.IGNORECASE)
            if match:
                email_body = email_body[match.start():]

            # 6. Package to Outlook
            logger.info(f"Packaging .eml file with {len(final_attachments)} approved attachments...")
            package_eml_file(email_body, final_attachments, event.src_path)

        except Exception as e:
            logger.error(f"Failed to process transcript: {e}", exc_info=True)

if __name__ == "__main__":
    setup_directories()
    ingest_pdfs_on_startup()

    # 1. Start watching Transcripts
    transcript_handler = TranscriptHandler()
    transcript_observer = PollingObserver()
    transcript_observer.schedule(transcript_handler, path=TRANSCRIPTS_DIR, recursive=False)
    
    # 2. Start watching Attachments
    attachment_handler = AttachmentHandler()
    attachment_observer = PollingObserver()
    attachment_observer.schedule(attachment_handler, path=ATTACHMENTS_DIR, recursive=False)
    
    logger.info(f"Watching for transcripts in {TRANSCRIPTS_DIR}...")
    logger.info(f"Watching for new PDFs in {ATTACHMENTS_DIR}...")
    
    transcript_observer.start()
    attachment_observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        transcript_observer.stop()
        attachment_observer.stop()
        
    transcript_observer.join()
    attachment_observer.join()