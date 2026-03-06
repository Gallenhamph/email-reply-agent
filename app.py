import os
import time
import mimetypes
import logging
import re
import chromadb
import markdown
import pymupdf4llm
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

# Initialize Models and Tools
llm = OllamaLLM(model="llama3.1", base_url=OLLAMA_URL)
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_URL)
search_tool = DuckDuckGoSearchResults()

# Initialize ChromaDB Client
chroma_client = chromadb.HttpClient(host="chromadb", port=8000)

# ==========================================
# PROMPT TEMPLATES
# ==========================================
EXTRACTION_PROMPT = PromptTemplate.from_template("""
You are a helpful assistant. Read the following meeting notes and identify the primary cybersecurity products, tools, or services discussed. 
Return ONLY a comma-separated list of the product names. Do not write a sentence. If none are found, return "Sophos and Secureworks".

TRANSCRIPT:
{transcript}
""")

EMAIL_PROMPT = PromptTemplate.from_template("""
You are an expert Sales Engineer representing Sophos and Secureworks. Your task is to write a short, punchy follow-up email based on the meeting transcript.

STRICT RULES:
1. PERSPECTIVE & PRONOUNS (CRITICAL): You are writing directly TO the customer. You MUST translate third-person meeting notes into direct address (e.g., "you asked"). Use "we/our" when referring to Sophos/Secureworks capabilities.
2. BRIEF RECAP: Begin the email with a highly concise (1 to 2 sentences max) summary of the salient business problems or goals discussed.
3. ACTION ITEMS: After the brief recap, focus strictly on deliverables, open questions, and next steps. 
4. Do NOT use words like: delve, robust, tailored, seamless, testament, crucial, or "I hope this email finds you well."
5. Use the LIVE WEB DATA to include accurate public context and hyperlink references using Markdown.
6. Mention that you have attached any relevant documents identified in the LOCAL PDF KNOWLEDGE.
7. You MUST adopt the tone and structure of the EXAMPLE EMAILS. Do NOT copy their exact content.
8. OUTPUT FORMAT: You must output ONLY the raw email text. Start directly with the greeting.

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
    """Reads PDFs, converts to Markdown, and loads into ChromaDB."""
    logger.info("Checking for new PDFs in /attachments to index...")
    
    # Reset collection to prevent duplicate chunks on restart
    try:
        chroma_client.delete_collection("sophos_docs")
        logger.info("Cleared old database collection for fresh ingestion.")
    except Exception:
        pass # Collection didn't exist yet

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
    
    vector_store.add_documents(splits)
    logger.info(f"Successfully indexed {len(splits)} Markdown chunks into ChromaDB!")

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

    base_name = os.path.basename(source_path).replace('.txt', '')
    output_file = os.path.join(OUTPUTS_DIR, f"Draft_{base_name}.eml")
    
    with open(output_file, 'wb') as f:
        f.write(bytes(msg))
    
    logger.info(f"Success! Ready to send: {output_file}")

# ==========================================
# EVENT HANDLER
# ==========================================
class AttachmentHandler(FileSystemEventHandler):
    def on_created(self, event):
        # Ignore directories and non-PDF files
        if event.is_directory or not event.src_path.lower().endswith('.pdf'):
            return
        
        logger.info(f"New PDF detected: {event.src_path}")
        time.sleep(2) # Give macOS a second to finish copying the large file
        
        # Re-run the ingestion function to update ChromaDB with the new file
        logger.info("Rebuilding vector database with new attachments...")
        ingest_pdfs_on_startup()
        
class TranscriptHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith('.txt'):
            return
        
        logger.info(f"New transcript detected: {event.src_path}")
        time.sleep(1) # Ensure file system lock releases
        
        with open(event.src_path, 'r', encoding='utf-8') as f:
            transcript = f.read()

        try:
            # 1. Topic Extraction
            logger.info("Extracting key topics...")
            topics = (EXTRACTION_PROMPT | llm).invoke({"transcript": transcript}).strip()
            
            # 2. Live Web Search
            logger.info(f"Searching web for topics: {topics}")
            web_data = execute_web_search(topics)

            # 3. Vector Database Retrieval
            logger.info("Querying ChromaDB for local PDF context...")
            vector_store = get_vector_store()
            pdf_results = vector_store.similarity_search(topics, k=3)
            
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
            
            # 5. Clean AI Preambles (The Guillotine)
            logger.info("Applying formatting cleanup...")
            match = re.search(r'^(Hi\s|Hello\s|Dear\s|Hey\s|Good\s)', email_body, re.MULTILINE | re.IGNORECASE)
            if match:
                email_body = email_body[match.start():]

            # 6. Package to Outlook
            logger.info("Packaging .eml file...")
            package_eml_file(email_body, list(files_to_attach), event.src_path)

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