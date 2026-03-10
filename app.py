import os
import time
import mimetypes
import logging
import re
import chromadb
import markdown
import pymupdf4llm
import yaml
import email
from email import policy
from email.message import EmailMessage


from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler

# LangChain Imports
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
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

class Config:
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
    BASE_DIR = "/app"
    TRANSCRIPTS_DIR = os.path.join(BASE_DIR, "transcripts")
    OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
    ATTACHMENTS_DIR = os.path.join(BASE_DIR, "attachments")
    SEEDS_DIR = os.path.join(BASE_DIR, "seeds")
    GLOSSARY_FILE = os.path.join(BASE_DIR, "glossary.yml")
    DEBOUNCE_SECONDS = 15
    
    # User Identity Variables
    SE_NAME = os.getenv("SE_NAME", "Sales Engineer")
    SE_COMPANY = os.getenv("SE_COMPANY", "Sophos and Secureworks")

# Initialize Models and Tools (Mistral Small Upgrade)
llm = OllamaLLM(
    model="mistral-small", 
    base_url=Config.OLLAMA_URL,
    num_ctx=32000,   # Unlocks the brain for 90-minute transcripts
    keep_alive="0"   # Instantly dumps the 14GB model from RAM when finished
)

embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=Config.OLLAMA_URL)
search_tool = DuckDuckGoSearchResults()
chroma_client = chromadb.HttpClient(host="chromadb", port=8000)

# Global variable to hold the BM25 Keyword Index
global_bm25_retriever = None

# ==========================================
# PROMPT TEMPLATES 
# ==========================================

EXTRACTION_PROMPT = PromptTemplate.from_template("""
You are an expert cybersecurity architect. Read the following raw voice-to-text meeting transcript.
Your goal is to identify the single most critical {se_company} product discussed that requires follow-up research.

CRITICAL RULE: Correct any phonetic spelling errors (e.g., "Sofos" -> "Sophos").

Return ONLY the name of the top 1 or 2 products, separated by a comma. Keep it incredibly brief (e.g., "Sophos MDR, Microsoft Intune"). Do not write a sentence.
TRANSCRIPT:
{transcript}
""")

EMAIL_PROMPT = PromptTemplate.from_template("""
You are {se_name}, an expert Cybersecurity Sales Engineer at {se_company}. Your ONLY job is to write a brand new follow-up email to a customer based strictly on the MEETING_NOTES provided below.

<CRITICAL_RULES>
1. FACTUAL ACCURACY: Base your business recap and action items strictly on the <MEETING_NOTES>. Do not invent deliverables. Specifically look for tasks assigned to {se_name} and list those as our next steps.
2. STYLE TRANSFER: Analyze the <STYLE_EXAMPLES> to understand the author's tone, sentence length, and formatting preferences. Write your brand new email using this exact persona.
3. TRANSCRIPTION CORRECTION: The notes contain phonetic errors. Automatically correct industry terms.
4. DIRECT ADDRESS: Write directly to the customer (e.g., "Great speaking with you...").
5. STRUCTURE & VERBOSITY: Do not just output a list of bullets. Start with 1 to 2 professional, conversational paragraphs summarizing the core business challenges and technical goals discussed in the meeting. Write in complete, human-sounding sentences. 
6. LOGICAL SUMMARIZATION: After the conversational recap, transition into a logically grouped, well-explained bulleted list outlining the specific action items, deliverables, and next steps.
7. PRISTINE FORMATTING: You must use perfect Markdown formatting for readability. ALWAYS leave a blank empty line before starting a bulleted list. Use standard bullets (`* ` or `- `) and ensure proper spacing.
8. ATTACHMENTS: At the very bottom of the email, list the exact filenames of any highly relevant PDFs from the <PDF_KNOWLEDGE> section in this format: ATTACHMENTS: file1.pdf, file2.pdf (If none are relevant, output ATTACHMENTS: NONE).
</CRITICAL_RULES>

<STYLE_EXAMPLES>
{seed_emails}
</STYLE_EXAMPLES>

<PDF_KNOWLEDGE>
{pdf_data}
</PDF_KNOWLEDGE>

<WEB_RESEARCH>
{web_data}
</WEB_RESEARCH>

<MEETING_NOTES>
{transcript}
</MEETING_NOTES>

TASK: Write the final email draft now. Base the content ENTIRELY on the <MEETING_NOTES> above.
""")

# ==========================================
# CORE SERVICES
# ==========================================
def setup_directories():
    for d in [Config.TRANSCRIPTS_DIR, Config.OUTPUTS_DIR, Config.ATTACHMENTS_DIR, Config.SEEDS_DIR]:
        os.makedirs(d, exist_ok=True)

def cleanup_old_transcripts(days_to_keep=7):
    """Sweeps the transcripts folder and deletes files older than the specified days."""
    now = time.time()
    cutoff_time = now - (days_to_keep * 86400) # 86400 seconds in a day
    
    if os.path.exists(Config.TRANSCRIPTS_DIR):
        for filename in os.listdir(Config.TRANSCRIPTS_DIR):
            if filename.endswith(".processed") or filename.endswith(".error"):
                filepath = os.path.join(Config.TRANSCRIPTS_DIR, filename)
                try:
                    file_mtime = os.path.getmtime(filepath)
                    if file_mtime < cutoff_time:
                        os.remove(filepath)
                        logger.info(f"Routine Cleanup: Deleted old transcript {filename}")
                except Exception as e:
                    logger.warning(f"Cleanup failed for {filename}: {e}")

def load_seed_emails() -> str:
    """Dynamically load few-shot examples from both .txt and .eml files in the seeds directory."""
    seeds_text = ""
    if os.path.exists(Config.SEEDS_DIR):
        for filename in os.listdir(Config.SEEDS_DIR):
            filepath = os.path.join(Config.SEEDS_DIR, filename)
            
            # Handle legacy .txt seeds
            if filename.lower().endswith(".txt"):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        seeds_text += f"--- Example: {filename} ---\n{f.read().strip()}\n\n"
                except Exception as e:
                    logger.warning(f"Failed to read text seed {filename}: {e}")
                    
            # Handle new .eml seeds directly from Outlook
            elif filename.lower().endswith(".eml"):
                try:
                    with open(filepath, 'rb') as f:
                        msg = email.message_from_binary_file(f, policy=policy.default)
                        
                        # Try to grab the plain text version of the email first
                        text_part = msg.get_body(preferencelist=('plain',))
                        
                        if text_part:
                            body = text_part.get_content()
                        else:
                            # Fallback: If it's HTML only, strip the tags out
                            html_part = msg.get_body(preferencelist=('html',))
                            if html_part:
                                raw_html = html_part.get_content()
                                body = re.sub(r'<[^>]+>', '', raw_html) # Basic HTML stripping
                            else:
                                body = ""
                                
                        if body.strip():
                            seeds_text += f"--- Example: {filename} ---\n{body.strip()}\n\n"
                            
                except Exception as e:
                    logger.warning(f"Failed to parse .eml seed {filename}: {e}")
                    
    return seeds_text or "No seed examples provided. Please use a professional, concise Sales Engineer tone."

def get_vector_store() -> Chroma:
    return Chroma(client=chroma_client, collection_name="sophos_docs", embedding_function=embeddings)

def ingest_pdfs_on_startup():
    global global_bm25_retriever
    logger.info("Checking for new PDFs in /attachments to index...")
    
    try:
        chroma_client.delete_collection("sophos_docs")
        logger.info("Cleared old database collection for fresh ingestion.")
    except Exception:
        pass 

    vector_store = get_vector_store()
    documents = []
    
    if os.path.exists(Config.ATTACHMENTS_DIR):
        for filename in os.listdir(Config.ATTACHMENTS_DIR):
            if filename.lower().endswith(".pdf"):
                file_path = os.path.join(Config.ATTACHMENTS_DIR, filename)
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
    
    logger.info("Building BM25 Keyword Index for Hybrid Search...")
    global_bm25_retriever = BM25Retriever.from_documents(splits)
    global_bm25_retriever.k = 8  

def execute_web_search(topics: str) -> str:
    topics_list = [t.strip() for t in topics.split(',') if t.strip()]
    if len(topics_list) == 1:
        topics_list = [topics[:60]]
        
    web_results = ""
    for topic in topics_list[:2]:
        logger.info(f"Querying web specifically for: {topic}")
        for domain in ["sophos.com", "secureworks.com"]:
            try:
                time.sleep(1.5) 
                res = search_tool.run(f"site:{domain} {topic}")
                if res and "No good DuckDuckGo Search Result" not in res:
                    web_results += f"{domain.upper()} - {topic}:\n{res}\n\n"
            except Exception as e:
                logger.warning(f"DDG Search failed for '{topic}' on {domain}: {e}")
                
    return web_results.strip() or "No live web data available at this time."

def package_eml_file(body: str, attachments: list, source_path: str):
    msg = EmailMessage()
    msg['Subject'] = "Follow-up regarding our recent discussion"
    msg['From'] = f"{Config.SE_NAME} <youremail@yourcompany.com>" 
    msg['To'] = "customer@example.com"
    msg['X-Unsent'] = '1'
    
    msg.set_content(body)
    html_body = markdown.markdown(body, extensions=['sane_lists', 'nl2br'])
    full_html = f"<html><body>{html_body}</body></html>"
    msg.add_alternative(full_html, subtype='html')

    for filename in attachments:
        filepath = os.path.join(Config.ATTACHMENTS_DIR, filename)
        if os.path.exists(filepath):
            ctype, encoding = mimetypes.guess_type(filepath)
            if ctype is None or encoding is not None:
                ctype = 'application/octet-stream'
            maintype, subtype = ctype.split('/', 1)
            
            with open(filepath, 'rb') as f:
                msg.add_attachment(f.read(), maintype=maintype, subtype=subtype, filename=filename)
                logger.info(f"Attached document: {filename}")

    base_name = os.path.basename(source_path).replace('.txt', '')
    output_file = os.path.join(Config.OUTPUTS_DIR, f"Draft_{base_name}.eml")
    
    with open(output_file, 'wb') as f:
        f.write(bytes(msg))
    logger.info(f"Success! Ready to send: {output_file}")

def clean_transcript_with_glossary(text: str) -> str:
    if not os.path.exists(Config.GLOSSARY_FILE):
        return text
    try:
        with open(Config.GLOSSARY_FILE, 'r', encoding='utf-8') as f:
            glossary = yaml.safe_load(f)
        if glossary:
            for key in sorted(glossary.keys(), key=len, reverse=True):
                pattern = re.compile(r'\b' + re.escape(key) + r'\b', re.IGNORECASE)
                text = pattern.sub(glossary[key], text)
    except Exception as e:
        logger.error(f"Failed to apply glossary: {e}")
    return text

def process_transcript(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_transcript = f.read()
        
    logger.info("Applying custom glossary corrections to transcript...")
    transcript = clean_transcript_with_glossary(raw_transcript)

    try:
        logger.info("Extracting key topics...")
        topics = (EXTRACTION_PROMPT | llm).invoke({
            "transcript": transcript,
            "se_company": Config.SE_COMPANY
        }).strip()
        
        logger.info(f"Searching web for topics: {topics}")
        web_data = execute_web_search(topics)

        logger.info("Running Native Hybrid Search (ChromaDB + BM25)...")
        vector_store = get_vector_store()
        chroma_retriever = vector_store.as_retriever(search_kwargs={"k": 8}) 
        
        if global_bm25_retriever:
            bm25_docs = global_bm25_retriever.invoke(topics)
            chroma_docs = chroma_retriever.invoke(topics)
            
            fused_scores = {}
            doc_map = {}
            for docs in [bm25_docs, chroma_docs]:
                for rank, doc in enumerate(docs):
                    content = doc.page_content
                    if content not in fused_scores:
                        fused_scores[content] = 0
                        doc_map[content] = doc
                    fused_scores[content] += 1 / (rank + 60)
                    
            sorted_items = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
            pdf_results = [doc_map[content] for content, score in sorted_items][:8] 
        else:
            pdf_results = chroma_retriever.invoke(topics)
        
        pdf_context = ""
        files_to_attach = set() 
        for doc in pdf_results:
            pdf_context += f"Source File: {doc.metadata['source']}\nContent: {doc.page_content}\n\n"
            files_to_attach.add(os.path.basename(doc.metadata['source']))
            
        logger.info(f"Generating final email draft as {Config.SE_NAME}...")
        email_body = (EMAIL_PROMPT | llm).invoke({
            "se_name": Config.SE_NAME,
            "se_company": Config.SE_COMPANY,
            "seed_emails": load_seed_emails(),
            "web_data": web_data,
            "pdf_data": pdf_context,
            "transcript": transcript
        })
        
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

        logger.info(f"Packaging .eml file with {len(final_attachments)} approved attachments...")
        package_eml_file(email_body, final_attachments, file_path)

        # --- NEW: STOP THE INFINITE LOOP ---
        processed_path = file_path + ".processed"
        os.rename(file_path, processed_path)
        logger.info(f"Success! Archived transcript to prevent reprocessing: {os.path.basename(processed_path)}")

    except Exception as e:
        logger.error(f"Failed to process transcript: {e}", exc_info=True)
        # Prevent infinite crash loops by renaming bad files
        error_path = file_path + ".error"
        if os.path.exists(file_path):
            os.rename(file_path, error_path)

# ==========================================
# EVENT HANDLERS
# ==========================================

class DebouncedEventHandler(FileSystemEventHandler):
    def __init__(self):
        self.processed_files = {}

    def is_debounced(self, path: str) -> bool:
        current_time = time.time()
        if current_time - self.processed_files.get(path, 0) < Config.DEBOUNCE_SECONDS:
            return True
        self.processed_files[path] = current_time
        return False

class AttachmentHandler(DebouncedEventHandler):
    def on_created(self, event):
        if event.is_directory or not event.src_path.lower().endswith('.pdf'):
            return
        if self.is_debounced(event.src_path):
            return
            
        logger.info(f"New PDF detected: {event.src_path}")
        time.sleep(2)
        logger.info("Rebuilding vector database with new attachments...")
        ingest_pdfs_on_startup()


class TranscriptHandler(DebouncedEventHandler):
    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith('.txt'):
            return
        if self.is_debounced(event.src_path):
            return 
            
        logger.info(f"New transcript detected: {event.src_path}")
        
        # --- NEW: Run the 7-day cleanup sweep ---
        logger.info("Running routine cleanup of old files...")
        cleanup_old_transcripts(days_to_keep=7)
        
        time.sleep(1) 
        process_transcript(event.src_path)

if __name__ == "__main__":
    setup_directories()
    ingest_pdfs_on_startup()

    transcript_handler = TranscriptHandler()
    transcript_observer = PollingObserver()
    transcript_observer.schedule(transcript_handler, path=Config.TRANSCRIPTS_DIR, recursive=False)
    
    attachment_handler = AttachmentHandler()
    attachment_observer = PollingObserver()
    attachment_observer.schedule(attachment_handler, path=Config.ATTACHMENTS_DIR, recursive=False)
    
    logger.info(f"Watching for transcripts in {Config.TRANSCRIPTS_DIR}...")
    logger.info(f"Watching for new PDFs in {Config.ATTACHMENTS_DIR}...")
    
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