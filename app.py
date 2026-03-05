import os
import time
import mimetypes
import re
import chromadb
import markdown # Added for HTML email rendering
import pymupdf4llm
from langchain_core.documents import Document
from email.message import EmailMessage
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler

# LangChain Imports
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
TRANSCRIPTS_DIR = "/app/transcripts"
OUTPUTS_DIR = "/app/outputs"
ATTACHMENTS_DIR = "/app/attachments"
SEEDS_DIR = "/app/seeds" # <--- Add this line

# Initialize Models and Tools
llm = OllamaLLM(model="llama3.1", base_url=OLLAMA_URL)
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_URL)
search_tool = DuckDuckGoSearchResults()

# Initialize ChromaDB Client
chroma_client = chromadb.HttpClient(host="chromadb", port=8000)
vector_store = Chroma(client=chroma_client, collection_name="sophos_docs", embedding_function=embeddings)

# --- PROMPT TEMPLATES ---
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

def ingest_pdfs_on_startup():
    print("\n[*] Checking for new PDFs in /attachments to index using PyMuPDF Markdown Extraction...")
    
    documents = []
    
    # Iterate through all PDFs in the folder
    for filename in os.listdir(ATTACHMENTS_DIR):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(ATTACHMENTS_DIR, filename)
            print(f"[*] Parsing {filename} into Markdown...")
            
            try:
                # Extract the PDF as formatted Markdown (preserves tables and structure)
                md_text = pymupdf4llm.to_markdown(file_path)
                
                # Create a LangChain Document object
                doc = Document(page_content=md_text, metadata={"source": file_path})
                documents.append(doc)
            except Exception as e:
                print(f"[!] Failed to parse {filename}: {e}")

    if not documents:
        print("[*] No PDFs found or parsed. Skipping database insertion.")
        return

    # Split the Markdown documents into manageable chunks
    print("[*] Splitting Markdown into chunks for ChromaDB...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    splits = text_splitter.split_documents(documents)
    
    # Optional: Clear the database before adding to prevent duplicates on restart
    # (Since we are loading everything at startup for this local app)
    vector_store.add_documents(splits)
    print(f"[*] Successfully indexed {len(splits)} Markdown chunks into ChromaDB!")

    def load_seed_emails():
    seeds_text = ""
    if not os.path.exists(SEEDS_DIR):
        return seeds_text
        
    for filename in os.listdir(SEEDS_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(SEEDS_DIR, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                seeds_text += f"--- Example: {filename} ---\n{f.read().strip()}\n\n"
                
    if not seeds_text:
        seeds_text = "No seed examples provided. Please use a professional, concise Sales Engineer tone."
        
    return seeds_text

class TranscriptHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith('.txt'):
            return
        
        print(f"\n[+] New transcript detected: {event.src_path}")
        time.sleep(1) 
        
        with open(event.src_path, 'r', encoding='utf-8') as f:
            transcript_text = f.read()

        email_body, dynamic_attachments = self.process_and_generate(transcript_text)
        self.create_eml_file(email_body, dynamic_attachments, event.src_path)

    def process_and_generate(self, transcript):
        print("[-] Step 1: Extracting key topics...")
        extraction_chain = EXTRACTION_PROMPT | llm
        topics = extraction_chain.invoke({"transcript": transcript}).strip()
        print(f"[-] Topics found: {topics}")

        print("[-] Step 2: Searching site:sophos.com AND site:secureworks.com...")
        web_results = ""
        try:
            sophos_res = search_tool.run(f"site:sophos.com {topics}")
            web_results += f"SOPHOS RESULTS:\n{sophos_res}\n\n"
        except Exception:
            pass
            
        try:
            secureworks_res = search_tool.run(f"site:secureworks.com {topics}")
            web_results += f"SECUREWORKS RESULTS:\n{secureworks_res}\n\n"
        except Exception:
            pass

        if not web_results.strip():
            web_results = "No live web data available at this time."

        print("[-] Step 3: Querying ChromaDB for local PDF context...")
        pdf_results = vector_store.similarity_search(topics, k=3)
        pdf_context = ""
        files_to_attach = set() 

        for doc in pdf_results:
            pdf_context += f"Source File: {doc.metadata['source']}\nContent: {doc.page_content}\n\n"
            filename = os.path.basename(doc.metadata['source'])
            files_to_attach.add(filename)

        print(f"[-] Identified attachments: {files_to_attach}")

print("[-] Step 4: Generating final email with Llama 3.1...")
        
        # Load the latest seeds right before generating
        current_seeds = load_seed_emails() 
        
        email_chain = EMAIL_PROMPT | llm
        email_body = email_chain.invoke({
            "seed_emails": current_seeds, # <--- Inject the seeds here
            "web_data": web_results,
            "pdf_data": pdf_context,
            "transcript": transcript
        })
        
        print("[-] Step 4.5: Slicing off AI preambles...")
        # This searches for the first instance of a standard greeting at the start of a line
        # and deletes everything the AI said before it.
        match = re.search(r'^(Hi\s|Hello\s|Dear\s|Hey\s|Good\s)', email_body, re.MULTILINE | re.IGNORECASE)
        if match:
            email_body = email_body[match.start():]
            
        # Optional: Clean up postambles (e.g., if it adds "Let me know if you need changes!" after your name)
        # This stops the email at the first blank line after your sign-off name.
        # (Assuming your name is at the end of the template).
        
        return email_body, list(files_to_attach)

    def create_eml_file(self, body, attachments, source_path):
        print("[-] Step 5: Packaging .eml file with HTML links...")
        msg = EmailMessage()
        msg['Subject'] = "Follow-up regarding our recent discussion"
        msg['From'] = "youremail@yourcompany.com" 
        msg['To'] = "customer@example.com"
        
        # --- THE MAGIC LINE FOR OUTLOOK DRAFTS ---
        msg['X-Unsent'] = '1'
        
        # Set the plain text fallback
        msg.set_content(body)
        
        # Convert Markdown (including links) to HTML and attach as the primary view
        html_body = markdown.markdown(body)
        # Wrap it in basic HTML tags to ensure Outlook reads it correctly
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
                    print(f"[-] Attached: {filename}")

        base_name = os.path.basename(source_path).replace('.txt', '')
        output_file = os.path.join(OUTPUTS_DIR, f"Draft_{base_name}.eml")
        
        with open(output_file, 'wb') as f:
            f.write(bytes(msg))
        
        print(f"[+] Success! Ready to send: {output_file}\n")

if __name__ == "__main__":
    for d in [TRANSCRIPTS_DIR, OUTPUTS_DIR, ATTACHMENTS_DIR]:
        os.makedirs(d, exist_ok=True)

    ingest_pdfs_on_startup()

    event_handler = TranscriptHandler()
    observer = PollingObserver()
    observer.schedule(event_handler, path=TRANSCRIPTS_DIR, recursive=False)
    
    print(f"Watching for transcripts in {TRANSCRIPTS_DIR}...")
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()