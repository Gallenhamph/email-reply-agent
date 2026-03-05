import os
import time
import mimetypes
import chromadb
import markdown # Added for HTML email rendering
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
You are an expert Sales Engineer representing Sophos and Secureworks. Write a follow-up email to a customer based on the meeting transcript.

STRICT RULES:
1. Do NOT use words like: delve, robust, tailored, seamless, testament, crucial, or "I hope this email finds you well."
2. Match the exact tone and structure of the "EXAMPLE EMAILS" provided below.
3. Keep it concise. Address specific concerns from the meeting.
4. Use the "LIVE WEB DATA" to include accurate public context. You MUST hyperlink references back to the source URLs from sophos.com and secureworks.com as evidence using Markdown (e.g., [read more about Taegis here](https://www.secureworks.com/...)).
5. Explicitly mention that you have attached the relevant documents identified in the LOCAL PDF KNOWLEDGE.

EXAMPLE EMAILS (Mimic this exact writing style):
---
Example 1:
HHi <customer>,

Thank you for your time on the call today.  I’m sorry that it was under these difficult circumstances; however I appreciate you talking through the incident with me and discussing how Sophos may be able to assist moving forwards.

To summarise what we discussed on the call:

Your network is primarily ESET currently; however you were trialling Sophos on a small subset of devices (~4 user devices, ~8 servers)

On the 16th September you suffered a Cryptolocker ransomware attack

This caused significant damage to your estate – including encrypted virtual machines and damaged backups

You did not see any Sophos detections for this attack

You identified a locker.exe file that you believe was undetected by Sophos – however ESET detected it as malicious.  By your own admission you checked this file on VirusTotal and only 2-3 security vendors identified this file as malicious

I completely empathise at the difficult position this has put you in.  With the loss of servers comes a large chunk of downtime which is not only costly to the business to repair but also has the knock on effect of causing a loss in earnings and therefore more financial difficulty.  In the current global climate this is only more compounded.

It is worth understanding that Ransomware is not something that is distributed by amateur hackers.  Almost all attacks that we deal with in the present day are directed and driven by organised cybercriminal gangs who benefit from significant funding and often government approval.  These gangs are staffed by expert hackers, software engineers and malware authors and therefore can easily and swiftly craft a directed attack against your network.  As I mentioned on the call; due to this if you give one of these gangs enough time on your network they’re going to be able to perform any damaging action that they wish; in this case ransomware.

Unfortunately often in these cases the attacks occur due to one of the below reasons:

Not all devices are protected by a next-gen antivirus solution

We know that only around 12 devices were protected by Sophos.  I know that you are using ESET on the other devices but unfortunately we do not have visibility into that estate to fully audit whether there were/are any unprotected machines

Unprotected machines allow attackers to stage and prepare for an attack without being noticed by an administrator or security product

Security policies were not correctly defined

We know that for the devices protected by Sophos the policies were set to best practice

However we have no visibility into ESET to be confident that the same is true there

There are unpatched vulnerabilities in your network

We would need a full investigation to understand if this is true

A user or administrator has had their credentials stolen or phished and therefore has made it incredibly easy for an attacker to access the network

Upon reviewing the detections that we did have in Sophos Central we did see a large amount of unexpected behaviour around using the NirSoft suite of products and PSExec.  Both of these are administration tools.  The NirSoft suite allows administrators to perform a mhuge number of tasks – in this case we could see attempts to steal passwords from:

LSASS

Chrome

SAM

Opera

Kerberos

We could also see PSExec detections.  PsExec is an administrative tool created by System Internals, now part of Microsoft. The tool can be used by administrators to execute processes locally or remotely, as administrator or as a system account. It is often used for installing applications or running scripts across multiple devices. In the wrong hands these features can be used to deploy malware rapidly across an environment. 

You need to investigate if these detections were related to legitimate administration tasks.  If not it would indicate that these were actions being undertaken by the attacker at the beginning of September.  This would of course indicate that the attacker has been in your network nearly a month.  This is a huge amount of time to allow them to understand the flaws in your security and how they can cause the most damage (for example encrypting your virtual machines and destroying your backups).

From my perspective the priority now is to ensure that this does not happen again.  This needs a full and thorough investigation to understand:

Whether the attacker is still on your network – and to remove them

Identify exactly the path the attacker has taken through your network

The tools, techniques and processes undertaken by the attacker – and remove them so that the attacker cannot just return and cause more damage

What the attacker has done in relation to stealing data or damaging systems – as you may need to report this to government or compliance bodies

Ensure that during the time of the investigation the attacker does not return and cause additional damage due to non-payment of ransom

These are all actions undertaken by the Sophos Rapid Response service.  The Sophos Rapid Response service provides 45 days of full incident response for a fixed price.  As part of the service we would provide:

24x7 monitoring of every device on your network

Full threat detection, identification, containment and neutralisation across every device on your network

Assistance in confirming whether the attacker has installed additional persistence mechanisms

Investigation into the root cause of the issue and a full threat summary outlining our findings and recommended next steps (I have attached a sample copy)

 I’ve attached a datasheet for the Rapid Response service as well as a sample of the reports that we create for our Rapid Response customers

To proceed further we would need to host another call with the member of your management team who would be approving this service; we can then discuss terms and conditions and any further questions you may have.

If you need anything in the interim please do not hesitate to let me know.

Kind regards,

Example 2:
Hi Luke,

Thanks for your time on the call today.  Appreciate you speaking with us about your ongoing Web Application testing requirements.  I also want to (potentially re)introduce @Edward Winfield who is your account manager at Sophos and will be able to assist even further with any specific commercial queries or questions.

As discussed there are a few ways we can test and review these applications to ensure that the stringent security policies that you have in place can be maintained.

In summary the options are:

Web application security assessment - full interactive end-to-end security test of a built web application to identify common vulnerabilities and weaknesses as well as misconfiguration errors that could lead to further security implications
https://docs.taegis.secureworks.com/services/secureworks-services-taegismxdr/web-application-security-assessment/ 
Suitable for applications with a web front end that users interact with directly
Web Service / API test - end-to-end testing for applications that interact with the suer or other services purely via API
https://docs.taegis.secureworks.com/services/secureworks-services-taegismxdr/web-service-test/ 
Suitable for back and or intermediary systems that connect or communicate between other front ends but require testing in a vacuum
Secure Code Analysis - Static code analysis in a variety of different coding languages to unearth potential flaws, vulnerabilities and risks before commits are made to develop or publish th resulting application
https://docs.taegis.secureworks.com/services/secureworks-services-taegismxdr/secure-code-analysis/ 
Suitable for applications that are still in the development stage

To be able to provide accurate scoping and costings on these projects it would be useful to have some additional information - we can then provide you with commercials:

Web Application Security Assessment
What platform the Web Application is based off, or if it is fully custom built
Number of API calls that the web application interacts with
We can provide more detailed scoping questions if this is not a question with a logical answer due to the scope or style of the web app
Web Service / API test
API Type - SOAP/REST
Sum total of API methods that the service interacts with
Secure Code Analysis
Number of lines of active code (not including comments or blank lines)

I hope that helps.  Please do let us know if you have any questions or require any further information.

I look forward to hearing from you in due course.

Kind regards,

Example 3:
Good afternoon all,

Thanks for your time on the call today to discuss the Sophos Incident Management Retainer (IMR) and how it could be utilised to provide both reactive and proactive support to SKAO against your cybersecurity goals.

As a refresher please see below for an overview of the IMR coverage and options:

image.png

image.png

As promised I’ve included some additional collateral as discussed.  Please find included:

IMR Datasheet (attached)
IMR Service Catalogue - https://docs.taegis.secureworks.com/services/secureworks-services-taegismxdr/internal-penetration-test/
Sophos MDR Data Privacy Datasheet (includes a line for “Rapid Response” - which references our Incident Response services) - https://www.sophos.com/en-us/legal/sophos-managed-detection-and-response
Data Processing Sub Processors page (look for line items containing MDR or consulting services) - https://www.sophos.com/en-us/legal/sub-processor

Unit costs:

Generally when looking through the service catalogue the services are broken down into small, medium and large.  The general SU cost is:

Small - 8 SU
Medium - 16 SU
Large - 32 SU

Items that have a more freeform or custom scope (such as playbook analysis or IR Plan review) can differ depending on the requirements.  For example IR plan review:

image.png

And IR Playbook development:

image.png

These should be called out where they differ from the standard model.

Chain of Custody

I’ve been discussing this with one of our IR Directors and post Secureworks acquisition we can now support chain of custody where required

Emergency Incident Response Hourly Cost

As requested I’ve run a calculation on our hourly rate model to give an indicative cost for a one off ransomware incident. My estimation of hourly rate was indeed incorrect - I’ve included the correct numbers in my calculations below:

Ransomware incident = 120 hours x £404/hour = £48,480.00 ex VAT per incident MSRP

As mentioned, only hours consumed accrue costs, so this would be the final cost should all 120 hours be used.

-------------------------------------------------

If there are any further questions or clarifications that you may have, please don’t hesitate to let me know.

Mia and Jake will be in contact in due course with indicative costs for the Essentials and Essentials Plus retainers as well as the costs of additional service units per tiers.

Kind regards,

Example 4:
Hi James,

Great to meet and speak with you today at the MSP Community Event.  Thanks for taking the time to come down and I hope the day was useful to cover off our recent MSP updates as well as how Sophos MDR and our Advisory Services can be wrapped by yourselves to add to your security consultancy offerings.

Off the back of the discussions we had in the break there were a couple of items that you asked for some further details on.

MDR Warranty vs Insurance

Along with MDR Complete we provide an up to $1,000,000 breach protection warranty for cases where a customer suffers unrecoverable data loss due to a fault or failure of Sophos.  As you quite rightly pointed out, this is not a replacement or a competitor for sufficient Cyber Insurance but in fact can be a complementary offering.  Customers should continue to evaluate their cyber insurance needs in the same way as before.

A useful analogy is a car warranty vs. car insurance. They are separate and complementary. The warranty covers you if something goes wrong with the car while the insurance covers you if someone hits your vehicle.

Sophos will reimburse a broad range of incurred expenses including legal consultation fees, notification of impacted individuals, ransom payments, and regulatory penalties. Payment will only be made for expenses incurred with vendors authorized and approved by Sophos. For example, we will not cover expenses incurred with a direct competitor under the warranty.

Note, the warranty is not compensation for the incident, rather it is financial support to cover expenses incurred as a result of the incident.

Cybersecurity Tabletops

Post acquisition of Secureworks we can now offer a number of readiness or proactive security consulting services - including Cybersecurity Tabletops:

image.png

Our Cybersecurity tabletops are customised half day sessions where we will sit down with the customer’s team and run through a simulated cybersecurity scenario.  The goal of these sessions are to fire-drill a customers response to an incident, utilising an IR plan if they have one to help explore the completeness of their ability to respond, their knowledge of their own internal processes and to insure that when the worst occurs and they need to action their response processes in a real would situation they have trained the required muscle memory to deliver the response that they need.

I’ve included some further detail below on the deliverables for the service:

https://docs.taegis.secureworks.com/services/secureworks-services-taegismxdr/tabletop-exercise/

Thanks again for attending today.  If there’s anything further that you want to discuss please don’t hesitate to reach out.

Kind regards,


---

LIVE WEB DATA (Sophos & Secureworks):
{web_data}

LOCAL PDF KNOWLEDGE:
{pdf_data}

MEETING TRANSCRIPT / ACTION ITEMS:
{transcript}

Draft the email below using Markdown for hyperlinks:
""")

def ingest_pdfs_on_startup():
    print("\n[*] Checking for new PDFs in /attachments to index...")
    loader = PyPDFDirectoryLoader(ATTACHMENTS_DIR)
    documents = loader.load()
    
    if not documents:
        print("[*] No PDFs found. Skipping ingestion.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    vector_store.add_documents(splits)
    print(f"[*] Successfully indexed {len(documents)} PDF chunks into ChromaDB!")

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
        email_chain = EMAIL_PROMPT | llm
        email_body = email_chain.invoke({
            "web_data": web_results,
            "pdf_data": pdf_context,
            "transcript": transcript
        })
        
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