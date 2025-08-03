# app.py

import json
import logging
import os
import io
import hashlib
import uuid
import jwt
from typing import List, Optional
import requests
import time

# Import FastAPI libraries
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form, Request, status, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.exceptions import RequestValidationError
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field
from datetime import datetime, timedelta

# Import LangChain and related libraries
from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document as LangchainDocument
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Import for OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()

# --- Global Configurations ---
# Configuration for the RAG system
CHROMA_HOST = os.getenv("CHROMA_HOST", "10.10.32.78")
CHROMA_PORT = os.getenv("CHROMA_PORT", "8001")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set the LLM and Embedding models based on environment variables
if OPENAI_API_KEY:
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-3.5-turbo")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
else:
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama3.1:8b")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "nomic-embed-text:latest")

# --- Azure AD (Entra ID) & JWT Configuration ---
TENANT_ID = os.getenv("TENANT_ID")
CLIENT_ID = os.getenv("CLIENT_ID")
SECRET_KEY = os.getenv("SECRET_KEY", "your-super-secret-key-for-jwt") # A local secret key is still used for local JWTs if needed
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
jwks_uri = f"https://login.microsoftonline.com/{TENANT_ID}/discovery/v2.0/keys"
jwks_cache = {}

class AzureToken(BaseModel):
    id_token: str

class User(BaseModel):
    username: str

# --- JWT Dependency and Validation ---
def get_jwks():
    """ Fetches or retrieves JWKS from cache. """
    current_time = time.time()
    # In a real app, you would check for cache expiry headers. For simplicity, we just cache for 1 hour.
    if "keys" in jwks_cache and jwks_cache.get("timestamp", 0) > current_time - 3600:
        return jwks_cache["keys"]
    
    try:
        response = requests.get(jwks_uri)
        response.raise_for_status()
        jwks = response.json()
        jwks_cache["keys"] = jwks["keys"]
        jwks_cache["timestamp"] = current_time
        return jwks["keys"]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching JWKS: {e}")
        return None

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """ Validates the JWT from Azure AD and extracts the user. """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Fetch JWKS from Azure AD to validate the token's signature
        jwks = get_jwks()
        if not jwks:
            raise credentials_exception

        header = jwt.get_unverified_header(token)
        kid = header.get('kid')
        
        # Find the correct key in the JWKS using the 'kid'
        rsa_key = {}
        for key in jwks:
            if key['kid'] == kid:
                rsa_key = {
                    "kty": key['kty'],
                    "kid": key['kid'],
                    "use": key['use'],
                    "n": key['n'],
                    "e": key['e'],
                }
                break

        if not rsa_key:
            raise credentials_exception

        # Decode and validate the token
        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=["RS256"],
            audience=CLIENT_ID,
            options={"verify_exp": True, "verify_aud": True, "verify_iss": True},
            issuer=f"https://login.microsoftonline.com/{TENANT_ID}/v2.0"
        )
        
        # Extract username from the token's payload
        username: str = payload.get("preferred_username")
        if not username:
            username = payload.get("upn")
            if not username:
                raise credentials_exception
        
        return User(username=username)
    except jwt.PyJWTError as e:
        print(f"JWT Validation failed: {e}")
        raise credentials_exception
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during token validation.")

# --- Helper Functions (unchanged) ---
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """ Creates a JWT access token. (This might not be needed for Azure SSO, but we'll keep it for local testing if necessary) """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_pdf_text(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def get_docx_text(docx_file):
    document = Document(docx_file)
    text = ""
    for paragraph in document.paragraphs:
        text += paragraph.text + "\n"
    return text

def get_txt_text(txt_file):
    return txt_file.read().decode('utf-8')

def get_md_text(md_file):
    return md_file.read().decode('utf-8')

def get_memory(request: Request):
    session_id = request.session.get("session_id", "default_session_id")
    if session_id not in app.state.memories:
        print(f"Initializing new memory for session: {session_id}")
        if llm_memory_summarizer:
            app.state.memories[session_id] = ConversationSummaryBufferMemory(
                llm=llm_memory_summarizer,
                max_token_limit=1000,
                memory_key="chat_history",
                return_messages=True
            )
        else:
            raise HTTPException(status_code=500, detail="LLM for Memory is not initialized")
    return app.state.memories[session_id]

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# --- Global Instances ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

embeddings = None
chroma_client = None
collection = None
vectorstore = None
retriever = None
llm_qa = None
llm_memory_summarizer = None
app.state.memories = {}

# --- Event Listener for FastAPI startup ---
@app.on_event("startup")
async def startup_event():
    global embeddings, chroma_client, collection, vectorstore, retriever, llm_qa, llm_memory_summarizer

    try:
        if OPENAI_API_KEY:
            embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME, openai_api_key=OPENAI_API_KEY)
            llm_qa = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.7, openai_api_key=OPENAI_API_KEY)
            llm_memory_summarizer = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.1, openai_api_key=OPENAI_API_KEY)
            print(f"Initialized models with OpenAI: {LLM_MODEL_NAME} & {EMBEDDING_MODEL_NAME}")
        else:
            embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME)
            llm_qa = Ollama(base_url="http://localhost:11434", model=LLM_MODEL_NAME, temperature=0.7)
            llm_memory_summarizer = ChatOllama(base_url="http://localhost:11434", model=LLM_MODEL_NAME, temperature=0.1)
            print(f"Initialized models with Ollama: {LLM_MODEL_NAME} & {EMBEDDING_MODEL_NAME}")
    except Exception as e:
        print(f"FATAL ERROR: Could not initialize LLM or Embedding models. Error: {e}")
        embeddings = None
        llm_qa = None
        llm_memory_summarizer = None
    
    try:
        import chromadb
        chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=int(CHROMA_PORT))
        collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
        print(f"Connected to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}, using collection: {CHROMA_COLLECTION_NAME}")
        print(f"Current documents in collection: {collection.count()}")

        if embeddings:
            vectorstore = Chroma(
                client=chroma_client,
                collection_name=CHROMA_COLLECTION_NAME,
                embedding_function=embeddings
            )
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        else:
            print("WARNING: Embeddings model not initialized, vectorstore and retriever will not be available.")
    except Exception as e:
        print(f"FATAL ERROR: Could not connect to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}. Error: {e}")
        chroma_client = None
        collection = None
        vectorstore = None
        retriever = None

# --- API Endpoints ---

@app.get("/")
async def root():
    return {"message": "Hello, this is the RAG API with Azure SSO."}

@app.post("/token")
async def login_for_access_token(payload: AzureToken):
    """ 
    Receives an ID Token from the client and validates it with Azure AD.
    If valid, returns the same token to be used for API calls.
    """
    try:
        # Re-use the validation logic from get_current_user to verify the token
        user = await get_current_user(payload.id_token)
        return {"access_token": payload.id_token, "token_type": "bearer"}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token provided.")

# This endpoint now serves the HTML page for managing documents with Azure SSO
@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RAG Document Dashboard (Azure SSO)</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            .container {{ max-width: 800px; margin: 0 auto; }}
            .file-list {{ list-style-type: none; padding: 0; }}
            .file-item {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px;
                border: 1px solid #ddd;
                margin-bottom: 5px;
                border-radius: 5px;
            }}
            .file-name {{ font-size: 1.1em; flex-grow: 1; }}
            .delete-btn {{
                background-color: #f44336;
                color: white;
                border: none;
                padding: 8px 12px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 14px;
                margin-left: 10px;
                cursor: pointer;
                border-radius: 4px;
            }}
            .status-message {{
                margin-top: 20px;
                padding: 10px;
                border-radius: 5px;
                display: none;
            }}
            .status-message.success {{ background-color: #d4edda; color: #155724; }}
            .status-message.error {{ background-color: #f8d7da; color: #721c24; }}
            .auth-section {{
                margin-bottom: 20px;
                padding: 20px;
                border: 1px solid #ccc;
                border-radius: 8px;
            }}
            .auth-section h2 {{ margin-top: 0; }}
            .azure-login-btn {{
                width: 100%;
                padding: 10px;
                background-color: #0078D4; /* Azure Blue */
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-weight: bold;
            }}
            .logout-btn {{
                background-color: #f44336 !important;
                margin-top: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>RAG Document Dashboard (Azure SSO)</h1>
            <div class="auth-section">
                <h2 id="auth-title">เข้าสู่ระบบ</h2>
                <div id="login-form">
                    <p>เข้าสู่ระบบเพื่อจัดการเอกสารของคุณ</p>
                    <button id="azure-login-button" class="azure-login-btn">เข้าสู่ระบบด้วย Microsoft</button>
                </div>
                <div id="user-info" style="display: none;">
                    <p>เข้าสู่ระบบในฐานะ: <strong id="current-user"></strong></p>
                    <button id="logout-button" class="logout-btn">ออกจากระบบ</button>
                    <button id="fetch-files-button">โหลดเอกสาร</button>
                </div>
            </div>
            
            <div class="status-message" id="statusMessage"></div>
            <ul id="fileList" class="file-list">
                <li>กรุณาเข้าสู่ระบบเพื่อดูเอกสาร...</li>
            </ul>
        </div>

        <script src="https://alcdn.msauth.net/browser/2.16.0/js/msal-browser.js"></script>
        <script>
            const msalConfig = {{
                auth: {{
                    clientId: "{CLIENT_ID}",
                    authority: "https://login.microsoftonline.com/{TENANT_ID}"
                }},
                cache: {{
                    cacheLocation: "localStorage",
                    storeAuthStateInCookie: true
                }}
            }};

            const msalInstance = new msal.PublicClientApplication(msalConfig);
            const loginRequest = {{
                scopes: ["openid", "profile"]
            }};

            const fileListElement = document.getElementById('fileList');
            const statusMessageElement = document.getElementById('statusMessage');
            const azureLoginButton = document.getElementById('azure-login-button');
            const logoutButton = document.getElementById('logout-button');
            const fetchFilesButton = document.getElementById('fetch-files-button');
            const loginForm = document.getElementById('login-form');
            const userInfo = document.getElementById('user-info');
            const currentUserSpan = document.getElementById('current-user');
            const authTitle = document.getElementById('auth-title');
            
            let accessToken = localStorage.getItem('access_token');
            let currentUsername = localStorage.getItem('username');

            // Initial UI state
            if (accessToken && currentUsername) {{
                showLoggedInState(currentUsername);
                fetchFiles();
            }} else {{
                showLoggedOutState();
            }}

            azureLoginButton.addEventListener('click', login);
            logoutButton.addEventListener('click', logout);
            fetchFilesButton.addEventListener('click', fetchFiles);

            async function login() {{
                try {{
                    const loginResponse = await msalInstance.loginPopup(loginRequest);
                    const account = loginResponse.account;
                    const idToken = loginResponse.idToken;
                    
                    // Call backend to validate token
                    const backendResponse = await fetch('/token', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json'
                        }},
                        body: JSON.stringify({{ id_token: idToken }})
                    }});
                    
                    const backendResult = await backendResponse.json();
                    
                    if (!backendResponse.ok) {{
                        throw new Error(backendResult.detail || 'Token validation failed.');
                    }}

                    accessToken = backendResult.access_token;
                    currentUsername = account.username;
                    localStorage.setItem('access_token', accessToken);
                    localStorage.setItem('username', currentUsername);

                    showMessage('เข้าสู่ระบบสำเร็จ!', 'success');
                    showLoggedInState(currentUsername);
                    fetchFiles();

                }} catch (error) {{
                    console.error('Login error:', error);
                    showMessage('การเข้าสู่ระบบล้มเหลว: ' + error.message, 'error');
                }}
            }}

            function logout() {{
                localStorage.removeItem('access_token');
                localStorage.removeItem('username');
                accessToken = null;
                currentUsername = null;
                msalInstance.logoutRedirect({{
                    postLogoutRedirectUri: window.location.origin
                }});
                showLoggedOutState();
                fileListElement.innerHTML = '<li>กรุณาเข้าสู่ระบบเพื่อดูเอกสาร...</li>';
                showMessage('ออกจากระบบสำเร็จ', 'success');
            }}

            function showLoggedInState(username) {{
                authTitle.textContent = 'ข้อมูลผู้ใช้';
                loginForm.style.display = 'none';
                userInfo.style.display = 'block';
                currentUserSpan.textContent = username;
            }}

            function showLoggedOutState() {{
                authTitle.textContent = 'เข้าสู่ระบบ';
                loginForm.style.display = 'block';
                userInfo.style.display = 'none';
            }}

            async function fetchFiles() {{
                if (!accessToken) {{
                    showMessage('กรุณาเข้าสู่ระบบก่อน', 'error');
                    return;
                }}

                fileListElement.innerHTML = '<li>กำลังโหลดไฟล์...</li>';
                
                try {{
                    const response = await fetch('/files_list', {{
                        headers: {{
                            'Authorization': `Bearer ${{accessToken}}`
                        }}
                    }});

                    if (!response.ok) {{
                        const errorData = await response.json();
                        throw new Error(errorData.detail || `HTTP error! status: ${{response.status}}`);
                    }}
                    const files = await response.json();
                    renderFileList(files);
                }} catch (error) {{
                    console.error('Failed to fetch files:', error);
                    fileListElement.innerHTML = '<li>Error loading files.</li>';
                    showMessage(error.message, 'error');
                    if (error.message.includes('401') || error.message.includes('403')) {{
                        logout();
                    }}
                }}
            }}

            function renderFileList(files) {{
                fileListElement.innerHTML = '';
                if (files.length === 0) {{
                    fileListElement.innerHTML = '<li>ไม่พบเอกสารในระบบ</li>';
                    return;
                }}
                files.forEach(filename => {{
                    const listItem = document.createElement('li');
                    listItem.className = 'file-item';
                    listItem.innerHTML = `
                        <span class="file-name">${{filename}}</span>
                        <button class="delete-btn" data-filename="${{filename}}">Delete</button>
                    `;
                    fileListElement.appendChild(listItem);
                }});
                document.querySelectorAll('.delete-btn').forEach(button => {{
                    button.addEventListener('click', handleDelete);
                }});
            }}

            async function handleDelete(event) {{
                const filename = event.target.dataset.filename;
                
                if (!accessToken) {{
                    showMessage('กรุณาเข้าสู่ระบบก่อนลบ', 'error');
                    return;
                }}
                
                // Use custom modal instead of `confirm()`
                const userConfirmed = await new Promise(resolve => {{
                    const modal = document.createElement('div');
                    modal.style.position = 'fixed';
                    modal.style.top = '50%';
                    modal.style.left = '50%';
                    modal.style.transform = 'translate(-50%, -50%)';
                    modal.style.backgroundColor = 'white';
                    modal.style.padding = '20px';
                    modal.style.border = '1px solid #ccc';
                    modal.style.boxShadow = '0 4px 8px rgba(0,0,0,0.1)';
                    modal.style.zIndex = '1000';
                    modal.innerHTML = `
                        <p>ต้องการลบไฟล์ "${{filename}}" ใช่หรือไม่?</p>
                        <button id="confirm-yes">ใช่</button>
                        <button id="confirm-no">ไม่</button>
                    `;
                    document.body.appendChild(modal);

                    document.getElementById('confirm-yes').onclick = () => {{
                        document.body.removeChild(modal);
                        resolve(true);
                    }};
                    document.getElementById('confirm-no').onclick = () => {{
                        document.body.removeChild(modal);
                        resolve(false);
                    }};
                }});
                
                if (!userConfirmed) {{
                    return;
                }}

                try {{
                    const response = await fetch('/delete_document', {{
                        method: 'DELETE',
                        headers: {{
                            'Content-Type': 'application/json',
                            'Authorization': `Bearer ${{accessToken}}`
                        }},
                        body: JSON.stringify({{ filename: filename }})
                    }});

                    const result = await response.json();
                    
                    if (!response.ok) {{
                        throw new Error(result.detail || 'Failed to delete document.');
                    }}

                    showMessage(`ลบเอกสาร: "${{filename}}" สำเร็จ!`, 'success');
                    fetchFiles();
                }} catch (error) {{
                    console.error('Deletion error:', error);
                    showMessage(error.message, 'error');
                }}
            }}

            function showMessage(message, type) {{
                statusMessageElement.textContent = message;
                statusMessageElement.className = `status-message ${{type}}`;
                statusMessageElement.style.display = 'block';
                setTimeout(() => {{
                    statusMessageElement.style.display = 'none';
                }}, 5000);
            }}
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/files_list", dependencies=[Depends(get_current_user)])
async def get_files_list():
    if collection is None:
        raise HTTPException(status_code=500, detail="ChromaDB not initialized.")
    
    try:
        results = collection.get(limit=collection.count(), include=['metadatas'])
        unique_filenames = set()
        if results['metadatas']:
            for metadata in results['metadatas']:
                if 'source_filename' in metadata:
                    unique_filenames.add(metadata['source_filename'])
        
        return list(unique_filenames)
    except Exception as e:
        print(f"Error fetching filenames from ChromaDB: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve filenames from the database.")

@app.post("/ingest", dependencies=[Depends(get_current_user)])
async def ingest_document(file: UploadFile = File(...), metadata: str = Form(None)):
    if collection is None or embeddings is None:
        raise HTTPException(status_code=500, detail="RAG system not initialized (ChromaDB or Embedding Model issue).")

    content = await file.read()
    filename = file.filename

    filename_hash = hashlib.sha256(filename.encode('utf-8')).hexdigest()
    
    try:
        collection.delete(where={"_filename_hash": {"$eq": filename_hash}})
        print(f"Successfully deleted all old chunks for filename '{filename}' before adding new ones.")
    except Exception as e:
        print(f"No old chunks found for filename '{filename}' to delete. Proceeding with new ingestion.")

    metadata_dict = {}
    if metadata:
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Metadata JSON is invalid")

    metadata_dict["source_filename"] = filename
    metadata_dict["_filename_hash"] = filename_hash

    raw_text = ""
    file_stream = io.BytesIO(content)
    if filename.endswith(".pdf"):
        try:
            raw_text = get_pdf_text(file_stream)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading PDF file: {e}")
    elif filename.endswith(".docx"):
        try:
            raw_text = get_docx_text(file_stream)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading DOCX file: {e}")
    elif filename.endswith(".txt"):
        try:
            raw_text = get_txt_text(file_stream)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading TXT file: {e}")
    elif filename.endswith(".md"):
        try:
            raw_text = get_md_text(file_stream)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading Markdown file: {e}")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format (.pdf, .docx, .txt, .md)")

    if not raw_text:
        raise HTTPException(status_code=400, detail="No readable text found in the document")

    text_chunks = text_splitter.split_text(raw_text)

    documents_to_add = []
    metadatas_to_add = []
    ids_to_add = []

    for i, chunk in enumerate(text_chunks):
        documents_to_add.append(chunk)
        chunk_metadata = metadata_dict.copy()
        chunk_metadata["chunk_id"] = i
        metadatas_to_add.append(chunk_metadata)
        ids_to_add.append(f"{filename}_{i}")

    try:
        if documents_to_add:
            chunk_embeddings = embeddings.embed_documents(documents_to_add)
            collection.add(
                documents=documents_to_add,
                metadatas=metadatas_to_add,
                embeddings=chunk_embeddings,
                ids=ids_to_add
            )
            print(f"Added {len(documents_to_add)} chunks to ChromaDB from {filename}")
            print(f"Total documents in collection now: {collection.count()}")
        else:
            print(f"No chunks to add for {filename}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding data to ChromaDB: {e}")

    return {
        "message": f"Successfully uploaded and processed '{filename}'",
        "filename": filename,
        "metadata": metadata_dict,
        "chunks_added": len(documents_to_add),
        "total_documents_in_db": collection.count() if collection else 0
    }

@app.delete("/delete_document", dependencies=[Depends(get_current_user)])
async def delete_document(payload: dict):
    if collection is None:
        raise HTTPException(status_code=500, detail="ChromaDB is not ready for document deletion")

    filename = payload.get("filename")
    if not filename:
        raise HTTPException(status_code=400, detail="Please provide the filename to delete in the JSON payload: {'filename': 'your_file.pdf'}")

    try:
        file_hash = hashlib.sha256(filename.encode('utf-8')).hexdigest()
        deleted_ids = collection.delete(where={"source_filename": {"$eq": filename}})

        deleted_ids_count = len(deleted_ids['ids']) if deleted_ids and deleted_ids['ids'] else 0
        
        if deleted_ids_count > 0:
            print(f"Deleted {deleted_ids_count} chunks related to '{filename}'.")
            return {"message": f"Successfully deleted data from file '{filename}'!", "chunks_deleted": deleted_ids_count}
        else:
            print(f"No chunks found for filename '{filename}' to delete. Check filename or metadata.")
            return {"message": f"No data found for file '{filename}'", "chunks_deleted": 0}

    except Exception as e:
        print(f"Error during document deletion: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while deleting the document: {e}")

@app.post("/query", dependencies=[Depends(get_current_user)])
async def query_rag(payload: dict, request: Request):
    if collection is None or embeddings is None or llm_qa is None or llm_memory_summarizer is None or vectorstore is None:
        raise HTTPException(status_code=500, detail="RAG system not fully initialized.")

    query = payload.get("query")
    filters = payload.get("filters", {})
    top_k = payload.get("top_k", 5)

    if not query:
        raise HTTPException(status_code=400, detail="Please provide a question")
        
    try:
        retriever_kwargs = {"k": top_k}
        if filters:
            chroma_filters = {key: {"$eq": value} for key, value in filters.items()}
            retriever_kwargs["filter"] = chroma_filters

        retriever = vectorstore.as_retriever(search_kwargs=retriever_kwargs)
        
        relevant_docs = retriever.invoke(query)
        
        template = """
        คุณคือผู้ช่วย AI ที่เชี่ยวชาญในการตอบคำถามจากบริบทที่ได้รับเท่านั้น โดยละเอียดและสุภาพ 
        คุณจะได้รับประวัติการสนทนาและบริบทบางส่วนจากเอกสาร ตอบลงท้ายด้วย "ครับ"
        
        ---
        ประวัติการสนทนา:
        {chat_history}
        ---
        บริบทจากเอกสาร:
        {context}
        ---
        คำถาม:
        {question}

        คำตอบ:
        """

        prompt = PromptTemplate.from_template(template)
        
        rag_chain = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(relevant_docs)))
            | prompt
            | llm_qa
            | StrOutputParser()
        )
        
        chat_history = []
        
        chain_input = {"question": query, "chat_history": chat_history}
        answer = rag_chain.invoke(chain_input)
        
        source_files = set([doc.metadata.get('source_filename', 'Unknown Source') for doc in relevant_docs])
        
        response_data = {
            "answer": answer,
            "relevant_sources": list(source_files),
            "source_chunks": [{"content": c.page_content, "metadata": c.metadata} for c in relevant_docs]
        }
        
        return response_data
    
    except Exception as e:
        print(f"Error during query: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during the query: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
