from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
import uvicorn
from nexus import NEXUS
import json
import os

app = FastAPI(title="Knowledge Nexus")

# Get the absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Mount static files and templates
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Initialize NEXUS
nexus = NEXUS()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process-text")
async def process_text(text: str = Form(...)):
    print(f"Processing text: {text}")  # Debug print
    # Extract entities and relationships
    triplets = nexus.extract_entities_relationships(text)
    print(f"Extracted triplets: {triplets}")  # Debug print
    
    # Build knowledge graph
    nexus.build_knowledge_graph(triplets)
    print(f"Knowledge graph nodes: {list(nexus.knowledge_graph.nodes())}")  # Debug print
    print(f"Knowledge graph edges: {list(nexus.knowledge_graph.edges(data=True))}")  # Debug print
    
    # Cluster the text
    clusters = nexus.cluster_documents([text])
    
    return {
        "triplets": triplets,
        "clusters": clusters
    }

@app.post("/query")
async def process_query(query: str = Form(...)):
    print(f"Processing query: {query}")  # Debug print
    print(f"Current knowledge graph nodes: {list(nexus.knowledge_graph.nodes())}")  # Debug print
    results = nexus.process_query(query)
    print(f"Query results: {results}")  # Debug print
    return results

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
