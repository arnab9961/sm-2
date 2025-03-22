from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import google.generativeai as genai
import json
from typing import List, Optional, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import logging
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Import settings from config and environment variables
try:
    # Get Gemini API key from environment variable
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    # Import other settings from config
    from app.config import EMBEDDING_MODEL, DATASET_PATH
    from app.config import MAX_LENGTH, TEMPERATURE, TOP_P

    if not gemini_api_key:
        logger.error("GEMINI_API_KEY not found in .env file")
        raise ValueError("GEMINI_API_KEY is required")

    # Configure the Gemini API
    genai.configure(api_key=gemini_api_key)

except ImportError:
    logger.error("Config file not found. Please create app/config.py from the template.")
    raise

# Global variables
model = None
embedding_model = None
dataset = {}
faq_embeddings = []
faq_items = []

# Startup and shutdown event handlers
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models and data
    await startup_event()
    yield
    # Shutdown: Clean up resources
    await shutdown_event()

async def startup_event():
    """Initialize models and load data on startup"""
    global model, embedding_model, dataset, faq_embeddings, faq_items
    
    logger.info("Initializing application...")
    
    try:
        # Create a Gemini model instance
        model = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("Gemini model initialized successfully")
        
        # Load sentence transformer for embeddings
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info(f"Embedding model {EMBEDDING_MODEL} loaded successfully")
        
        # Load the dataset
        await load_dataset()
        
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        raise

async def shutdown_event():
    """Clean up resources on shutdown"""
    logger.info("Shutting down application...")
    # Add cleanup code if needed

async def load_dataset():
    """Load and process the dataset"""
    global dataset, faq_embeddings, faq_items
    
    try:
        if not os.path.exists(DATASET_PATH):
            logger.warning(f"Dataset file not found at {DATASET_PATH}. Creating empty dataset.")
            dataset = {"faq": [], "company": {}}
            with open(DATASET_PATH, 'w') as f:
                json.dump(dataset, f, indent=4)
        else:
            with open(DATASET_PATH, 'r') as f:
                dataset = json.load(f)
            logger.info(f"Dataset loaded successfully from {DATASET_PATH}")
        
        # Precompute embeddings for all questions and answers in the dataset
        if "faq" in dataset:
            faq_data = dataset["faq"]
            faq_texts = [item["question"] + " " + item["answer"] for item in faq_data]
            if faq_texts:
                faq_embeddings = embedding_model.encode(faq_texts)
                faq_items = faq_data
                logger.info(f"Precomputed embeddings for {len(faq_texts)} FAQ items")
    
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

# FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again later."}
    )

# Pydantic models
class ChatInput(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="User message")

class DatasetItem(BaseModel):
    text: str = Field(..., min_length=1, description="Question text")
    response: str = Field(..., min_length=1, description="Answer text")

class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[Dict[str, Any]]] = None

# Dependency for rate limiting (example, implement as needed)
async def rate_limiter(request: Request):
    # Implement rate limiting logic here
    return True

# Root endpoint
@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Function to find relevant context from the dataset
def retrieve_context(query, top_k=3, threshold=0.6):
    query_lower = query.lower()
    matched_items = []

    # Handle company-specific questions with direct matching
    if any(term in query_lower for term in ["owner", "owns", "owned by", "own", "ownership", "ceo"]):
        if any(term in query_lower for term in ["sm", "sm tech", "sm technology", "bdcalling", "bdcalling it"]):
            if "parent_company_info" in dataset.get("company", {}) and "owner" in dataset["company"]["parent_company_info"]:
                matched_items.append({
                    "question": "Who is the owner of SM Technology?",
                    "answer": f"The owner of SM Technology's parent company, bdCalling IT, is {dataset['company']['parent_company_info']['owner']}. Since SM Technology is a sister concern of bdCalling IT, {dataset['company']['parent_company_info']['owner']} is effectively the owner of SM Technology as well.",
                    "similarity": 1.0
                })

    # Direct matching for GM questions
    if any(term in query_lower for term in ["gm", "general manager"]):
        if any(term in query_lower for term in ["sm", "sm tech", "sm technology"]):
            if "management" in dataset.get("company", {}) and "general_manager" in dataset["company"]["management"]:
                matched_items.append({
                    "question": "Who is the General Manager of SM Technology?",
                    "answer": f"{dataset['company']['management']['general_manager']} is the General Manager (GM) of SM Technology.",
                    "similarity": 1.0
                })

    if any(term in query_lower for term in ["gm", "general manager", "sales"]):
        if any(term in query_lower for term in ["sm", "sm tech", "sm technology"]):
            if "management" in dataset.get("company", {}) and "gm_sales" in dataset["company"]["management"]:
                matched_items.append({
                    "question": "Who is the General Manager of Sales of SM Technology?",
                    "answer": f"{dataset['company']['management']['gm_sales']} is the General Manager (GM) of Sales at SM Technology.",
                    "similarity": 1.0
                })

    # Direct matching for sister concerns
    if any(term in query_lower for term in ["sister concern", "sister concerns", "subsidiaries", "related companies"]):
        if any(term in query_lower for term in ["bdcalling", "bdcalling it"]):
            if "parent_company_info" in dataset.get("company", {}) and "sister_concerns" in dataset["company"]["parent_company_info"]:
                sister_concerns_list = ", ".join(dataset["company"]["parent_company_info"]["sister_concerns"])
                matched_items.append({
                    "question": "What are the sister concerns of bdCalling IT?",
                    "answer": f"The sister concerns of bdCalling IT are: {sister_concerns_list}.",
                    "similarity": 1.0
                })

    # If direct matches were found, return them
    if matched_items:
        return matched_items

    # If no direct matches and no embeddings available, return empty list
    if len(faq_embeddings) == 0:
        return []

    # Otherwise, perform semantic search
    try:
        # Encode the query
        query_embedding = embedding_model.encode([query])[0]

        # Calculate similarity between query and all FAQ questions
        similarities = cosine_similarity([query_embedding], faq_embeddings)[0]

        # Get top_k most similar questions
        most_similar_indices = np.argsort(similarities)[::-1][:top_k]

        # Filter by threshold
        context = []
        for idx in most_similar_indices:
            if float(similarities[idx]) >= threshold:
                context.append({
                    "question": faq_items[idx]["question"],
                    "answer": faq_items[idx]["answer"],
                    "similarity": float(similarities[idx])
                })

        return context
    except Exception as e:
        logger.error(f"Error in retrieve_context: {str(e)}")
        return []

# Chat endpoint with improved error handling and response structure
@app.post("/chat/", response_model=ChatResponse)
async def chat(chat_input: ChatInput, _: bool = Depends(rate_limiter)):
    try:
        # Process the user query
        user_query = chat_input.message.lower()
        logger.info(f"Received chat query: {user_query}")

        # First check for direct matches
        for pattern_check in [
            # Ownership questions
            {"terms": ["owner", "owns", "owned by", "own", "ownership", "ceo"], 
             "company_terms": ["sm", "sm tech", "sm technology", "bdcalling", "bdcalling it"],
             "data_path": ["company", "parent_company_info", "owner"],
             "response_template": "The owner of SM Technology's parent company, bdCalling IT, is {0}. Since SM Technology is a sister concern of bdCalling IT, {0} is effectively the owner of SM Technology as well."},
            
            # GM questions
            {"terms": ["gm", "general manager"], 
             "company_terms": ["sm", "sm tech", "sm technology"],
             "data_path": ["company", "management", "general_manager"],
             "response_template": "{0} is the General Manager (GM) of SM Technology."},
            
            # GM Sales questions
            {"terms": ["gm", "general manager", "sales"], 
             "company_terms": ["sm", "sm tech", "sm technology"],
             "data_path": ["company", "management", "gm_sales"],
             "response_template": "{0} is the General Manager (GM) of Sales at SM Technology."},
            
            # Sister concerns
            {"terms": ["sister concern", "sister concerns", "subsidiaries", "related companies"], 
             "company_terms": ["bdcalling", "bdcalling it"],
             "data_path": ["company", "parent_company_info", "sister_concerns"],
             "response_template": "The sister concerns of bdCalling IT are: {0}.",
             "join_list": True}
        ]:
            if any(term in user_query for term in pattern_check["terms"]) and any(term in user_query for term in pattern_check["company_terms"]):
                # Navigate through the data path
                current_data = dataset
                valid_path = True
                for path_item in pattern_check["data_path"]:
                    if path_item in current_data:
                        current_data = current_data[path_item]
                    else:
                        valid_path = False
                        break
                
                if valid_path:
                    if "join_list" in pattern_check and pattern_check["join_list"] and isinstance(current_data, list):
                        value = ", ".join(current_data)
                    else:
                        value = current_data
                    
                    return {"response": pattern_check["response_template"].format(value), "sources": None}

        # Check for exact matches in FAQ data
        if "faq" in dataset:
            for item in dataset["faq"]:
                if user_query in item["question"].lower():
                    return {"response": item["answer"], "sources": None}

        # Special case for graphics design course
        if "graphics" in user_query and "course" in user_query:
            return {"response": "SM Technology does not currently offer graphics design courses.", "sources": None}

        # Semantic search for relevant context (RAG approach)
        relevant_context = retrieve_context(chat_input.message)
        logger.info(f"Found {len(relevant_context)} relevant context items")

        # Prepare context for the model
        context_text = ""
        if relevant_context:
            context_text = "Here is some relevant information:\n"
            for ctx in relevant_context:
                context_text += f"Q: {ctx['question']}\nA: {ctx['answer']}\n\n"

        # Enhanced system prompt with improved instructions
        system_prompt = """You are the SM Technology chatbot.
Answer directly with information from the context only.
Keep responses concise and to the point.
For unknown information say only: "This information isn't available in my knowledge base."
Be professional and helpful.
If the question is about pricing, mention the starting price and be specific.
If the question is about services, list the services clearly.
If the question is about technologies, list the technologies clearly.
If the question is about sister concerns, list the sister concerns clearly.
If the question is about the CEO, mention the CEO of bdCalling IT.
If the question is about the chairperson, mention the chairperson of bdCalling IT.
NO reasoning, NO numbered points, NO explanations unless specifically asked for.
"""

        try:
            # Configure generation parameters
            generation_config = genai.GenerationConfig(
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_output_tokens=MAX_LENGTH,
            )

            # Include context in prompt if available
            if context_text:
                prompt = f"{system_prompt}\n\n{context_text}\nAnswer the following question based on the information above when relevant.\nUser: {chat_input.message}"
            else:
                # Improved prompt for handling questions not in the dataset
                prompt = f"{system_prompt}\n\nAnswer the following technology-related question to the best of your ability. Be honest about limitations in your knowledge.\nUser: {chat_input.message}"

            # Generate response using Gemini API
            response = model.generate_content(
                prompt,
                generation_config=generation_config
            )

            response_text = response.text.strip()

            # If we still got a problematic response, use a fallback
            if len(response_text) < 10:
                return {"response": "I don't have enough information to answer that question accurately. Could you please ask something else?", "sources": None}

            # Return response with sources for transparency
            return {
                "response": response_text, 
                "sources": relevant_context if relevant_context else None
            }

        except Exception as inner_e:
            logger.error(f"Model generation error: {str(inner_e)}")
            return {"response": "I'm having trouble processing your request. Could you try asking in a different way?", "sources": None}

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to view the entire dataset
@app.get("/dataset/")
async def get_dataset():
    try:
        if "faq" in dataset:
            return dataset["faq"]
        return []
    except Exception as e:
        logger.error(f"Error retrieving dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to add new data to the dataset
@app.post("/dataset/")
async def add_to_dataset(item: DatasetItem):
    try:
        if "faq" not in dataset:
            dataset["faq"] = []

        # Check for duplicates
        for existing_item in dataset["faq"]:
            if existing_item["question"].lower() == item.text.lower():
                return JSONResponse(
                    status_code=400,
                    content={"detail": "A question with similar text already exists in the dataset."}
                )

        # Add new item to dataset
        new_item = {
            "question": item.text,
            "answer": item.response
        }
        dataset["faq"].append(new_item)

        # Update embeddings
        global faq_embeddings, faq_items
        faq_items = dataset["faq"]
        faq_texts = [item["question"] + " " + item["answer"] for item in faq_items]
        faq_embeddings = embedding_model.encode(faq_texts)

        # Save updated dataset
        with open(DATASET_PATH, 'w') as f:
            json.dump(dataset, f, indent=4)
            
        logger.info(f"Added new item to dataset: {item.text}")
        return {"message": "Item added to dataset successfully!"}
    except Exception as e:
        logger.error(f"Error adding to dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to update existing FAQ item
@app.put("/dataset/{item_id}")
async def update_dataset_item(item_id: int, item: DatasetItem):
    try:
        if "faq" not in dataset or item_id < 0 or item_id >= len(dataset["faq"]):
            raise HTTPException(status_code=404, detail="Item not found")
        
        # Update the item
        dataset["faq"][item_id]["question"] = item.text
        dataset["faq"][item_id]["answer"] = item.response
        
        # Update embeddings
        global faq_embeddings, faq_items
        faq_items = dataset["faq"]
        faq_texts = [item["question"] + " " + item["answer"] for item in faq_items]
        faq_embeddings = embedding_model.encode(faq_texts)
        
        # Save updated dataset
        with open(DATASET_PATH, 'w') as f:
            json.dump(dataset, f, indent=4)
            
        logger.info(f"Updated dataset item {item_id}: {item.text}")
        return {"message": "Item updated successfully!"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating dataset item: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to delete FAQ item
@app.delete("/dataset/{item_id}")
async def delete_dataset_item(item_id: int):
    try:
        if "faq" not in dataset or item_id < 0 or item_id >= len(dataset["faq"]):
            raise HTTPException(status_code=404, detail="Item not found")
        
        # Remove the item
        removed_item = dataset["faq"].pop(item_id)
        
        # Update embeddings
        global faq_embeddings, faq_items
        faq_items = dataset["faq"]
        
        if faq_items:
            faq_texts = [item["question"] + " " + item["answer"] for item in faq_items]
            faq_embeddings = embedding_model.encode(faq_texts)
        else:
            faq_embeddings = []
        
        # Save updated dataset
        with open(DATASET_PATH, 'w') as f:
            json.dump(dataset, f, indent=4)
            
        logger.info(f"Deleted dataset item {item_id}: {removed_item['question']}")
        return {"message": "Item deleted successfully!"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting dataset item: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": model is not None and embedding_model is not None}

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
