from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import json
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Request
import os
from dotenv import load_dotenv

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
        print("Warning: GEMINI_API_KEY not found in .env file")
        raise ValueError("GEMINI_API_KEY is required")
    
    # Configure the Gemini API
    genai.configure(api_key=gemini_api_key)
    
except ImportError:
    print("Error: Config file not found. Please create app/config.py from the template.")
    raise

# Create a Gemini model instance using the Flash 1.5 model
model = genai.GenerativeModel('gemini-1.5-flash')

# Load sentence transformer for embeddings (used in RAG)
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Load the dataset
with open(DATASET_PATH, 'r') as f:
    dataset = json.load(f)

# Precompute embeddings for all questions in the dataset
faq_embeddings = []
faq_items = []
if "faq" in dataset:
    questions = [item["question"] for item in dataset["faq"]]
    if questions:
        faq_embeddings = embedding_model.encode(questions)
        faq_items = dataset["faq"]

# FastAPI app
app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Pydantic model for chat input
class ChatInput(BaseModel):
    message: str

# Pydantic model for adding new data to the dataset
class DatasetItem(BaseModel):
    text: str
    response: str

# Root endpoint
@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Function to find relevant context from the dataset
def retrieve_context(query, top_k=3, threshold=0.6):
    query_lower = query.lower()
    
    # Direct matching for company-specific questions
    if any(term in query_lower for term in ["owner", "owns", "owned by", "own", "ownership", "ceo"]):
        if any(term in query_lower for term in ["sm", "sm tech", "sm technology"]):
            return [{
                "question": "Who is the owner of SM Technology?",
                "answer": "The owner of SM Technology's parent company, bdCalling IT, is MD. Monir Hossain. Since SM Technology is a sister concern of bdCalling IT, MD. Monir Hossain is effectively the owner of SM Technology as well.",
                "similarity": 1.0
            }]
    
    # Direct matching for GM questions
    if any(term in query_lower for term in ["gm", "general manager"]):
        if any(term in query_lower for term in ["sm", "sm tech", "sm technology"]):
            return [{
                "question": "Who is the General Manager of SM Technology?",
                "answer": "MD. Shamim Miah is the General Manager (GM) of SM Technology.",
                "similarity": 1.0
            }]
    
    if len(faq_embeddings) == 0:
        return []
    
    # Encode the query
    query_embedding = embedding_model.encode([query])[0]
    
    # Calculate similarity between query and all FAQ questions
    similarities = cosine_similarity([query_embedding], faq_embeddings)[0]
    
    # Get top_k most similar questions
    most_similar_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Filter by threshold
    context = []
    for idx in most_similar_indices:
        if float(similarities[idx]) >= threshold:  # Explicitly convert to float
            context.append({
                "question": faq_items[idx]["question"],
                "answer": faq_items[idx]["answer"],
                "similarity": float(similarities[idx])  # Explicitly convert to float
            })
    
    return context

# Chat endpoint
@app.post("/chat/")
async def chat(chat_input: ChatInput):
    try:
        # Process the user query
        user_query = chat_input.message.lower()
        
        # Enhanced direct matching for ownership and management questions
        if any(term in user_query for term in ["owner", "owns", "owned by", "own", "ownership", "ceo"]):
            if any(term in user_query for term in ["sm", "sm tech", "sm technology"]):
                return {
                    "response": "The owner of SM Technology's parent company, bdCalling IT, is MD. Monir Hossain. Since SM Technology is a sister concern of bdCalling IT, MD. Monir Hossain is effectively the owner of SM Technology as well."
                }
                
        # Direct matching for GM questions
        if any(term in user_query for term in ["gm", "general manager"]):
            if any(term in user_query for term in ["sm", "sm tech", "sm technology"]):
                return {
                    "response": "MD. Shamim Miah is the General Manager (GM) of SM Technology."
                }
        
        # Exact match check
        if "faq" in dataset:
            for item in dataset["faq"]:
                if chat_input.message.lower() in item["question"].lower():
                    return {"response": item["answer"]}
                
        # For the specific graphics design question
        if "graphics" in user_query and "course" in user_query:
            if "faq" in dataset and not relevant_context:
                return {"response": "SM Technology does not currently offer graphics design courses in our curriculum."}
        
        # Semantic search for relevant context (RAG approach)
        relevant_context = retrieve_context(chat_input.message)
        
        # Prepare context for the model
        context_text = ""
        if relevant_context:
            context_text = "Here is some relevant information:\n"
            for ctx in relevant_context:
                context_text += f"Q: {ctx['question']}\nA: {ctx['answer']}\n\n"
        
        # Enhanced system prompt to guide the model behavior
        system_prompt = """You are the SM Technology chatbot.
Answer directly with information from the context only.
Keep responses to 1 sentence when possible.
For unknown information say only: "This information isn't available in my knowledge base."
NO reasoning, NO numbered points, NO explanations.
Be professional and concise."""
        
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
                return {"response": "I don't have enough information to answer that question accurately. Could you please ask something else?"}
                
            return {"response": response_text}
            
        except Exception as inner_e:
            print(f"Model generation error: {str(inner_e)}")
            return {"response": "I'm having trouble processing your request. Could you try asking in a different way?"}
            
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to view the entire dataset
@app.get("/dataset/")
async def get_dataset():
    if "faq" in dataset:
        return dataset["faq"]
    return []

# Endpoint to add new data to the dataset
@app.post("/dataset/")
async def add_to_dataset(item: DatasetItem):
    try:
        if "faq" not in dataset:
            dataset["faq"] = []
        
        # Add new item to dataset
        dataset["faq"].append({
            "question": item.text,
            "answer": item.response
        })
        
        # Update embeddings
        global faq_embeddings, faq_items
        faq_items = dataset["faq"]
        questions = [item["question"] for item in faq_items]
        faq_embeddings = embedding_model.encode(questions)
        
        # Save updated dataset
        with open(DATASET_PATH, 'w') as f:
            json.dump(dataset, f, indent=4)
        return {"message": "Item added to dataset successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)