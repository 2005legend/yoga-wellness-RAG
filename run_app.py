import uvicorn
import os
import sys

def main():
    print("Starting Yoga Wellness RAG Application...")
    print("Ensure you have processed the knowledge base first: python scripts/process_knowledge_base.py")
    
    # Check if we can import the app
    try:
        from backend.api.main import app
        print("Application backend found.")
    except ImportError as e:
        print(f"Error importing app: {e}")
        sys.exit(1)
        
    print("Starting server on http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("Frontend: http://localhost:8000/static/index.html (or root based on serving)")
    
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()

