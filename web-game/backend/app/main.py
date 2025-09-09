import sys
import os
from pathlib import Path

# Add the main project to Python path for imports
main_project_path = Path(__file__).parent.parent.parent.parent / "debate_mnist"
if main_project_path.exists():
    sys.path.insert(0, str(main_project_path))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import time
import traceback
from dotenv import load_dotenv

from app.api.game import router as game_router
from app.api.health import router as health_router

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="AI Debate Game API",
    description="Backend API for the Human vs AI Pixel Debate Game",
    version="1.0.0",
)

# Configure CORS properly for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # Legacy port
        "http://localhost:3001",   # Current React development server
        "http://127.0.0.1:3000",   # Legacy
        "http://127.0.0.1:3001",   # Current
        "https://*.cloudfront.net",  # CloudFront distribution
        "https://*.amazonaws.com",   # API Gateway
    ],
    allow_credentials=False,  # Set to False for better security
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Add timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Include routers
app.include_router(health_router, prefix="/api", tags=["health"])
app.include_router(game_router, prefix="/api/game", tags=["game"])

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with better logging"""
    # Log errors in development only
    if os.getenv("DEBUG", "false").lower() == "true":
        print(f"Global exception handler caught: {exc}")
        print(f"Request: {request.method} {request.url}")
        print(f"Traceback: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "type": "server_error",
            "timestamp": time.time(),
            "error": str(exc) if os.getenv("DEBUG", "false").lower() == "true" else "Something went wrong"
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP exception handler with better error format"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "type": "client_error",
            "timestamp": time.time()
        }
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Debate Game API",
        "status": "running",
        "version": "1.0.0"
    }

# Lambda handler for AWS deployment
try:
    from mangum import Mangum
    handler = Mangum(app, lifespan="off")
except ImportError:
    # Mangum not available - running in standard mode
    handler = None

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )