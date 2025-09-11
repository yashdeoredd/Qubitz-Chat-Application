"""
Main FastAPI application for the chat service.
"""

import os
import uvicorn
from fastapi import FastAPI

from api import (
    health_check,
    chat_endpoint,
    chat_streaming_endpoint,
    ChatRequest,
    HealthResponse
)

# Create FastAPI application
app = FastAPI(
    title="Chat API",
    description="A general-purpose chat API powered by AI agents",
    version="1.0.0"
)

# Health check endpoint
@app.get('/health', response_model=HealthResponse)
def health():
    """Health check endpoint for the load balancer."""
    return health_check()

# Chat endpoints
@app.post('/chat')
async def chat(request: ChatRequest):
    """Endpoint to get chat responses."""
    return await chat_endpoint(request)

@app.post('/chat-streaming')
async def chat_streaming(request: ChatRequest):
    """Endpoint to stream chat responses as they are generated."""
    return await chat_streaming_endpoint(request)

# Legacy endpoints for backward compatibility
@app.post('/weather')
async def weather_legacy(request: dict):
    """Legacy weather endpoint - redirects to chat with weather context."""
    chat_request = ChatRequest(
        message=request.get('prompt', ''),
        system_prompt="You are a weather assistant. Help users with weather-related questions and information."
    )
    return await chat_endpoint(chat_request)

@app.post('/weather-streaming')
async def weather_streaming_legacy(request: dict):
    """Legacy weather streaming endpoint - redirects to chat streaming with weather context."""
    chat_request = ChatRequest(
        message=request.get('prompt', ''),
        system_prompt="You are a weather assistant. Help users with weather-related questions and information."
    )
    return await chat_streaming_endpoint(chat_request)

if __name__ == '__main__':
    # Get port from environment variable or default to 8000
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run(app, host='0.0.0.0', port=port)