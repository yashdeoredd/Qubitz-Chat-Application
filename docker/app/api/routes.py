"""
API routes for the chat application.
"""

from fastapi import HTTPException
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel

from config.agent_config import create_chat_agent, create_streaming_agent


class ChatRequest(BaseModel):
    """Request model for chat interactions."""
    message: str
    system_prompt: str = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str


def health_check() -> HealthResponse:
    """Health check endpoint for the load balancer."""
    return HealthResponse(status="healthy")


async def chat_endpoint(request: ChatRequest) -> PlainTextResponse:
    """
    Endpoint to get chat responses.
    
    Args:
        request: Chat request containing the user message and optional system prompt.
        
    Returns:
        Plain text response from the chat agent.
        
    Raises:
        HTTPException: If no message is provided or if processing fails.
    """
    message = request.message
    
    if not message:
        raise HTTPException(status_code=400, detail="No message provided")

    try:
        chat_agent = create_chat_agent(request.system_prompt)
        response = chat_agent(message)
        content = str(response)
        return PlainTextResponse(content=content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def run_streaming_agent_and_yield_response(message: str, system_prompt: str = None):
    """
    A helper function to yield response text chunks one by one as they come in,
    allowing the web server to emit them to caller live.
    
    Args:
        message: The user message to process.
        system_prompt: Optional system prompt to use.
        
    Yields:
        Text chunks from the streaming agent response.
    """
    is_summarizing = False

    streaming_agent = create_streaming_agent(system_prompt)

    async for item in streaming_agent.stream_async(message):
        if not is_summarizing:
            # Check if this item indicates we're ready to summarize
            if "data" in item and "ready_to_summarize" in str(item.get("data", "")):
                is_summarizing = True
            continue
        if "data" in item:
            yield item['data']


async def chat_streaming_endpoint(request: ChatRequest) -> StreamingResponse:
    """
    Endpoint to stream chat responses as they are generated.
    
    Args:
        request: Chat request containing the user message and optional system prompt.
        
    Returns:
        Streaming response with chat content.
        
    Raises:
        HTTPException: If no message is provided or if processing fails.
    """
    try:
        message = request.message

        if not message:
            raise HTTPException(status_code=400, detail="No message provided")

        return StreamingResponse(
            run_streaming_agent_and_yield_response(message, request.system_prompt),
            media_type="text/plain"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))