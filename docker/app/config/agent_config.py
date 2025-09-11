"""
Agent configuration for the chat application.
"""

from strands import Agent
from tools import AVAILABLE_TOOLS, ready_to_summarize, http_request
from .prompts import CHAT_SYSTEM_PROMPT


def create_chat_agent(system_prompt: str = None) -> Agent:
    """
    Create a chat agent with the specified system prompt.
    
    Args:
        system_prompt: The system prompt to use. Defaults to CHAT_SYSTEM_PROMPT.
        
    Returns:
        Configured Agent instance for chat interactions.
    """
    prompt = system_prompt or CHAT_SYSTEM_PROMPT
    
    return Agent(
        system_prompt=prompt,
        tools=[http_request],
    )


def create_streaming_agent(system_prompt: str = None) -> Agent:
    """
    Create a streaming chat agent with the specified system prompt.
    
    Args:
        system_prompt: The system prompt to use. Defaults to CHAT_SYSTEM_PROMPT.
        
    Returns:
        Configured Agent instance for streaming chat interactions.
    """
    prompt = system_prompt or CHAT_SYSTEM_PROMPT
    
    return Agent(
        system_prompt=prompt,
        tools=[http_request, ready_to_summarize],
        callback_handler=None
    )