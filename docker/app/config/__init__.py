"""
Configuration module for the chat application.
"""

from .prompts import CHAT_SYSTEM_PROMPT, RESEARCH_ASSISTANT_PROMPT, GENERAL_HELPER_PROMPT
from .agent_config import create_chat_agent, create_streaming_agent

__all__ = [
    'CHAT_SYSTEM_PROMPT',
    'RESEARCH_ASSISTANT_PROMPT', 
    'GENERAL_HELPER_PROMPT',
    'create_chat_agent',
    'create_streaming_agent'
]