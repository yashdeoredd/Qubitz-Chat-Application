"""
Agent configuration for the chat application.
"""

import boto3
from strands import Agent
from strands.models import BedrockModel
from strands_tools import swarm, workflow, graph
from tools import AVAILABLE_TOOLS, ready_to_summarize, http_request
from .prompts import (
    CHAT_SYSTEM_PROMPT,
    RESEARCH_SPECIALIST_PROMPT,
    ANALYSIS_SPECIALIST_PROMPT,
    COORDINATION_SPECIALIST_PROMPT,
    GENERAL_CHAT_PROMPT
)


def create_bedrock_model() -> BedrockModel:
    """
    Create a configured BedrockModel with us-east-1 region.
    
    Returns:
        Configured BedrockModel instance.
    """
    # Create a boto3 session configured for us-east-1
    session = boto3.Session(region_name='us-east-1')
    
    return BedrockModel(
        model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",  # Using us. prefix for us-east-1
        boto_session=session,
        temperature=0.7,
        streaming=True
    )


def create_chat_agent(system_prompt: str = None) -> Agent:
    """
    Create a chat agent with the specified system prompt.
    
    Args:
        system_prompt: The system prompt to use. Defaults to CHAT_SYSTEM_PROMPT.
        
    Returns:
        Configured Agent instance for chat interactions.
    """
    prompt = system_prompt or CHAT_SYSTEM_PROMPT
    bedrock_model = create_bedrock_model()
    
    return Agent(
        model=bedrock_model,
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
    bedrock_model = create_bedrock_model()
    
    return Agent(
        model=bedrock_model,
        system_prompt=prompt,
        tools=[http_request, ready_to_summarize],
        callback_handler=None
    )


def create_research_agent() -> Agent:
    """
    Create a specialized research agent with research-specific tools and capabilities.
    
    Returns:
        Configured Agent instance specialized for research tasks.
    """
    bedrock_model = create_bedrock_model()
    
    return Agent(
        model=bedrock_model,
        system_prompt=RESEARCH_SPECIALIST_PROMPT,
        tools=[http_request, ready_to_summarize],
        callback_handler=None
    )


def create_analysis_agent() -> Agent:
    """
    Create a specialized analysis agent with analysis-specific tools and capabilities.
    
    Returns:
        Configured Agent instance specialized for data analysis tasks.
    """
    bedrock_model = create_bedrock_model()
    
    return Agent(
        model=bedrock_model,
        system_prompt=ANALYSIS_SPECIALIST_PROMPT,
        tools=[http_request, ready_to_summarize],
        callback_handler=None
    )


def create_coordination_agent() -> Agent:
    """
    Create a specialized coordination agent with multi-agent orchestration tools.
    
    Returns:
        Configured Agent instance specialized for agent coordination tasks.
    """
    bedrock_model = create_bedrock_model()
    
    return Agent(
        model=bedrock_model,
        system_prompt=COORDINATION_SPECIALIST_PROMPT,
        tools=[swarm, workflow, graph, http_request, ready_to_summarize],
        callback_handler=None
    )


def create_general_chat_agent() -> Agent:
    """
    Create a general-purpose chat agent for simple conversational queries.
    
    Returns:
        Configured Agent instance for general chat interactions.
    """
    bedrock_model = create_bedrock_model()
    
    return Agent(
        model=bedrock_model,
        system_prompt=GENERAL_CHAT_PROMPT,
        tools=[ready_to_summarize],
        callback_handler=None
    )