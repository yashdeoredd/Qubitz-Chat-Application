"""
HTTP-related tools for the chat application.
"""

from strands import tool
from strands_tools import http_request

# Re-export the http_request tool for easy access
__all__ = ['http_request', 'ready_to_summarize']

@tool
def ready_to_summarize():
    """
    A tool that is intended to be called by the agent right before summarizing the response.
    This signals that the agent is ready to provide a final summary to the user.
    """
    return "Ok - continue providing the summary!"