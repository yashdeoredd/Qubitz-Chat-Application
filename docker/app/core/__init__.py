"""
Core classes and interfaces for multi-agent system.
"""

from .base_strategy import BaseStrategy, StrategyExecutor
from .agent_registry import AgentRegistry
from .execution_manager import ExecutionManager

__all__ = [
    'BaseStrategy',
    'StrategyExecutor', 
    'AgentRegistry',
    'ExecutionManager'
]