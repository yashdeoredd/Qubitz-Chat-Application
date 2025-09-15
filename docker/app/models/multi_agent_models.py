"""
Core data models for multi-agent system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel


class PatternType(str, Enum):
    """Enumeration of multi-agent orchestration patterns."""
    SINGLE = "single"
    SWARM = "swarm"
    WORKFLOW = "workflow"
    GRAPH = "graph"
    A2A = "a2a"


class AgentRole(str, Enum):
    """Enumeration of agent roles and specializations."""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    COORDINATION = "coordination"
    GENERAL_CHAT = "general_chat"
    WORKFLOW = "workflow"


class ExecutionStatus(str, Enum):
    """Enumeration of execution statuses."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class TimeoutConfig:
    """Configuration for timeout settings."""
    agent_timeout: int = 300  # 5 minutes in seconds
    workflow_timeout: int = 900  # 15 minutes in seconds
    coordination_timeout: int = 180  # 3 minutes in seconds
    streaming_timeout: int = 60  # 1 minute in seconds


@dataclass
class ExecutionParams:
    """Parameters for agent execution."""
    max_agents: Optional[int] = None
    timeout_config: TimeoutConfig = field(default_factory=TimeoutConfig)
    retry_attempts: int = 3
    enable_streaming: bool = False
    coordination_preference: Optional[PatternType] = None
    context_sharing: bool = True
    parallel_execution: bool = False


@dataclass
class AgentResult:
    """Result from a single agent execution."""
    agent_name: str
    agent_role: AgentRole
    status: ExecutionStatus
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: Optional[float] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExecutionMetadata:
    """Metadata about multi-agent execution."""
    start_time: datetime
    end_time: Optional[datetime] = None
    total_execution_time: Optional[float] = None
    agents_used: List[str] = field(default_factory=list)
    pattern_used: Optional[PatternType] = None
    coordination_events_count: int = 0
    retry_count: int = 0
    resource_usage: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoordinationEvent:
    """Event representing coordination between agents."""
    timestamp: datetime
    event_type: str  # HANDOFF, COLLABORATION, CONFLICT_RESOLUTION, ESCALATION
    source_agent: str
    target_agent: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class AgentStrategy:
    """Strategy for multi-agent execution."""
    pattern_type: PatternType
    agents: List[str]  # Agent identifiers to use
    coordination_config: Dict[str, Any] = field(default_factory=dict)
    execution_params: ExecutionParams = field(default_factory=ExecutionParams)
    priority: int = 1  # Higher numbers = higher priority
    fallback_strategy: Optional['AgentStrategy'] = None


@dataclass
class CollaborationConfig:
    """Configuration for agent collaboration preferences."""
    preferred_patterns: List[PatternType] = field(default_factory=lambda: [PatternType.SINGLE])
    handoff_triggers: List[str] = field(default_factory=list)
    conflict_resolution_strategy: str = "consensus"
    timeout_settings: TimeoutConfig = field(default_factory=TimeoutConfig)
    max_handoffs: int = 5
    enable_context_sharing: bool = True


@dataclass
class AgentConfig:
    """Configuration for individual agents."""
    name: str
    role: AgentRole
    system_prompt: str
    tools: List[str] = field(default_factory=list)
    specialization_areas: List[str] = field(default_factory=list)
    collaboration_preferences: CollaborationConfig = field(default_factory=CollaborationConfig)
    enabled: bool = True
    priority: int = 1


class MultiAgentRequest(BaseModel):
    """Request model for multi-agent chat interactions."""
    message: str
    system_prompt: Optional[str] = None
    strategy_hint: Optional[str] = None
    max_agents: Optional[int] = None
    coordination_preference: Optional[PatternType] = None
    enable_streaming: bool = False
    context: Dict[str, Any] = {}
    user_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class MultiAgentResult:
    """Result from multi-agent execution."""
    status: ExecutionStatus
    primary_result: str
    agent_contributions: Dict[str, AgentResult] = field(default_factory=dict)
    execution_metadata: ExecutionMetadata = field(default_factory=lambda: ExecutionMetadata(start_time=datetime.now()))
    coordination_log: List[CoordinationEvent] = field(default_factory=list)
    strategy_used: Optional[AgentStrategy] = None
    error_context: Optional['ErrorContext'] = None


@dataclass
class ErrorContext:
    """Context information for error handling and recovery."""
    error_type: str
    error_message: str
    failed_agent: Optional[str] = None
    failed_strategy: Optional[AgentStrategy] = None
    recovery_attempts: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryAction:
    """Action to take for error recovery."""
    action_type: str  # RETRY, FALLBACK, ESCALATE, ABORT
    fallback_strategy: Optional[AgentStrategy] = None
    retry_count: int = 0
    escalation_required: bool = False
    user_notification: Optional[str] = None