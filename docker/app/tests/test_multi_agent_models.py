"""
Unit tests for multi-agent data models.
"""

import pytest
from datetime import datetime
from typing import Dict, Any

from app.models.multi_agent_models import (
    PatternType,
    AgentRole,
    ExecutionStatus,
    TimeoutConfig,
    ExecutionParams,
    AgentResult,
    ExecutionMetadata,
    CoordinationEvent,
    AgentStrategy,
    CollaborationConfig,
    AgentConfig,
    MultiAgentRequest,
    MultiAgentResult,
    ErrorContext,
    RecoveryAction
)


class TestEnums:
    """Test enum classes."""
    
    def test_pattern_type_values(self):
        """Test PatternType enum values."""
        assert PatternType.SINGLE == "single"
        assert PatternType.SWARM == "swarm"
        assert PatternType.WORKFLOW == "workflow"
        assert PatternType.GRAPH == "graph"
        assert PatternType.A2A == "a2a"
    
    def test_agent_role_values(self):
        """Test AgentRole enum values."""
        assert AgentRole.RESEARCH == "research"
        assert AgentRole.ANALYSIS == "analysis"
        assert AgentRole.COORDINATION == "coordination"
        assert AgentRole.GENERAL_CHAT == "general_chat"
        assert AgentRole.WORKFLOW == "workflow"
    
    def test_execution_status_values(self):
        """Test ExecutionStatus enum values."""
        assert ExecutionStatus.PENDING == "pending"
        assert ExecutionStatus.RUNNING == "running"
        assert ExecutionStatus.COMPLETED == "completed"
        assert ExecutionStatus.FAILED == "failed"
        assert ExecutionStatus.TIMEOUT == "timeout"
        assert ExecutionStatus.CANCELLED == "cancelled"


class TestTimeoutConfig:
    """Test TimeoutConfig dataclass."""
    
    def test_default_values(self):
        """Test default timeout values."""
        config = TimeoutConfig()
        assert config.agent_timeout == 300
        assert config.workflow_timeout == 900
        assert config.coordination_timeout == 180
        assert config.streaming_timeout == 60
    
    def test_custom_values(self):
        """Test custom timeout values."""
        config = TimeoutConfig(
            agent_timeout=600,
            workflow_timeout=1800,
            coordination_timeout=360,
            streaming_timeout=120
        )
        assert config.agent_timeout == 600
        assert config.workflow_timeout == 1800
        assert config.coordination_timeout == 360
        assert config.streaming_timeout == 120


class TestExecutionParams:
    """Test ExecutionParams dataclass."""
    
    def test_default_values(self):
        """Test default execution parameters."""
        params = ExecutionParams()
        assert params.max_agents is None
        assert isinstance(params.timeout_config, TimeoutConfig)
        assert params.retry_attempts == 3
        assert params.enable_streaming is False
        assert params.coordination_preference is None
        assert params.context_sharing is True
        assert params.parallel_execution is False
    
    def test_custom_values(self):
        """Test custom execution parameters."""
        timeout_config = TimeoutConfig(agent_timeout=600)
        params = ExecutionParams(
            max_agents=5,
            timeout_config=timeout_config,
            retry_attempts=5,
            enable_streaming=True,
            coordination_preference=PatternType.SWARM,
            context_sharing=False,
            parallel_execution=True
        )
        assert params.max_agents == 5
        assert params.timeout_config == timeout_config
        assert params.retry_attempts == 5
        assert params.enable_streaming is True
        assert params.coordination_preference == PatternType.SWARM
        assert params.context_sharing is False
        assert params.parallel_execution is True


class TestAgentResult:
    """Test AgentResult dataclass."""
    
    def test_successful_result(self):
        """Test successful agent result."""
        result = AgentResult(
            agent_name="test_agent",
            agent_role=AgentRole.RESEARCH,
            status=ExecutionStatus.COMPLETED,
            content="Test response",
            execution_time=1.5
        )
        assert result.agent_name == "test_agent"
        assert result.agent_role == AgentRole.RESEARCH
        assert result.status == ExecutionStatus.COMPLETED
        assert result.content == "Test response"
        assert result.execution_time == 1.5
        assert result.error_message is None
        assert isinstance(result.timestamp, datetime)
        assert isinstance(result.metadata, dict)
    
    def test_failed_result(self):
        """Test failed agent result."""
        result = AgentResult(
            agent_name="test_agent",
            agent_role=AgentRole.ANALYSIS,
            status=ExecutionStatus.FAILED,
            content="",
            error_message="Test error"
        )
        assert result.agent_name == "test_agent"
        assert result.agent_role == AgentRole.ANALYSIS
        assert result.status == ExecutionStatus.FAILED
        assert result.content == ""
        assert result.error_message == "Test error"


class TestCoordinationEvent:
    """Test CoordinationEvent dataclass."""
    
    def test_successful_event(self):
        """Test successful coordination event."""
        event = CoordinationEvent(
            timestamp=datetime.now(),
            event_type="HANDOFF",
            source_agent="agent1",
            target_agent="agent2",
            context={"key": "value"}
        )
        assert event.event_type == "HANDOFF"
        assert event.source_agent == "agent1"
        assert event.target_agent == "agent2"
        assert event.context == {"key": "value"}
        assert event.success is True
        assert event.error_message is None
    
    def test_failed_event(self):
        """Test failed coordination event."""
        event = CoordinationEvent(
            timestamp=datetime.now(),
            event_type="COLLABORATION",
            source_agent="agent1",
            success=False,
            error_message="Communication failed"
        )
        assert event.event_type == "COLLABORATION"
        assert event.source_agent == "agent1"
        assert event.target_agent is None
        assert event.success is False
        assert event.error_message == "Communication failed"


class TestAgentStrategy:
    """Test AgentStrategy dataclass."""
    
    def test_basic_strategy(self):
        """Test basic agent strategy."""
        strategy = AgentStrategy(
            pattern_type=PatternType.SWARM,
            agents=["agent1", "agent2"]
        )
        assert strategy.pattern_type == PatternType.SWARM
        assert strategy.agents == ["agent1", "agent2"]
        assert isinstance(strategy.coordination_config, dict)
        assert isinstance(strategy.execution_params, ExecutionParams)
        assert strategy.priority == 1
        assert strategy.fallback_strategy is None
    
    def test_complex_strategy(self):
        """Test complex agent strategy with all parameters."""
        fallback = AgentStrategy(
            pattern_type=PatternType.SINGLE,
            agents=["fallback_agent"]
        )
        
        strategy = AgentStrategy(
            pattern_type=PatternType.WORKFLOW,
            agents=["agent1", "agent2", "agent3"],
            coordination_config={"max_steps": 5},
            execution_params=ExecutionParams(max_agents=3),
            priority=2,
            fallback_strategy=fallback
        )
        assert strategy.pattern_type == PatternType.WORKFLOW
        assert strategy.agents == ["agent1", "agent2", "agent3"]
        assert strategy.coordination_config == {"max_steps": 5}
        assert strategy.priority == 2
        assert strategy.fallback_strategy == fallback


class TestAgentConfig:
    """Test AgentConfig dataclass."""
    
    def test_basic_config(self):
        """Test basic agent configuration."""
        config = AgentConfig(
            name="test_agent",
            role=AgentRole.RESEARCH,
            system_prompt="You are a research assistant."
        )
        assert config.name == "test_agent"
        assert config.role == AgentRole.RESEARCH
        assert config.system_prompt == "You are a research assistant."
        assert config.tools == []
        assert config.specialization_areas == []
        assert isinstance(config.collaboration_preferences, CollaborationConfig)
        assert config.enabled is True
        assert config.priority == 1
    
    def test_full_config(self):
        """Test full agent configuration."""
        collab_config = CollaborationConfig(
            preferred_patterns=[PatternType.SWARM, PatternType.WORKFLOW]
        )
        
        config = AgentConfig(
            name="research_agent",
            role=AgentRole.RESEARCH,
            system_prompt="You are a specialized research agent.",
            tools=["http_request", "web_search"],
            specialization_areas=["web_research", "data_collection"],
            collaboration_preferences=collab_config,
            enabled=True,
            priority=2
        )
        assert config.name == "research_agent"
        assert config.role == AgentRole.RESEARCH
        assert config.tools == ["http_request", "web_search"]
        assert config.specialization_areas == ["web_research", "data_collection"]
        assert config.collaboration_preferences == collab_config
        assert config.priority == 2


class TestMultiAgentRequest:
    """Test MultiAgentRequest model."""
    
    def test_basic_request(self):
        """Test basic multi-agent request."""
        request = MultiAgentRequest(message="Hello, world!")
        assert request.message == "Hello, world!"
        assert request.system_prompt is None
        assert request.strategy_hint is None
        assert request.max_agents is None
        assert request.coordination_preference is None
        assert request.enable_streaming is False
        assert request.context == {}
        assert request.user_id is None
        assert request.session_id is None
    
    def test_full_request(self):
        """Test full multi-agent request."""
        request = MultiAgentRequest(
            message="Analyze this data",
            system_prompt="You are an analyst",
            strategy_hint="use_swarm",
            max_agents=3,
            coordination_preference=PatternType.SWARM,
            enable_streaming=True,
            context={"data": "sample"},
            user_id="user123",
            session_id="session456"
        )
        assert request.message == "Analyze this data"
        assert request.system_prompt == "You are an analyst"
        assert request.strategy_hint == "use_swarm"
        assert request.max_agents == 3
        assert request.coordination_preference == PatternType.SWARM
        assert request.enable_streaming is True
        assert request.context == {"data": "sample"}
        assert request.user_id == "user123"
        assert request.session_id == "session456"


class TestMultiAgentResult:
    """Test MultiAgentResult dataclass."""
    
    def test_basic_result(self):
        """Test basic multi-agent result."""
        result = MultiAgentResult(
            status=ExecutionStatus.COMPLETED,
            primary_result="Task completed successfully"
        )
        assert result.status == ExecutionStatus.COMPLETED
        assert result.primary_result == "Task completed successfully"
        assert isinstance(result.agent_contributions, dict)
        assert isinstance(result.execution_metadata, ExecutionMetadata)
        assert isinstance(result.coordination_log, list)
        assert result.strategy_used is None
        assert result.error_context is None
    
    def test_complex_result(self):
        """Test complex multi-agent result."""
        agent_result = AgentResult(
            agent_name="test_agent",
            agent_role=AgentRole.RESEARCH,
            status=ExecutionStatus.COMPLETED,
            content="Research complete"
        )
        
        coordination_event = CoordinationEvent(
            timestamp=datetime.now(),
            event_type="HANDOFF",
            source_agent="agent1",
            target_agent="agent2"
        )
        
        strategy = AgentStrategy(
            pattern_type=PatternType.SWARM,
            agents=["agent1", "agent2"]
        )
        
        result = MultiAgentResult(
            status=ExecutionStatus.COMPLETED,
            primary_result="Multi-agent task completed",
            agent_contributions={"test_agent": agent_result},
            coordination_log=[coordination_event],
            strategy_used=strategy
        )
        assert result.agent_contributions["test_agent"] == agent_result
        assert result.coordination_log[0] == coordination_event
        assert result.strategy_used == strategy


class TestErrorContext:
    """Test ErrorContext dataclass."""
    
    def test_basic_error_context(self):
        """Test basic error context."""
        context = ErrorContext(
            error_type="ValueError",
            error_message="Invalid input"
        )
        assert context.error_type == "ValueError"
        assert context.error_message == "Invalid input"
        assert context.failed_agent is None
        assert context.failed_strategy is None
        assert context.recovery_attempts == 0
        assert isinstance(context.timestamp, datetime)
        assert isinstance(context.context, dict)
    
    def test_full_error_context(self):
        """Test full error context."""
        strategy = AgentStrategy(
            pattern_type=PatternType.SINGLE,
            agents=["failed_agent"]
        )
        
        context = ErrorContext(
            error_type="TimeoutError",
            error_message="Agent execution timed out",
            failed_agent="test_agent",
            failed_strategy=strategy,
            recovery_attempts=2,
            context={"timeout_duration": 300}
        )
        assert context.error_type == "TimeoutError"
        assert context.error_message == "Agent execution timed out"
        assert context.failed_agent == "test_agent"
        assert context.failed_strategy == strategy
        assert context.recovery_attempts == 2
        assert context.context == {"timeout_duration": 300}


class TestRecoveryAction:
    """Test RecoveryAction dataclass."""
    
    def test_basic_recovery_action(self):
        """Test basic recovery action."""
        action = RecoveryAction(action_type="RETRY")
        assert action.action_type == "RETRY"
        assert action.fallback_strategy is None
        assert action.retry_count == 0
        assert action.escalation_required is False
        assert action.user_notification is None
    
    def test_full_recovery_action(self):
        """Test full recovery action."""
        fallback_strategy = AgentStrategy(
            pattern_type=PatternType.SINGLE,
            agents=["backup_agent"]
        )
        
        action = RecoveryAction(
            action_type="FALLBACK",
            fallback_strategy=fallback_strategy,
            retry_count=3,
            escalation_required=True,
            user_notification="Switching to backup agent"
        )
        assert action.action_type == "FALLBACK"
        assert action.fallback_strategy == fallback_strategy
        assert action.retry_count == 3
        assert action.escalation_required is True
        assert action.user_notification == "Switching to backup agent"