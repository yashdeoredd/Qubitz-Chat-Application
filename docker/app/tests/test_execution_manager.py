"""
Unit tests for execution manager.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from app.models.multi_agent_models import (
    AgentConfig,
    AgentRole,
    ExecutionStatus,
    CoordinationEvent
)
from app.core.agent_registry import AgentRegistry
from app.core.execution_manager import ExecutionManager


class TestExecutionManager:
    """Test ExecutionManager class."""
    
    def test_initialization(self):
        """Test execution manager initialization."""
        registry = AgentRegistry()
        manager = ExecutionManager(registry)
        
        assert manager.agent_registry == registry
        assert len(manager._active_executions) == 0
        assert len(manager._coordination_events) == 0
        assert len(manager._execution_callbacks) == 0
    
    def test_add_remove_execution_callback(self):
        """Test adding and removing execution callbacks."""
        registry = AgentRegistry()
        manager = ExecutionManager(registry)
        
        callback1 = Mock()
        callback2 = Mock()
        
        manager.add_execution_callback(callback1)
        manager.add_execution_callback(callback2)
        
        assert len(manager._execution_callbacks) == 2
        assert callback1 in manager._execution_callbacks
        assert callback2 in manager._execution_callbacks
        
        manager.remove_execution_callback(callback1)
        
        assert len(manager._execution_callbacks) == 1
        assert callback1 not in manager._execution_callbacks
        assert callback2 in manager._execution_callbacks
    
    @pytest.mark.asyncio
    async def test_execute_single_agent_success(self):
        """Test successful single agent execution."""
        registry = AgentRegistry()
        manager = ExecutionManager(registry)
        
        # Setup agent config and mock agent
        config = AgentConfig(
            name="test_agent",
            role=AgentRole.RESEARCH,
            system_prompt="Test prompt"
        )
        registry.register_agent_config(config)
        
        mock_agent = Mock()
        mock_agent.return_value = "Test response"
        
        with patch.object(registry, 'get_agent_instance', return_value=mock_agent):
            result = await manager.execute_single_agent("test_agent", "test message")
        
        assert result.agent_name == "test_agent"
        assert result.agent_role == AgentRole.RESEARCH
        assert result.status == ExecutionStatus.COMPLETED
        assert result.content == "Test response"
        assert result.error_message is None
        assert result.execution_time is not None
        mock_agent.assert_called_once_with("test message")
    
    @pytest.mark.asyncio
    async def test_execute_single_agent_not_found(self):
        """Test single agent execution when agent not found."""
        registry = AgentRegistry()
        manager = ExecutionManager(registry)
        
        with patch.object(registry, 'get_agent_instance', return_value=None):
            result = await manager.execute_single_agent("nonexistent", "test message")
        
        assert result.agent_name == "nonexistent"
        assert result.status == ExecutionStatus.FAILED
        assert result.content == ""
        assert "not found or disabled" in result.error_message
    
    @pytest.mark.asyncio
    async def test_execute_single_agent_exception(self):
        """Test single agent execution with exception."""
        registry = AgentRegistry()
        manager = ExecutionManager(registry)
        
        # Setup agent config
        config = AgentConfig(
            name="test_agent",
            role=AgentRole.ANALYSIS,
            system_prompt="Test prompt"
        )
        registry.register_agent_config(config)
        
        mock_agent = Mock()
        mock_agent.side_effect = RuntimeError("Test error")
        
        with patch.object(registry, 'get_agent_instance', return_value=mock_agent):
            result = await manager.execute_single_agent("test_agent", "test message")
        
        assert result.agent_name == "test_agent"
        assert result.agent_role == AgentRole.ANALYSIS
        assert result.status == ExecutionStatus.FAILED
        assert result.content == ""
        assert result.error_message == "Test error"
        assert result.execution_time is not None
    
    @pytest.mark.asyncio
    async def test_execute_parallel_agents(self):
        """Test parallel agent execution."""
        registry = AgentRegistry()
        manager = ExecutionManager(registry)
        
        # Setup agent configs
        config1 = AgentConfig(
            name="agent1",
            role=AgentRole.RESEARCH,
            system_prompt="Research prompt"
        )
        config2 = AgentConfig(
            name="agent2",
            role=AgentRole.ANALYSIS,
            system_prompt="Analysis prompt"
        )
        registry.register_agent_config(config1)
        registry.register_agent_config(config2)
        
        mock_agent1 = Mock()
        mock_agent1.return_value = "Response 1"
        mock_agent2 = Mock()
        mock_agent2.return_value = "Response 2"
        
        def mock_get_instance(name):
            if name == "agent1":
                return mock_agent1
            elif name == "agent2":
                return mock_agent2
            return None
        
        with patch.object(registry, 'get_agent_instance', side_effect=mock_get_instance):
            results = await manager.execute_parallel_agents(
                ["agent1", "agent2"], 
                "test message"
            )
        
        assert len(results) == 2
        
        # Results might be in any order due to parallel execution
        agent_names = [r.agent_name for r in results]
        assert "agent1" in agent_names
        assert "agent2" in agent_names
        
        for result in results:
            assert result.status == ExecutionStatus.COMPLETED
            assert result.content in ["Response 1", "Response 2"]
    
    @pytest.mark.asyncio
    async def test_execute_parallel_agents_with_exception(self):
        """Test parallel agent execution with one agent failing."""
        registry = AgentRegistry()
        manager = ExecutionManager(registry)
        
        # Setup agent configs
        config1 = AgentConfig(
            name="agent1",
            role=AgentRole.RESEARCH,
            system_prompt="Research prompt"
        )
        config2 = AgentConfig(
            name="agent2",
            role=AgentRole.ANALYSIS,
            system_prompt="Analysis prompt"
        )
        registry.register_agent_config(config1)
        registry.register_agent_config(config2)
        
        mock_agent1 = Mock()
        mock_agent1.return_value = "Response 1"
        mock_agent2 = Mock()
        mock_agent2.side_effect = RuntimeError("Agent 2 failed")
        
        def mock_get_instance(name):
            if name == "agent1":
                return mock_agent1
            elif name == "agent2":
                return mock_agent2
            return None
        
        with patch.object(registry, 'get_agent_instance', side_effect=mock_get_instance):
            results = await manager.execute_parallel_agents(
                ["agent1", "agent2"], 
                "test message"
            )
        
        assert len(results) == 2
        
        # Find results by agent name
        agent1_result = next(r for r in results if r.agent_name == "agent1")
        agent2_result = next(r for r in results if r.agent_name == "agent2")
        
        assert agent1_result.status == ExecutionStatus.COMPLETED
        assert agent1_result.content == "Response 1"
        
        assert agent2_result.status == ExecutionStatus.FAILED
        assert agent2_result.error_message == "Agent 2 failed"
    
    def test_record_coordination_event(self):
        """Test recording coordination events."""
        registry = AgentRegistry()
        manager = ExecutionManager(registry)
        
        manager.record_coordination_event(
            event_type="HANDOFF",
            source_agent="agent1",
            target_agent="agent2",
            context={"key": "value"},
            success=True
        )
        
        assert len(manager._coordination_events) == 1
        event = manager._coordination_events[0]
        
        assert event.event_type == "HANDOFF"
        assert event.source_agent == "agent1"
        assert event.target_agent == "agent2"
        assert event.context == {"key": "value"}
        assert event.success is True
        assert event.error_message is None
    
    def test_record_coordination_event_failure(self):
        """Test recording failed coordination event."""
        registry = AgentRegistry()
        manager = ExecutionManager(registry)
        
        manager.record_coordination_event(
            event_type="COLLABORATION",
            source_agent="agent1",
            success=False,
            error_message="Communication failed"
        )
        
        assert len(manager._coordination_events) == 1
        event = manager._coordination_events[0]
        
        assert event.event_type == "COLLABORATION"
        assert event.source_agent == "agent1"
        assert event.target_agent is None
        assert event.success is False
        assert event.error_message == "Communication failed"
    
    def test_coordination_events_limit(self):
        """Test coordination events limit."""
        registry = AgentRegistry()
        manager = ExecutionManager(registry)
        
        # Add more than 10000 events
        for i in range(10100):
            manager.record_coordination_event(
                event_type="TEST",
                source_agent=f"agent{i}"
            )
        
        # Should be limited to 10000
        assert len(manager._coordination_events) == 10000
        # Should keep the most recent ones
        assert manager._coordination_events[0].source_agent == "agent100"
        assert manager._coordination_events[-1].source_agent == "agent10099"
    
    def test_get_coordination_events_no_filter(self):
        """Test getting coordination events without filters."""
        registry = AgentRegistry()
        manager = ExecutionManager(registry)
        
        # Add some events
        manager.record_coordination_event("HANDOFF", "agent1", "agent2")
        manager.record_coordination_event("COLLABORATION", "agent2", "agent3")
        manager.record_coordination_event("CONFLICT_RESOLUTION", "agent1")
        
        events = manager.get_coordination_events()
        
        assert len(events) == 3
        # Should be sorted by timestamp (most recent first)
        assert events[0].event_type == "CONFLICT_RESOLUTION"
        assert events[1].event_type == "COLLABORATION"
        assert events[2].event_type == "HANDOFF"
    
    def test_get_coordination_events_with_filters(self):
        """Test getting coordination events with filters."""
        registry = AgentRegistry()
        manager = ExecutionManager(registry)
        
        # Add some events
        manager.record_coordination_event("HANDOFF", "agent1", "agent2")
        manager.record_coordination_event("COLLABORATION", "agent2", "agent3")
        manager.record_coordination_event("HANDOFF", "agent3", "agent1")
        manager.record_coordination_event("CONFLICT_RESOLUTION", "agent1")
        
        # Filter by event type
        handoff_events = manager.get_coordination_events(event_type="HANDOFF")
        assert len(handoff_events) == 2
        assert all(e.event_type == "HANDOFF" for e in handoff_events)
        
        # Filter by agent name
        agent1_events = manager.get_coordination_events(agent_name="agent1")
        assert len(agent1_events) == 3  # agent1 as source or target
        
        # Filter with limit
        limited_events = manager.get_coordination_events(limit=2)
        assert len(limited_events) == 2
    
    @pytest.mark.asyncio
    async def test_notify_callbacks_sync(self):
        """Test notifying synchronous callbacks."""
        registry = AgentRegistry()
        manager = ExecutionManager(registry)
        
        callback = Mock()
        manager.add_execution_callback(callback)
        
        await manager._notify_callbacks("test_event", {"key": "value"})
        
        callback.assert_called_once_with("test_event", {"key": "value"})
    
    @pytest.mark.asyncio
    async def test_notify_callbacks_async(self):
        """Test notifying asynchronous callbacks."""
        registry = AgentRegistry()
        manager = ExecutionManager(registry)
        
        callback = AsyncMock()
        manager.add_execution_callback(callback)
        
        await manager._notify_callbacks("test_event", {"key": "value"})
        
        callback.assert_called_once_with("test_event", {"key": "value"})
    
    @pytest.mark.asyncio
    async def test_notify_callbacks_exception_handling(self):
        """Test callback exception handling."""
        registry = AgentRegistry()
        manager = ExecutionManager(registry)
        
        failing_callback = Mock()
        failing_callback.side_effect = RuntimeError("Callback failed")
        working_callback = Mock()
        
        manager.add_execution_callback(failing_callback)
        manager.add_execution_callback(working_callback)
        
        # Should not raise exception
        await manager._notify_callbacks("test_event", {"key": "value"})
        
        # Both callbacks should be called
        failing_callback.assert_called_once()
        working_callback.assert_called_once()
    
    def test_get_execution_statistics(self):
        """Test getting execution statistics."""
        registry = AgentRegistry()
        manager = ExecutionManager(registry)
        
        # Add some coordination events
        manager.record_coordination_event("HANDOFF", "agent1", success=True)
        manager.record_coordination_event("COLLABORATION", "agent2", success=True)
        manager.record_coordination_event("CONFLICT_RESOLUTION", "agent1", success=False)
        manager.record_coordination_event("HANDOFF", "agent3", success=True)
        
        stats = manager.get_execution_statistics()
        
        assert stats["total_coordination_events"] == 4
        assert stats["successful_coordination_events"] == 3
        assert stats["coordination_success_rate"] == 0.75
        assert stats["event_type_counts"]["HANDOFF"] == 2
        assert stats["event_type_counts"]["COLLABORATION"] == 1
        assert stats["event_type_counts"]["CONFLICT_RESOLUTION"] == 1
        assert stats["active_executions"] == 0