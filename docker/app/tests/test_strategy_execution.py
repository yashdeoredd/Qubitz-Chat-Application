"""
Unit tests for strategy execution engine.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from app.core.agent_router import AgentRouter
from app.core.agent_registry import AgentRegistry
from app.core.base_strategy import StrategyExecutor
from app.models.multi_agent_models import (
    MultiAgentRequest,
    MultiAgentResult,
    AgentStrategy,
    AgentResult,
    AgentConfig,
    AgentRole,
    PatternType,
    ExecutionStatus,
    ExecutionMetadata,
    ExecutionParams,
    ErrorContext
)


class TestStrategyExecution:
    """Test cases for strategy execution engine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock dependencies
        self.agent_registry = Mock(spec=AgentRegistry)
        self.strategy_executor = Mock(spec=StrategyExecutor)
        
        # Create agent router
        self.router = AgentRouter(self.agent_registry, self.strategy_executor)
        
        # Set up default mocks
        self._setup_default_mocks()
    
    def _setup_default_mocks(self):
        """Set up default mock responses."""
        # Mock agent configs
        research_config = AgentConfig(
            name="research_agent",
            role=AgentRole.RESEARCH,
            system_prompt="Research specialist",
            enabled=True,
            priority=2
        )
        
        analysis_config = AgentConfig(
            name="analysis_agent",
            role=AgentRole.ANALYSIS,
            system_prompt="Analysis specialist",
            enabled=True,
            priority=2
        )
        
        # Mock registry methods
        self.agent_registry.list_agents_by_role.side_effect = lambda role: {
            AgentRole.RESEARCH: [research_config],
            AgentRole.ANALYSIS: [analysis_config]
        }.get(role, [])
        
        self.agent_registry.get_agent_config.side_effect = lambda name: {
            "research_agent": research_config,
            "analysis_agent": analysis_config
        }.get(name)
    
    @pytest.mark.asyncio
    async def test_successful_strategy_execution(self):
        """Test successful strategy execution with result synthesis."""
        # Create test strategy
        strategy = AgentStrategy(
            pattern_type=PatternType.SWARM,
            agents=["research_agent", "analysis_agent"],
            coordination_config={},
            execution_params=ExecutionParams(),
            priority=1
        )
        
        # Mock successful execution result
        mock_result = MultiAgentResult(
            status=ExecutionStatus.COMPLETED,
            primary_result="Combined analysis complete",
            agent_contributions={
                "research_agent": AgentResult(
                    agent_name="research_agent",
                    agent_role=AgentRole.RESEARCH,
                    status=ExecutionStatus.COMPLETED,
                    content="Research findings: AI is advancing rapidly"
                ),
                "analysis_agent": AgentResult(
                    agent_name="analysis_agent",
                    agent_role=AgentRole.ANALYSIS,
                    status=ExecutionStatus.COMPLETED,
                    content="Analysis shows significant potential"
                )
            },
            execution_metadata=ExecutionMetadata(
                start_time=datetime.now(),
                total_execution_time=5.0
            )
        )
        
        # Mock strategy executor
        self.strategy_executor.execute_strategy = AsyncMock(return_value=mock_result)
        
        # Execute strategy
        request = MultiAgentRequest(message="Analyze AI trends")
        result = await self.router.execute_strategy(request, strategy)
        
        # Verify execution
        assert result.status == ExecutionStatus.COMPLETED
        assert "Collaborative Analysis" in result.primary_result
        assert "research_agent" in result.primary_result
        assert "analysis_agent" in result.primary_result
        assert result.execution_metadata.pattern_used == PatternType.SWARM
        
        # Verify strategy executor was called
        self.strategy_executor.execute_strategy.assert_called_once_with(request, strategy)
    
    @pytest.mark.asyncio
    async def test_workflow_pattern_synthesis(self):
        """Test result synthesis for workflow pattern."""
        # Create workflow strategy
        strategy = AgentStrategy(
            pattern_type=PatternType.WORKFLOW,
            agents=["research_agent", "analysis_agent"],
            coordination_config={},
            execution_params=ExecutionParams(),
            priority=1
        )
        
        # Mock workflow execution result
        mock_result = MultiAgentResult(
            status=ExecutionStatus.COMPLETED,
            primary_result="Workflow complete",
            agent_contributions={
                "research_agent": AgentResult(
                    agent_name="research_agent",
                    agent_role=AgentRole.RESEARCH,
                    status=ExecutionStatus.COMPLETED,
                    content="Step 1: Data collected"
                ),
                "analysis_agent": AgentResult(
                    agent_name="analysis_agent",
                    agent_role=AgentRole.ANALYSIS,
                    status=ExecutionStatus.COMPLETED,
                    content="Step 2: Analysis performed"
                )
            },
            execution_metadata=ExecutionMetadata(
                start_time=datetime.now(),
                total_execution_time=8.0
            )
        )
        
        self.strategy_executor.execute_strategy = AsyncMock(return_value=mock_result)
        
        # Execute strategy
        request = MultiAgentRequest(message="Process data workflow")
        result = await self.router.execute_strategy(request, strategy)
        
        # Verify workflow synthesis
        assert result.status == ExecutionStatus.COMPLETED
        assert "Workflow Execution" in result.primary_result
        assert "Step 1" in result.primary_result
        assert "Step 2" in result.primary_result
        assert result.execution_metadata.pattern_used == PatternType.WORKFLOW
    
    @pytest.mark.asyncio
    async def test_graph_pattern_synthesis(self):
        """Test result synthesis for graph pattern."""
        # Create graph strategy
        strategy = AgentStrategy(
            pattern_type=PatternType.GRAPH,
            agents=["research_agent", "analysis_agent"],
            coordination_config={},
            execution_params=ExecutionParams(),
            priority=1
        )
        
        # Mock graph execution result
        mock_result = MultiAgentResult(
            status=ExecutionStatus.COMPLETED,
            primary_result="Decision tree complete",
            agent_contributions={
                "research_agent": AgentResult(
                    agent_name="research_agent",
                    agent_role=AgentRole.RESEARCH,
                    status=ExecutionStatus.COMPLETED,
                    content="Path A: Research indicates positive trend"
                ),
                "analysis_agent": AgentResult(
                    agent_name="analysis_agent",
                    agent_role=AgentRole.ANALYSIS,
                    status=ExecutionStatus.COMPLETED,
                    content="Path B: Analysis confirms viability"
                )
            },
            execution_metadata=ExecutionMetadata(
                start_time=datetime.now(),
                total_execution_time=6.0
            )
        )
        
        self.strategy_executor.execute_strategy = AsyncMock(return_value=mock_result)
        
        # Execute strategy
        request = MultiAgentRequest(message="Make decision")
        result = await self.router.execute_strategy(request, strategy)
        
        # Verify graph synthesis
        assert result.status == ExecutionStatus.COMPLETED
        assert "Decision Tree Analysis" in result.primary_result
        assert "Decision" in result.primary_result
        assert result.execution_metadata.pattern_used == PatternType.GRAPH
    
    @pytest.mark.asyncio
    async def test_a2a_pattern_synthesis(self):
        """Test result synthesis for agent-to-agent pattern."""
        # Create A2A strategy
        strategy = AgentStrategy(
            pattern_type=PatternType.A2A,
            agents=["research_agent", "analysis_agent"],
            coordination_config={},
            execution_params=ExecutionParams(),
            priority=1
        )
        
        # Mock A2A execution result
        mock_result = MultiAgentResult(
            status=ExecutionStatus.COMPLETED,
            primary_result="Communication complete",
            agent_contributions={
                "research_agent": AgentResult(
                    agent_name="research_agent",
                    agent_role=AgentRole.RESEARCH,
                    status=ExecutionStatus.COMPLETED,
                    content="Shared research data with analysis agent"
                ),
                "analysis_agent": AgentResult(
                    agent_name="analysis_agent",
                    agent_role=AgentRole.ANALYSIS,
                    status=ExecutionStatus.COMPLETED,
                    content="Received data and provided feedback"
                )
            },
            execution_metadata=ExecutionMetadata(
                start_time=datetime.now(),
                total_execution_time=4.0
            )
        )
        
        self.strategy_executor.execute_strategy = AsyncMock(return_value=mock_result)
        
        # Execute strategy
        request = MultiAgentRequest(message="Coordinate agents")
        result = await self.router.execute_strategy(request, strategy)
        
        # Verify A2A synthesis
        assert result.status == ExecutionStatus.COMPLETED
        assert "Agent Communications" in result.primary_result
        assert "Coordinated Response" in result.primary_result
        assert result.execution_metadata.pattern_used == PatternType.A2A
    
    @pytest.mark.asyncio
    async def test_single_agent_execution(self):
        """Test execution with single agent (no synthesis needed)."""
        # Create single agent strategy
        strategy = AgentStrategy(
            pattern_type=PatternType.SINGLE,
            agents=["research_agent"],
            coordination_config={},
            execution_params=ExecutionParams(),
            priority=1
        )
        
        # Mock single agent result
        mock_result = MultiAgentResult(
            status=ExecutionStatus.COMPLETED,
            primary_result="Single agent response",
            agent_contributions={
                "research_agent": AgentResult(
                    agent_name="research_agent",
                    agent_role=AgentRole.RESEARCH,
                    status=ExecutionStatus.COMPLETED,
                    content="Research complete"
                )
            },
            execution_metadata=ExecutionMetadata(
                start_time=datetime.now(),
                total_execution_time=3.0
            )
        )
        
        self.strategy_executor.execute_strategy = AsyncMock(return_value=mock_result)
        
        # Execute strategy
        request = MultiAgentRequest(message="Simple research")
        result = await self.router.execute_strategy(request, strategy)
        
        # Verify single agent execution
        assert result.status == ExecutionStatus.COMPLETED
        assert result.primary_result == "Single agent response"
        assert result.execution_metadata.pattern_used == PatternType.SINGLE
    
    @pytest.mark.asyncio
    async def test_execution_failure_with_fallback(self):
        """Test strategy execution failure with fallback strategy."""
        # Create strategy with fallback
        fallback_strategy = AgentStrategy(
            pattern_type=PatternType.SINGLE,
            agents=["research_agent"],
            coordination_config={'fallback': True},
            execution_params=ExecutionParams(),
            priority=0
        )
        
        main_strategy = AgentStrategy(
            pattern_type=PatternType.SWARM,
            agents=["research_agent", "analysis_agent"],
            coordination_config={},
            execution_params=ExecutionParams(),
            priority=1,
            fallback_strategy=fallback_strategy
        )
        
        # Mock main strategy failure and fallback success
        fallback_result = MultiAgentResult(
            status=ExecutionStatus.COMPLETED,
            primary_result="Fallback successful",
            execution_metadata=ExecutionMetadata(
                start_time=datetime.now(),
                total_execution_time=2.0
            )
        )
        
        self.strategy_executor.execute_strategy.side_effect = [
            Exception("Main strategy failed"),
            fallback_result
        ]
        
        # Execute strategy
        request = MultiAgentRequest(message="Test request")
        result = await self.router.execute_strategy(request, main_strategy)
        
        # Verify fallback was used
        assert result.status == ExecutionStatus.COMPLETED
        assert result.primary_result == "Fallback successful"
        assert result.execution_metadata.pattern_used == PatternType.SINGLE
        
        # Verify both strategies were called
        assert self.strategy_executor.execute_strategy.call_count == 2
    
    @pytest.mark.asyncio
    async def test_execution_failure_without_fallback(self):
        """Test strategy execution failure without fallback."""
        # Create strategy without fallback
        strategy = AgentStrategy(
            pattern_type=PatternType.SWARM,
            agents=["research_agent", "analysis_agent"],
            coordination_config={},
            execution_params=ExecutionParams(),
            priority=1
        )
        
        # Mock strategy failure
        self.strategy_executor.execute_strategy = AsyncMock(
            side_effect=Exception("Strategy execution failed")
        )
        
        # Execute strategy
        request = MultiAgentRequest(message="Test request")
        result = await self.router.execute_strategy(request, strategy)
        
        # Verify error result
        assert result.status == ExecutionStatus.FAILED
        assert "Strategy execution failed" in result.primary_result
        assert result.error_context is not None
        assert result.error_context.error_type == "Exception"
        assert result.execution_metadata.pattern_used == PatternType.SWARM
    
    @pytest.mark.asyncio
    async def test_coordination_summary_addition(self):
        """Test addition of coordination summary to results."""
        # Create multi-agent strategy
        strategy = AgentStrategy(
            pattern_type=PatternType.SWARM,
            agents=["research_agent", "analysis_agent"],
            coordination_config={},
            execution_params=ExecutionParams(),
            priority=1
        )
        
        # Mock execution result with coordination events
        mock_result = MultiAgentResult(
            status=ExecutionStatus.COMPLETED,
            primary_result="Analysis complete",
            agent_contributions={
                "research_agent": AgentResult(
                    agent_name="research_agent",
                    agent_role=AgentRole.RESEARCH,
                    status=ExecutionStatus.COMPLETED,
                    content="Research done"
                ),
                "analysis_agent": AgentResult(
                    agent_name="analysis_agent",
                    agent_role=AgentRole.ANALYSIS,
                    status=ExecutionStatus.COMPLETED,
                    content="Analysis done"
                )
            },
            execution_metadata=ExecutionMetadata(
                start_time=datetime.now(),
                total_execution_time=7.0
            ),
            coordination_log=[]
        )
        
        self.strategy_executor.execute_strategy = AsyncMock(return_value=mock_result)
        
        # Execute strategy
        request = MultiAgentRequest(message="Coordinate analysis")
        result = await self.router.execute_strategy(request, strategy)
        
        # Verify coordination summary was added
        assert hasattr(result.execution_metadata, 'coordination_summary')
        summary = result.execution_metadata.coordination_summary
        assert summary['pattern_used'] == 'swarm'
        assert summary['agents_involved'] == 2
        assert summary['successful_agents'] == 2
        assert summary['execution_time'] == 7.0
    
    @pytest.mark.asyncio
    async def test_performance_metrics_update(self):
        """Test that performance metrics are updated after execution."""
        # Create strategy
        strategy = AgentStrategy(
            pattern_type=PatternType.SINGLE,
            agents=["research_agent"],
            coordination_config={},
            execution_params=ExecutionParams(),
            priority=1
        )
        
        # Mock successful execution
        mock_result = MultiAgentResult(
            status=ExecutionStatus.COMPLETED,
            primary_result="Success",
            execution_metadata=ExecutionMetadata(
                start_time=datetime.now(),
                total_execution_time=3.0
            )
        )
        
        self.strategy_executor.execute_strategy = AsyncMock(return_value=mock_result)
        
        # Execute strategy
        request = MultiAgentRequest(message="Test request")
        await self.router.execute_strategy(request, strategy)
        
        # Verify performance metrics were updated
        # Check the internal performance tracking directly since get_routing_statistics
        # requires routing history which we don't have in this test
        assert 'single' in self.router._strategy_performance
        perf_data = self.router._strategy_performance['single']
        assert perf_data['total_executions'] == 1
        assert perf_data['successful_executions'] == 1
        assert perf_data['success_rate'] == 1.0


if __name__ == "__main__":
    pytest.main([__file__])