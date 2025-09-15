"""
Unit tests for agent router.
"""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime

from app.core.agent_router import AgentRouter
from app.core.agent_registry import AgentRegistry
from app.core.base_strategy import StrategyExecutor, BaseStrategy
from app.models.multi_agent_models import (
    MultiAgentRequest,
    MultiAgentResult,
    AgentStrategy,
    AgentConfig,
    AgentRole,
    PatternType,
    ExecutionStatus,
    ExecutionMetadata
)


class MockStrategy(BaseStrategy):
    """Mock strategy for testing."""
    
    def __init__(self, name: str):
        super().__init__(name)
    
    async def execute(self, request: MultiAgentRequest, strategy: AgentStrategy) -> MultiAgentResult:
        return MultiAgentResult(
            status=ExecutionStatus.COMPLETED,
            primary_result="Mock result",
            execution_metadata=ExecutionMetadata(start_time=datetime.now())
        )
    
    def validate_strategy(self, strategy: AgentStrategy) -> bool:
        return len(strategy.agents) > 0


class TestAgentRouter:
    """Test cases for AgentRouter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock agent registry
        self.agent_registry = Mock(spec=AgentRegistry)
        
        # Create mock strategy executor
        self.strategy_executor = Mock(spec=StrategyExecutor)
        
        # Create agent router
        self.router = AgentRouter(self.agent_registry, self.strategy_executor)
        
        # Set up default mock responses
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
        
        general_config = AgentConfig(
            name="general_agent",
            role=AgentRole.GENERAL_CHAT,
            system_prompt="General chat agent",
            enabled=True,
            priority=1
        )
        
        coordination_config = AgentConfig(
            name="coordination_agent",
            role=AgentRole.COORDINATION,
            system_prompt="Coordination specialist",
            enabled=True,
            priority=3
        )
        
        # Mock registry methods
        self.agent_registry.list_agents_by_role.side_effect = lambda role: {
            AgentRole.RESEARCH: [research_config],
            AgentRole.ANALYSIS: [analysis_config],
            AgentRole.GENERAL_CHAT: [general_config],
            AgentRole.COORDINATION: [coordination_config]
        }.get(role, [])
        
        self.agent_registry.get_agent_config.side_effect = lambda name: {
            "research_agent": research_config,
            "analysis_agent": analysis_config,
            "general_agent": general_config,
            "coordination_agent": coordination_config
        }.get(name)
        
        self.agent_registry.list_agent_configs.return_value = [
            research_config, analysis_config, general_config, coordination_config
        ]
        
        # Mock strategy executor
        mock_strategy = MockStrategy("test_strategy")
        self.strategy_executor.get_strategy.return_value = mock_strategy
    
    def test_route_simple_request(self):
        """Test routing of simple requests."""
        request = MultiAgentRequest(message="Hello")
        strategy = self.router.route_request(request)
        
        assert strategy.pattern_type == PatternType.SINGLE
        assert len(strategy.agents) >= 1
        assert strategy.priority >= 0
    
    def test_route_research_request(self):
        """Test routing of research requests."""
        request = MultiAgentRequest(message="Find information about climate change")
        strategy = self.router.route_request(request)
        
        assert "research_agent" in strategy.agents
        assert strategy.coordination_config['domain'] == 'research'
    
    def test_route_analysis_request(self):
        """Test routing of analysis requests."""
        request = MultiAgentRequest(message="Analyze the pros and cons of renewable energy")
        strategy = self.router.route_request(request)
        
        assert "analysis_agent" in strategy.agents
        assert strategy.coordination_config['domain'] == 'analysis'
    
    def test_route_complex_multi_step_request(self):
        """Test routing of complex multi-step requests."""
        request = MultiAgentRequest(
            message="First research market trends, then analyze the data, and create a report"
        )
        strategy = self.router.route_request(request)
        
        assert strategy.pattern_type in [PatternType.WORKFLOW, PatternType.SWARM]
        assert len(strategy.agents) > 1
        assert strategy.coordination_config['complexity'] in ['complex', 'very_complex']
    
    def test_strategy_hint_override(self):
        """Test that strategy hints override default routing."""
        request = MultiAgentRequest(
            message="Simple question",
            strategy_hint="swarm"
        )
        strategy = self.router.route_request(request)
        
        assert strategy.pattern_type == PatternType.SWARM
    
    def test_coordination_preference_override(self):
        """Test that coordination preference overrides default routing."""
        request = MultiAgentRequest(
            message="Simple question",
            coordination_preference=PatternType.WORKFLOW
        )
        strategy = self.router.route_request(request)
        
        assert strategy.pattern_type == PatternType.WORKFLOW
    
    def test_max_agents_limit(self):
        """Test that max_agents limit is respected."""
        request = MultiAgentRequest(
            message="Complex research and analysis task",
            max_agents=2
        )
        strategy = self.router.route_request(request)
        
        assert len(strategy.agents) <= 2
    
    def test_fallback_strategy_creation(self):
        """Test fallback strategy creation when routing fails."""
        # Mock registry to raise exception during agent lookup
        self.agent_registry.list_agents_by_role.side_effect = Exception("Agent lookup failed")
        
        # Mock general agents available for fallback
        general_config = AgentConfig(
            name="fallback_agent",
            role=AgentRole.GENERAL_CHAT,
            system_prompt="Fallback agent",
            enabled=True,
            priority=1
        )
        
        # Override the side effect for general chat agents only in fallback
        def mock_fallback_lookup(role):
            if role == AgentRole.GENERAL_CHAT:
                return [general_config]
            raise Exception("Agent lookup failed")
        
        # We need to patch the fallback method directly since the exception handling calls it
        original_create_fallback = self.router._create_fallback_strategy
        
        def mock_create_fallback(request):
            from app.models.multi_agent_models import ExecutionParams
            return AgentStrategy(
                pattern_type=PatternType.SINGLE,
                agents=["fallback_agent"],
                coordination_config={'fallback': True, 'reason': 'routing_failure'},
                execution_params=ExecutionParams(max_agents=1, retry_attempts=1),
                priority=0
            )
        
        self.router._create_fallback_strategy = mock_create_fallback
        
        request = MultiAgentRequest(message="Test request")
        strategy = self.router.route_request(request)
        
        # Should use fallback strategy
        assert strategy.pattern_type == PatternType.SINGLE
        assert strategy.coordination_config.get('fallback') is True
        assert "fallback_agent" in strategy.agents
    
    def test_agent_availability_validation(self):
        """Test that only available agents are included in strategy."""
        # Mock one agent as disabled
        disabled_config = AgentConfig(
            name="disabled_agent",
            role=AgentRole.RESEARCH,
            system_prompt="Disabled agent",
            enabled=False
        )
        
        enabled_config = AgentConfig(
            name="enabled_agent",
            role=AgentRole.RESEARCH,
            system_prompt="Enabled agent",
            enabled=True
        )
        
        # Mock the registry to return both agents for the role
        def mock_list_agents_by_role(role):
            if role == AgentRole.RESEARCH:
                return [disabled_config, enabled_config]
            elif role == AgentRole.GENERAL_CHAT:
                return [AgentConfig(
                    name="general_agent",
                    role=AgentRole.GENERAL_CHAT,
                    system_prompt="General chat agent",
                    enabled=True,
                    priority=1
                )]
            return []
        
        self.agent_registry.list_agents_by_role.side_effect = mock_list_agents_by_role
        self.agent_registry.get_agent_config.side_effect = lambda name: {
            "disabled_agent": disabled_config,
            "enabled_agent": enabled_config,
            "general_agent": AgentConfig(
                name="general_agent",
                role=AgentRole.GENERAL_CHAT,
                system_prompt="General chat agent",
                enabled=True,
                priority=1
            )
        }.get(name)
        
        request = MultiAgentRequest(message="Research request")
        strategy = self.router.route_request(request)
        
        # Should only include enabled agents
        assert "enabled_agent" in strategy.agents
        assert "disabled_agent" not in strategy.agents
    
    def test_strategy_priority_calculation(self):
        """Test strategy priority calculation."""
        # Complex request should have higher priority than simple request
        complex_request = MultiAgentRequest(
            message="First research machine learning algorithms, then analyze their performance, "
                   "compare different approaches, and create a comprehensive technical report"
        )
        complex_strategy = self.router.route_request(complex_request)
        
        # Simple request
        simple_request = MultiAgentRequest(message="Hello")
        simple_strategy = self.router.route_request(simple_request)
        
        # Complex request should have higher priority due to complexity boost
        assert complex_strategy.priority > simple_strategy.priority
    
    def test_coordination_config_creation(self):
        """Test coordination configuration creation for different patterns."""
        # Test swarm pattern
        swarm_request = MultiAgentRequest(
            message="Collaborate on analyzing multiple data sources",
            strategy_hint="swarm"
        )
        swarm_strategy = self.router.route_request(swarm_request)
        
        assert swarm_strategy.pattern_type == PatternType.SWARM
        assert 'max_handoffs' in swarm_strategy.coordination_config
        assert 'collaboration_mode' in swarm_strategy.coordination_config
        
        # Test workflow pattern
        workflow_request = MultiAgentRequest(
            message="First do A, then B, then C",
            strategy_hint="workflow"
        )
        workflow_strategy = self.router.route_request(workflow_request)
        
        assert workflow_strategy.pattern_type == PatternType.WORKFLOW
        assert 'execution_order' in workflow_strategy.coordination_config
        assert 'dependency_handling' in workflow_strategy.coordination_config
    
    def test_performance_tracking(self):
        """Test strategy performance tracking."""
        request = MultiAgentRequest(message="Test request")
        strategy = self.router.route_request(request)
        
        # Mock execution result
        result = MultiAgentResult(
            status=ExecutionStatus.COMPLETED,
            primary_result="Success",
            execution_metadata=ExecutionMetadata(
                start_time=datetime.now(),
                total_execution_time=5.0
            )
        )
        
        # Update performance
        self.router.update_strategy_performance(strategy, result)
        
        # Check statistics
        stats = self.router.get_routing_statistics()
        assert stats['total_routes'] > 0
        assert strategy.pattern_type.value in stats['strategy_performance']
    
    def test_adaptive_routing_optimization(self):
        """Test adaptive routing optimizations."""
        # Enable adaptive routing
        self.router.set_adaptive_routing(True)
        
        # Create some performance history
        strategy = AgentStrategy(
            pattern_type=PatternType.SWARM,
            agents=["test_agent"],
            coordination_config={},
            priority=1
        )
        
        # Mock poor performance
        self.router._strategy_performance[PatternType.SWARM.value] = {
            'avg_execution_time': 600,  # 10 minutes
            'success_rate': 0.5  # 50% success rate
        }
        
        request = MultiAgentRequest(message="Test request", strategy_hint="swarm")
        optimized_strategy = self.router.route_request(request)
        
        # Should have adjusted timeouts and retry attempts
        assert optimized_strategy.execution_params.timeout_config.agent_timeout > 300
        assert optimized_strategy.execution_params.retry_attempts > 3
    
    def test_recommended_strategy_with_confidence(self):
        """Test getting recommended strategy with confidence score."""
        request = MultiAgentRequest(message="Research climate change impacts")
        strategy, confidence = self.router.get_recommended_strategy(request)
        
        assert isinstance(strategy, AgentStrategy)
        assert 0.0 <= confidence <= 1.0
        assert "research_agent" in strategy.agents
    
    def test_routing_statistics(self):
        """Test routing statistics collection."""
        # Route several requests
        requests = [
            MultiAgentRequest(message="Hello"),
            MultiAgentRequest(message="Research AI"),
            MultiAgentRequest(message="Analyze data trends")
        ]
        
        for request in requests:
            self.router.route_request(request)
        
        stats = self.router.get_routing_statistics()
        
        assert stats['total_routes'] == 3
        assert 'pattern_distribution' in stats
        assert 'domain_distribution' in stats
        assert 'complexity_distribution' in stats
        assert 'average_confidence' in stats
    
    def test_clear_performance_history(self):
        """Test clearing performance history."""
        # Route a request to create history
        request = MultiAgentRequest(message="Test request")
        self.router.route_request(request)
        
        # Verify history exists
        stats_before = self.router.get_routing_statistics()
        assert stats_before['total_routes'] > 0
        
        # Clear history
        self.router.clear_performance_history()
        
        # Verify history is cleared
        stats_after = self.router.get_routing_statistics()
        assert stats_after['total_routes'] == 0
    
    def test_error_handling_in_routing(self):
        """Test error handling during routing."""
        # Mock agent registry to raise exception for most roles but not for general chat
        def mock_list_agents_with_error(role):
            if role == AgentRole.GENERAL_CHAT:
                return [AgentConfig(
                    name="fallback_agent",
                    role=AgentRole.GENERAL_CHAT,
                    system_prompt="Fallback agent",
                    enabled=True,
                    priority=1
                )]
            raise Exception("Registry error")
        
        self.agent_registry.list_agents_by_role.side_effect = mock_list_agents_with_error
        self.agent_registry.get_agent_config.side_effect = lambda name: AgentConfig(
            name="fallback_agent",
            role=AgentRole.GENERAL_CHAT,
            system_prompt="Fallback agent",
            enabled=True,
            priority=1
        ) if name == "fallback_agent" else None
        
        request = MultiAgentRequest(message="Research request")  # This would normally need research agents
        
        # Should not raise exception, should return fallback strategy
        strategy = self.router.route_request(request)
        
        assert strategy is not None
        assert strategy.pattern_type == PatternType.SINGLE
        assert "fallback_agent" in strategy.agents
    
    def test_no_agents_available_error(self):
        """Test handling when no agents are available."""
        # Reset all mocks to return empty
        self.agent_registry.reset_mock()
        self.agent_registry.list_agents_by_role.side_effect = None
        self.agent_registry.list_agents_by_role.return_value = []
        self.agent_registry.list_agent_configs.return_value = []
        self.agent_registry.get_agent_config.side_effect = None
        self.agent_registry.get_agent_config.return_value = None
        
        request = MultiAgentRequest(message="Test request")
        
        # Should raise RuntimeError for no agents
        with pytest.raises(RuntimeError, match="No agents available"):
            self.router.route_request(request)
    
    def test_strategy_validation_failure(self):
        """Test handling of strategy validation failure."""
        # Mock strategy validation to fail
        mock_strategy = Mock(spec=BaseStrategy)
        mock_strategy.validate_strategy.return_value = False
        self.strategy_executor.get_strategy.return_value = mock_strategy
        
        request = MultiAgentRequest(message="Test request")
        strategy = self.router.route_request(request)
        
        # Should fall back to fallback strategy
        assert strategy.coordination_config.get('fallback') is True


if __name__ == "__main__":
    pytest.main([__file__])