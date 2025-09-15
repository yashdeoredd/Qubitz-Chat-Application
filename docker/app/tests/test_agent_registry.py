"""
Unit tests for agent registry.
"""

import pytest
from unittest.mock import Mock, patch

from app.models.multi_agent_models import (
    AgentConfig,
    AgentRole,
    CollaborationConfig,
    PatternType
)
from app.core.agent_registry import AgentRegistry


class TestAgentRegistry:
    """Test AgentRegistry class."""
    
    def test_initialization(self):
        """Test registry initialization."""
        registry = AgentRegistry()
        assert len(registry._agent_configs) == 0
        assert len(registry._agent_instances) == 0
    
    def test_register_agent_config(self):
        """Test registering agent configuration."""
        registry = AgentRegistry()
        config = AgentConfig(
            name="test_agent",
            role=AgentRole.RESEARCH,
            system_prompt="Test prompt"
        )
        
        registry.register_agent_config(config)
        
        assert "test_agent" in registry._agent_configs
        assert registry._agent_configs["test_agent"] == config
    
    def test_register_agent_config_clears_instance(self):
        """Test that registering config clears cached instance."""
        registry = AgentRegistry()
        config = AgentConfig(
            name="test_agent",
            role=AgentRole.RESEARCH,
            system_prompt="Test prompt"
        )
        
        # Add a mock instance
        mock_agent = Mock()
        registry._agent_instances["test_agent"] = mock_agent
        
        # Register new config
        registry.register_agent_config(config)
        
        # Instance should be cleared
        assert "test_agent" not in registry._agent_instances
    
    def test_get_agent_config(self):
        """Test getting agent configuration."""
        registry = AgentRegistry()
        config = AgentConfig(
            name="test_agent",
            role=AgentRole.RESEARCH,
            system_prompt="Test prompt"
        )
        registry.register_agent_config(config)
        
        retrieved_config = registry.get_agent_config("test_agent")
        assert retrieved_config == config
        
        # Test non-existent config
        assert registry.get_agent_config("nonexistent") is None
    
    def test_list_agent_configs(self):
        """Test listing all agent configurations."""
        registry = AgentRegistry()
        config1 = AgentConfig(
            name="agent1",
            role=AgentRole.RESEARCH,
            system_prompt="Prompt 1"
        )
        config2 = AgentConfig(
            name="agent2",
            role=AgentRole.ANALYSIS,
            system_prompt="Prompt 2"
        )
        
        registry.register_agent_config(config1)
        registry.register_agent_config(config2)
        
        configs = registry.list_agent_configs()
        assert len(configs) == 2
        assert config1 in configs
        assert config2 in configs
    
    def test_list_agents_by_role(self):
        """Test listing agents by role."""
        registry = AgentRegistry()
        research_config = AgentConfig(
            name="research_agent",
            role=AgentRole.RESEARCH,
            system_prompt="Research prompt"
        )
        analysis_config = AgentConfig(
            name="analysis_agent",
            role=AgentRole.ANALYSIS,
            system_prompt="Analysis prompt"
        )
        research_config2 = AgentConfig(
            name="research_agent2",
            role=AgentRole.RESEARCH,
            system_prompt="Research prompt 2"
        )
        
        registry.register_agent_config(research_config)
        registry.register_agent_config(analysis_config)
        registry.register_agent_config(research_config2)
        
        research_agents = registry.list_agents_by_role(AgentRole.RESEARCH)
        analysis_agents = registry.list_agents_by_role(AgentRole.ANALYSIS)
        
        assert len(research_agents) == 2
        assert research_config in research_agents
        assert research_config2 in research_agents
        
        assert len(analysis_agents) == 1
        assert analysis_config in analysis_agents
    
    @patch('app.core.agent_registry.Agent')
    def test_get_agent_instance_creates_new(self, mock_agent_class):
        """Test getting agent instance creates new instance."""
        registry = AgentRegistry()
        config = AgentConfig(
            name="test_agent",
            role=AgentRole.RESEARCH,
            system_prompt="Test prompt",
            tools=["tool1", "tool2"]
        )
        registry.register_agent_config(config)
        
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        
        instance = registry.get_agent_instance("test_agent")
        
        assert instance == mock_agent
        assert "test_agent" in registry._agent_instances
        mock_agent_class.assert_called_once_with(
            system_prompt="Test prompt",
            tools=[]  # Tools resolution is simplified in current implementation
        )
    
    @patch('app.core.agent_registry.Agent')
    def test_get_agent_instance_returns_cached(self, mock_agent_class):
        """Test getting agent instance returns cached instance."""
        registry = AgentRegistry()
        config = AgentConfig(
            name="test_agent",
            role=AgentRole.RESEARCH,
            system_prompt="Test prompt"
        )
        registry.register_agent_config(config)
        
        mock_agent = Mock()
        registry._agent_instances["test_agent"] = mock_agent
        
        instance = registry.get_agent_instance("test_agent")
        
        assert instance == mock_agent
        # Should not create new instance
        mock_agent_class.assert_not_called()
    
    def test_get_agent_instance_not_found(self):
        """Test getting non-existent agent instance."""
        registry = AgentRegistry()
        instance = registry.get_agent_instance("nonexistent")
        assert instance is None
    
    def test_get_agent_instance_disabled(self):
        """Test getting disabled agent instance."""
        registry = AgentRegistry()
        config = AgentConfig(
            name="test_agent",
            role=AgentRole.RESEARCH,
            system_prompt="Test prompt",
            enabled=False
        )
        registry.register_agent_config(config)
        
        instance = registry.get_agent_instance("test_agent")
        assert instance is None
    
    def test_remove_agent_config(self):
        """Test removing agent configuration."""
        registry = AgentRegistry()
        config = AgentConfig(
            name="test_agent",
            role=AgentRole.RESEARCH,
            system_prompt="Test prompt"
        )
        registry.register_agent_config(config)
        
        # Add cached instance
        mock_agent = Mock()
        registry._agent_instances["test_agent"] = mock_agent
        
        result = registry.remove_agent_config("test_agent")
        
        assert result is True
        assert "test_agent" not in registry._agent_configs
        assert "test_agent" not in registry._agent_instances
        
        # Test removing non-existent config
        result = registry.remove_agent_config("nonexistent")
        assert result is False
    
    def test_enable_agent(self):
        """Test enabling agent."""
        registry = AgentRegistry()
        config = AgentConfig(
            name="test_agent",
            role=AgentRole.RESEARCH,
            system_prompt="Test prompt",
            enabled=False
        )
        registry.register_agent_config(config)
        
        result = registry.enable_agent("test_agent")
        
        assert result is True
        assert config.enabled is True
        
        # Test enabling non-existent agent
        result = registry.enable_agent("nonexistent")
        assert result is False
    
    def test_disable_agent(self):
        """Test disabling agent."""
        registry = AgentRegistry()
        config = AgentConfig(
            name="test_agent",
            role=AgentRole.RESEARCH,
            system_prompt="Test prompt",
            enabled=True
        )
        registry.register_agent_config(config)
        
        # Add cached instance
        mock_agent = Mock()
        registry._agent_instances["test_agent"] = mock_agent
        
        result = registry.disable_agent("test_agent")
        
        assert result is True
        assert config.enabled is False
        assert "test_agent" not in registry._agent_instances
        
        # Test disabling non-existent agent
        result = registry.disable_agent("nonexistent")
        assert result is False
    
    def test_get_registry_statistics(self):
        """Test getting registry statistics."""
        registry = AgentRegistry()
        
        # Add some configurations
        config1 = AgentConfig(
            name="research_agent",
            role=AgentRole.RESEARCH,
            system_prompt="Research prompt",
            enabled=True
        )
        config2 = AgentConfig(
            name="analysis_agent",
            role=AgentRole.ANALYSIS,
            system_prompt="Analysis prompt",
            enabled=True
        )
        config3 = AgentConfig(
            name="disabled_agent",
            role=AgentRole.GENERAL_CHAT,
            system_prompt="Disabled prompt",
            enabled=False
        )
        
        registry.register_agent_config(config1)
        registry.register_agent_config(config2)
        registry.register_agent_config(config3)
        
        # Add cached instance
        mock_agent = Mock()
        registry._agent_instances["research_agent"] = mock_agent
        
        stats = registry.get_registry_statistics()
        
        assert stats["total_configurations"] == 3
        assert stats["enabled_configurations"] == 2
        assert stats["cached_instances"] == 1
        assert stats["configurations_by_role"]["research"] == 1
        assert stats["configurations_by_role"]["analysis"] == 1
        assert stats["configurations_by_role"]["general_chat"] == 1