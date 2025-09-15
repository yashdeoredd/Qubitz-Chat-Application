"""
Unit tests for specialized agent configurations.
"""

import pytest
from unittest.mock import Mock, patch
from strands import Agent
from strands.models import BedrockModel

from app.config.agent_config import (
    create_research_agent,
    create_analysis_agent,
    create_coordination_agent,
    create_general_chat_agent
)
from app.config.prompts import (
    RESEARCH_SPECIALIST_PROMPT,
    ANALYSIS_SPECIALIST_PROMPT,
    COORDINATION_SPECIALIST_PROMPT,
    GENERAL_CHAT_PROMPT
)


class TestResearchAgent:
    """Test cases for research specialist agent configuration."""
    
    @patch('app.config.agent_config.create_bedrock_model')
    def test_create_research_agent_configuration(self, mock_create_model):
        """Test that research agent is created with correct configuration."""
        # Mock the bedrock model
        mock_model = Mock(spec=BedrockModel)
        mock_create_model.return_value = mock_model
        
        # Create research agent
        agent = create_research_agent()
        
        # Verify agent is created
        assert isinstance(agent, Agent)
        assert agent.system_prompt == RESEARCH_SPECIALIST_PROMPT
        
        # Verify model creation was called
        mock_create_model.assert_called_once()
    
    @patch('app.config.agent_config.create_bedrock_model')
    def test_research_agent_has_correct_tools(self, mock_create_model):
        """Test that research agent has the correct tools configured."""
        # Mock the bedrock model
        mock_model = Mock(spec=BedrockModel)
        mock_create_model.return_value = mock_model
        
        # Create research agent
        agent = create_research_agent()
        
        # Verify agent has expected tools
        tool_names = agent.tool_names
        assert 'http_request' in tool_names
        assert 'ready_to_summarize' in tool_names
    
    @patch('app.config.agent_config.create_bedrock_model')
    def test_research_agent_system_prompt_content(self, mock_create_model):
        """Test that research agent has research-specific system prompt content."""
        # Mock the bedrock model
        mock_model = Mock(spec=BedrockModel)
        mock_create_model.return_value = mock_model
        
        # Create research agent
        agent = create_research_agent()
        
        # Verify system prompt contains research-specific content
        prompt = agent.system_prompt
        assert "research agent" in prompt.lower()
        assert "information gathering" in prompt.lower()
        assert "sources" in prompt.lower()
        assert "comprehensive research" in prompt.lower()


class TestAnalysisAgent:
    """Test cases for analysis specialist agent configuration."""
    
    @patch('app.config.agent_config.create_bedrock_model')
    def test_create_analysis_agent_configuration(self, mock_create_model):
        """Test that analysis agent is created with correct configuration."""
        # Mock the bedrock model
        mock_model = Mock(spec=BedrockModel)
        mock_create_model.return_value = mock_model
        
        # Create analysis agent
        agent = create_analysis_agent()
        
        # Verify agent is created
        assert isinstance(agent, Agent)
        assert agent.system_prompt == ANALYSIS_SPECIALIST_PROMPT
        
        # Verify model creation was called
        mock_create_model.assert_called_once()
    
    @patch('app.config.agent_config.create_bedrock_model')
    def test_analysis_agent_has_correct_tools(self, mock_create_model):
        """Test that analysis agent has the correct tools configured."""
        # Mock the bedrock model
        mock_model = Mock(spec=BedrockModel)
        mock_create_model.return_value = mock_model
        
        # Create analysis agent
        agent = create_analysis_agent()
        
        # Verify agent has expected tools
        tool_names = agent.tool_names
        assert 'http_request' in tool_names
        assert 'ready_to_summarize' in tool_names
    
    @patch('app.config.agent_config.create_bedrock_model')
    def test_analysis_agent_system_prompt_content(self, mock_create_model):
        """Test that analysis agent has analysis-specific system prompt content."""
        # Mock the bedrock model
        mock_model = Mock(spec=BedrockModel)
        mock_create_model.return_value = mock_model
        
        # Create analysis agent
        agent = create_analysis_agent()
        
        # Verify system prompt contains analysis-specific content
        prompt = agent.system_prompt
        assert "analysis agent" in prompt.lower()
        assert "data analysis" in prompt.lower()
        assert "patterns" in prompt.lower()
        assert "insights" in prompt.lower()


class TestCoordinationAgent:
    """Test cases for coordination specialist agent configuration."""
    
    @patch('app.config.agent_config.create_bedrock_model')
    def test_create_coordination_agent_configuration(self, mock_create_model):
        """Test that coordination agent is created with correct configuration."""
        # Mock the bedrock model
        mock_model = Mock(spec=BedrockModel)
        mock_create_model.return_value = mock_model
        
        # Create coordination agent
        agent = create_coordination_agent()
        
        # Verify agent is created
        assert isinstance(agent, Agent)
        assert agent.system_prompt == COORDINATION_SPECIALIST_PROMPT
        
        # Verify model creation was called
        mock_create_model.assert_called_once()
    
    @patch('app.config.agent_config.create_bedrock_model')
    def test_coordination_agent_has_multi_agent_tools(self, mock_create_model):
        """Test that coordination agent has multi-agent orchestration tools."""
        # Mock the bedrock model
        mock_model = Mock(spec=BedrockModel)
        mock_create_model.return_value = mock_model
        
        # Create coordination agent
        agent = create_coordination_agent()
        
        # Verify agent has expected multi-agent tools
        tool_names = agent.tool_names
        assert 'swarm' in tool_names
        assert 'workflow' in tool_names
        assert 'graph' in tool_names
        assert 'http_request' in tool_names
        assert 'ready_to_summarize' in tool_names
    
    @patch('app.config.agent_config.create_bedrock_model')
    def test_coordination_agent_system_prompt_content(self, mock_create_model):
        """Test that coordination agent has coordination-specific system prompt content."""
        # Mock the bedrock model
        mock_model = Mock(spec=BedrockModel)
        mock_create_model.return_value = mock_model
        
        # Create coordination agent
        agent = create_coordination_agent()
        
        # Verify system prompt contains coordination-specific content
        prompt = agent.system_prompt
        assert "coordination agent" in prompt.lower()
        assert "orchestrating" in prompt.lower()
        assert "multi-agent" in prompt.lower()
        assert "swarm" in prompt.lower()
        assert "workflow" in prompt.lower()
        assert "graph" in prompt.lower()


class TestGeneralChatAgent:
    """Test cases for general chat agent configuration."""
    
    @patch('app.config.agent_config.create_bedrock_model')
    def test_create_general_chat_agent_configuration(self, mock_create_model):
        """Test that general chat agent is created with correct configuration."""
        # Mock the bedrock model
        mock_model = Mock(spec=BedrockModel)
        mock_create_model.return_value = mock_model
        
        # Create general chat agent
        agent = create_general_chat_agent()
        
        # Verify agent is created
        assert isinstance(agent, Agent)
        assert agent.system_prompt == GENERAL_CHAT_PROMPT
        
        # Verify model creation was called
        mock_create_model.assert_called_once()
    
    @patch('app.config.agent_config.create_bedrock_model')
    def test_general_chat_agent_has_minimal_tools(self, mock_create_model):
        """Test that general chat agent has minimal tool set for simple conversations."""
        # Mock the bedrock model
        mock_model = Mock(spec=BedrockModel)
        mock_create_model.return_value = mock_model
        
        # Create general chat agent
        agent = create_general_chat_agent()
        
        # Verify agent has minimal tools (no http_request for simple chat)
        tool_names = agent.tool_names
        assert 'ready_to_summarize' in tool_names
        # General chat agent should not have http_request for simple conversations
        assert 'http_request' not in tool_names
    
    @patch('app.config.agent_config.create_bedrock_model')
    def test_general_chat_agent_system_prompt_content(self, mock_create_model):
        """Test that general chat agent has conversational system prompt content."""
        # Mock the bedrock model
        mock_model = Mock(spec=BedrockModel)
        mock_create_model.return_value = mock_model
        
        # Create general chat agent
        agent = create_general_chat_agent()
        
        # Verify system prompt contains conversational content
        prompt = agent.system_prompt
        assert "conversational agent" in prompt.lower()
        assert "friendly" in prompt.lower()
        assert "general-purpose" in prompt.lower()
        assert "everyday queries" in prompt.lower()


class TestAgentSpecialization:
    """Test cases for agent specialization and differentiation."""
    
    @patch('app.config.agent_config.create_bedrock_model')
    def test_agents_have_different_system_prompts(self, mock_create_model):
        """Test that each specialized agent has a unique system prompt."""
        # Mock the bedrock model
        mock_model = Mock(spec=BedrockModel)
        mock_create_model.return_value = mock_model
        
        # Create all agent types
        research_agent = create_research_agent()
        analysis_agent = create_analysis_agent()
        coordination_agent = create_coordination_agent()
        general_agent = create_general_chat_agent()
        
        # Verify all prompts are different
        prompts = [
            research_agent.system_prompt,
            analysis_agent.system_prompt,
            coordination_agent.system_prompt,
            general_agent.system_prompt
        ]
        
        # Check that all prompts are unique
        assert len(set(prompts)) == 4
    
    @patch('app.config.agent_config.create_bedrock_model')
    def test_agents_have_appropriate_tool_configurations(self, mock_create_model):
        """Test that each agent has tools appropriate for its specialization."""
        # Mock the bedrock model
        mock_model = Mock(spec=BedrockModel)
        mock_create_model.return_value = mock_model
        
        # Create all agent types
        research_agent = create_research_agent()
        analysis_agent = create_analysis_agent()
        coordination_agent = create_coordination_agent()
        general_agent = create_general_chat_agent()
        
        # Research and analysis agents should have http_request for data gathering
        assert 'http_request' in research_agent.tool_names
        assert 'http_request' in analysis_agent.tool_names
        
        # Coordination agent should have multi-agent tools
        coord_tools = coordination_agent.tool_names
        assert 'swarm' in coord_tools
        assert 'workflow' in coord_tools
        assert 'graph' in coord_tools
        
        # General agent should have minimal tools
        general_tools = general_agent.tool_names
        assert 'ready_to_summarize' in general_tools
        assert len(general_tools) == 1  # Only ready_to_summarize