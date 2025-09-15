"""
Unit tests for request classifier.
"""

import pytest
from app.core.request_classifier import (
    RequestClassifier, 
    RequestComplexity, 
    RequestDomain,
    ClassificationResult
)
from app.models.multi_agent_models import (
    MultiAgentRequest, 
    PatternType, 
    AgentRole
)


class TestRequestClassifier:
    """Test cases for RequestClassifier."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = RequestClassifier()
    
    def test_simple_greeting_classification(self):
        """Test classification of simple greetings."""
        request = MultiAgentRequest(message="Hello")
        result = self.classifier.classify_request(request)
        
        assert result.complexity == RequestComplexity.SIMPLE
        assert result.domain == RequestDomain.GENERAL
        assert result.suggested_pattern == PatternType.SINGLE
        assert AgentRole.GENERAL_CHAT in result.suggested_agents
    
    def test_research_request_classification(self):
        """Test classification of research requests."""
        request = MultiAgentRequest(message="Find information about climate change impacts")
        result = self.classifier.classify_request(request)
        
        assert result.domain == RequestDomain.RESEARCH
        assert AgentRole.RESEARCH in result.suggested_agents
        assert "find" in result.keywords or "information" in result.keywords
        assert result.confidence > 0.0
    
    def test_analysis_request_classification(self):
        """Test classification of analysis requests."""
        request = MultiAgentRequest(message="Analyze the pros and cons of renewable energy")
        result = self.classifier.classify_request(request)
        
        assert result.domain == RequestDomain.ANALYSIS
        assert AgentRole.ANALYSIS in result.suggested_agents
        assert "analyze" in result.keywords
        assert result.complexity in [RequestComplexity.MODERATE, RequestComplexity.COMPLEX]
    
    def test_technical_request_classification(self):
        """Test classification of technical requests."""
        request = MultiAgentRequest(message="How to debug a Python API error?")
        result = self.classifier.classify_request(request)
        
        assert result.domain == RequestDomain.TECHNICAL
        assert result.complexity in [RequestComplexity.MODERATE, RequestComplexity.COMPLEX]
        assert any(agent in result.suggested_agents for agent in [AgentRole.ANALYSIS, AgentRole.RESEARCH])
    
    def test_multi_step_request_classification(self):
        """Test classification of multi-step requests."""
        request = MultiAgentRequest(
            message="First research market trends, then analyze the data, and finally create a report"
        )
        result = self.classifier.classify_request(request)
        
        assert result.domain == RequestDomain.MULTI_STEP
        assert result.complexity in [RequestComplexity.COMPLEX, RequestComplexity.VERY_COMPLEX]
        assert result.suggested_pattern in [PatternType.WORKFLOW, PatternType.SWARM]
        assert AgentRole.COORDINATION in result.suggested_agents
    
    def test_coordination_request_classification(self):
        """Test classification of coordination requests."""
        request = MultiAgentRequest(
            message="Coordinate multiple tasks to organize a project workflow"
        )
        result = self.classifier.classify_request(request)
        
        assert result.domain == RequestDomain.COORDINATION
        assert AgentRole.COORDINATION in result.suggested_agents
        assert result.suggested_pattern in [PatternType.SWARM, PatternType.GRAPH]
    
    def test_creative_request_classification(self):
        """Test classification of creative requests."""
        request = MultiAgentRequest(message="Write a creative story about space exploration")
        result = self.classifier.classify_request(request)
        
        assert result.domain == RequestDomain.CREATIVE
        assert AgentRole.GENERAL_CHAT in result.suggested_agents
        assert "write" in result.keywords or "creative" in result.keywords
    
    def test_strategy_hint_override(self):
        """Test that strategy hints override default pattern selection."""
        request = MultiAgentRequest(
            message="Simple question",
            strategy_hint="swarm"
        )
        result = self.classifier.classify_request(request)
        
        assert result.suggested_pattern == PatternType.SWARM
    
    def test_coordination_preference_override(self):
        """Test that coordination preference overrides default pattern selection."""
        request = MultiAgentRequest(
            message="Simple question",
            coordination_preference=PatternType.WORKFLOW
        )
        result = self.classifier.classify_request(request)
        
        assert result.suggested_pattern == PatternType.WORKFLOW
    
    def test_complex_multi_domain_request(self):
        """Test classification of complex requests spanning multiple domains."""
        request = MultiAgentRequest(
            message="Research the latest AI developments, analyze their impact on software development, "
                   "and create a technical implementation plan with step-by-step workflow"
        )
        result = self.classifier.classify_request(request)
        
        assert result.complexity == RequestComplexity.VERY_COMPLEX
        assert result.suggested_pattern in [PatternType.WORKFLOW, PatternType.SWARM, PatternType.GRAPH]
        assert len(result.suggested_agents) > 1
        assert AgentRole.COORDINATION in result.suggested_agents
    
    def test_keyword_extraction(self):
        """Test keyword extraction functionality."""
        message = "research and analyze data to find patterns"
        keywords = self.classifier._extract_keywords(message)
        
        expected_keywords = ["research", "analyze", "data", "find", "patterns"]
        for keyword in expected_keywords:
            assert keyword in keywords
    
    def test_domain_classification_confidence(self):
        """Test domain classification confidence scoring."""
        # High confidence research request
        request = MultiAgentRequest(
            message="Research information about machine learning algorithms and find relevant papers"
        )
        result = self.classifier.classify_request(request)
        
        assert result.domain == RequestDomain.RESEARCH
        assert result.confidence > 0.5
    
    def test_complexity_word_count_factor(self):
        """Test that word count affects complexity classification."""
        # Short message
        short_request = MultiAgentRequest(message="What is AI?")
        short_result = self.classifier.classify_request(short_request)
        
        # Long message
        long_request = MultiAgentRequest(
            message="What is artificial intelligence and how does it work in machine learning "
                   "algorithms, what are the different types of neural networks, how do they "
                   "process information, and what are the implications for future technology development?"
        )
        long_result = self.classifier.classify_request(long_request)
        
        # Long message should have higher complexity
        complexity_order = [
            RequestComplexity.SIMPLE,
            RequestComplexity.MODERATE, 
            RequestComplexity.COMPLEX,
            RequestComplexity.VERY_COMPLEX
        ]
        
        short_idx = complexity_order.index(short_result.complexity)
        long_idx = complexity_order.index(long_result.complexity)
        
        assert long_idx >= short_idx
    
    def test_multiple_questions_increase_complexity(self):
        """Test that multiple questions increase complexity."""
        request = MultiAgentRequest(
            message="What is machine learning? How does it work? What are the applications? "
                   "How can I implement it?"
        )
        result = self.classifier.classify_request(request)
        
        assert result.complexity in [RequestComplexity.COMPLEX, RequestComplexity.VERY_COMPLEX]
    
    def test_reasoning_generation(self):
        """Test that reasoning is generated for classifications."""
        request = MultiAgentRequest(message="Analyze market trends for renewable energy")
        result = self.classifier.classify_request(request)
        
        assert result.reasoning is not None
        assert len(result.reasoning) > 0
        assert result.domain.value in result.reasoning
        assert result.complexity.value in result.reasoning
    
    def test_classification_statistics(self):
        """Test classification statistics method."""
        stats = self.classifier.get_classification_statistics()
        
        assert "supported_domains" in stats
        assert "complexity_levels" in stats
        assert "supported_patterns" in stats
        assert "domain_keywords_count" in stats
        
        assert len(stats["supported_domains"]) > 0
        assert len(stats["complexity_levels"]) == 4  # Four complexity levels
        assert len(stats["supported_patterns"]) > 0
    
    def test_edge_case_empty_message(self):
        """Test classification of empty or whitespace-only messages."""
        request = MultiAgentRequest(message="   ")
        result = self.classifier.classify_request(request)
        
        assert result.complexity == RequestComplexity.SIMPLE
        assert result.domain == RequestDomain.GENERAL
        assert result.suggested_pattern == PatternType.SINGLE
    
    def test_edge_case_very_long_message(self):
        """Test classification of very long messages."""
        long_message = " ".join(["word"] * 200)  # 200 words
        request = MultiAgentRequest(message=long_message)
        result = self.classifier.classify_request(request)
        
        assert result.complexity in [RequestComplexity.COMPLEX, RequestComplexity.VERY_COMPLEX]
    
    def test_pattern_suggestions_consistency(self):
        """Test that pattern suggestions are consistent with complexity and domain."""
        test_cases = [
            (RequestComplexity.SIMPLE, RequestDomain.GENERAL, PatternType.SINGLE),
            (RequestComplexity.VERY_COMPLEX, RequestDomain.MULTI_STEP, PatternType.WORKFLOW),
            (RequestComplexity.COMPLEX, RequestDomain.COORDINATION, PatternType.SWARM)
        ]
        
        for complexity, domain, expected_pattern in test_cases:
            # Create a mock request that would result in the desired classification
            request = MultiAgentRequest(message="test message")
            
            # Test the pattern suggestion logic directly
            suggested_pattern = self.classifier._suggest_pattern(complexity, domain, request)
            
            # For simple requests, should always be single
            if complexity == RequestComplexity.SIMPLE:
                assert suggested_pattern == PatternType.SINGLE
            # For multi-step very complex, should be workflow
            elif complexity == RequestComplexity.VERY_COMPLEX and domain == RequestDomain.MULTI_STEP:
                assert suggested_pattern == PatternType.WORKFLOW


class TestClassificationResult:
    """Test cases for ClassificationResult dataclass."""
    
    def test_classification_result_creation(self):
        """Test creation of ClassificationResult."""
        result = ClassificationResult(
            complexity=RequestComplexity.MODERATE,
            domain=RequestDomain.RESEARCH,
            confidence=0.8,
            keywords=["research", "find"],
            suggested_agents=[AgentRole.RESEARCH],
            suggested_pattern=PatternType.SINGLE,
            reasoning="Test reasoning"
        )
        
        assert result.complexity == RequestComplexity.MODERATE
        assert result.domain == RequestDomain.RESEARCH
        assert result.confidence == 0.8
        assert "research" in result.keywords
        assert AgentRole.RESEARCH in result.suggested_agents
        assert result.suggested_pattern == PatternType.SINGLE
        assert result.reasoning == "Test reasoning"


if __name__ == "__main__":
    pytest.main([__file__])