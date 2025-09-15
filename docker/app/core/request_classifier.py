"""
Request classifier for determining agent strategies.

This module provides functionality to analyze incoming requests and classify them
to determine the most appropriate agent strategy and routing decisions.
"""

import re
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from app.models.multi_agent_models import (
    PatternType, 
    AgentRole, 
    MultiAgentRequest,
    AgentStrategy,
    ExecutionParams
)


class RequestComplexity(str, Enum):
    """Enumeration of request complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class RequestDomain(str, Enum):
    """Enumeration of request domains."""
    GENERAL = "general"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    TECHNICAL = "technical"
    CREATIVE = "creative"
    COORDINATION = "coordination"
    MULTI_STEP = "multi_step"


@dataclass
class ClassificationResult:
    """Result of request classification."""
    complexity: RequestComplexity
    domain: RequestDomain
    confidence: float  # 0.0 to 1.0
    keywords: List[str]
    suggested_agents: List[AgentRole]
    suggested_pattern: PatternType
    reasoning: str


class RequestClassifier:
    """
    Classifier for analyzing requests and determining appropriate agent strategies.
    
    This class analyzes incoming requests to determine their complexity, domain,
    and the most appropriate agent routing strategy.
    """
    
    def __init__(self):
        """Initialize the request classifier."""
        self._initialize_classification_rules()
    
    def _initialize_classification_rules(self):
        """Initialize classification rules and patterns."""
        
        # Keywords for different domains
        self.domain_keywords = {
            RequestDomain.RESEARCH: {
                'keywords': [
                    'research', 'find', 'search', 'investigate', 'explore', 'discover',
                    'information', 'data', 'facts', 'evidence', 'sources', 'references',
                    'study', 'survey', 'review', 'literature', 'papers', 'articles',
                    'what is', 'who is', 'when did', 'where is', 'how many', 'statistics'
                ],
                'patterns': [
                    r'\b(find|search|look up|research)\b.*\b(information|data|facts)\b',
                    r'\bwhat (is|are|was|were)\b',
                    r'\b(who|when|where|how many|statistics)\b',
                    r'\b(latest|recent|current)\b.*\b(news|updates|developments)\b'
                ]
            },
            RequestDomain.ANALYSIS: {
                'keywords': [
                    'analyze', 'analysis', 'compare', 'evaluate', 'assess', 'examine',
                    'interpret', 'explain', 'understand', 'breakdown', 'summarize',
                    'trends', 'patterns', 'insights', 'conclusions', 'implications',
                    'pros and cons', 'advantages', 'disadvantages', 'benefits', 'risks'
                ],
                'patterns': [
                    r'\b(analyze|compare|evaluate|assess)\b',
                    r'\b(pros and cons|advantages and disadvantages)\b',
                    r'\b(what does.*mean|explain.*to me)\b',
                    r'\b(trends|patterns|insights)\b'
                ]
            },
            RequestDomain.TECHNICAL: {
                'keywords': [
                    'code', 'programming', 'software', 'development', 'algorithm',
                    'debug', 'error', 'bug', 'implementation', 'architecture',
                    'database', 'api', 'framework', 'library', 'deployment',
                    'configuration', 'setup', 'install', 'troubleshoot'
                ],
                'patterns': [
                    r'\b(code|programming|software|development)\b',
                    r'\b(debug|error|bug|troubleshoot)\b',
                    r'\b(api|database|framework|library)\b',
                    r'\b(how to (implement|setup|install|configure))\b'
                ]
            },
            RequestDomain.CREATIVE: {
                'keywords': [
                    'create', 'generate', 'write', 'compose', 'design', 'brainstorm',
                    'ideas', 'creative', 'story', 'poem', 'article', 'content',
                    'marketing', 'copy', 'script', 'proposal', 'plan'
                ],
                'patterns': [
                    r'\b(create|generate|write|compose)\b',
                    r'\b(brainstorm|ideas|creative)\b',
                    r'\b(story|poem|article|content)\b'
                ]
            },
            RequestDomain.COORDINATION: {
                'keywords': [
                    'coordinate', 'manage', 'organize', 'plan', 'schedule',
                    'workflow', 'process', 'steps', 'sequence', 'order',
                    'multiple tasks', 'several things', 'various aspects'
                ],
                'patterns': [
                    r'\b(coordinate|manage|organize|plan)\b',
                    r'\b(workflow|process|steps|sequence)\b',
                    r'\b(multiple|several|various)\b.*\b(tasks|things|aspects)\b'
                ]
            },
            RequestDomain.MULTI_STEP: {
                'keywords': [
                    'first', 'then', 'next', 'finally', 'step by step',
                    'process', 'procedure', 'workflow', 'sequence',
                    'multiple', 'several', 'various', 'different'
                ],
                'patterns': [
                    r'\b(first.*then|step by step|step-by-step)\b',
                    r'\b(process|procedure|workflow|sequence)\b',
                    r'\b(multiple|several|various|different)\b.*\b(steps|phases|stages)\b'
                ]
            }
        }
        
        # Complexity indicators
        self.complexity_indicators = {
            RequestComplexity.SIMPLE: {
                'max_words': 20,
                'simple_patterns': [
                    r'^\w+\?$',  # Single word questions
                    r'^(what|who|when|where|how) is \w+\?$',  # Simple what/who/when questions
                    r'^(yes|no|maybe|hello|hi|thanks|thank you)\.?$'  # Simple responses
                ],
                'keywords': ['hello', 'hi', 'thanks', 'yes', 'no']
            },
            RequestComplexity.MODERATE: {
                'max_words': 50,
                'indicators': [
                    'single question',
                    'one main topic',
                    'straightforward request'
                ]
            },
            RequestComplexity.COMPLEX: {
                'max_words': 150,
                'indicators': [
                    'multiple questions',
                    'requires analysis',
                    'comparison needed',
                    'multi-part answer'
                ]
            },
            RequestComplexity.VERY_COMPLEX: {
                'indicators': [
                    'multiple domains',
                    'requires coordination',
                    'multi-step process',
                    'research and analysis',
                    'complex workflow'
                ]
            }
        }
    
    def classify_request(self, request: MultiAgentRequest) -> ClassificationResult:
        """
        Classify a request to determine appropriate agent strategy.
        
        Args:
            request: The multi-agent request to classify
            
        Returns:
            ClassificationResult containing classification details
        """
        message = request.message.lower().strip()
        
        # Extract keywords
        keywords = self._extract_keywords(message)
        
        # Determine domain
        domain, domain_confidence = self._classify_domain(message, keywords)
        
        # Determine complexity
        complexity, complexity_confidence = self._classify_complexity(message, request)
        
        # Calculate overall confidence
        confidence = (domain_confidence + complexity_confidence) / 2
        
        # Suggest agents based on domain and complexity
        suggested_agents = self._suggest_agents(domain, complexity)
        
        # Suggest pattern based on complexity and domain
        suggested_pattern = self._suggest_pattern(complexity, domain, request)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(domain, complexity, keywords, suggested_pattern)
        
        return ClassificationResult(
            complexity=complexity,
            domain=domain,
            confidence=confidence,
            keywords=keywords,
            suggested_agents=suggested_agents,
            suggested_pattern=suggested_pattern,
            reasoning=reasoning
        )
    
    def _extract_keywords(self, message: str) -> List[str]:
        """Extract relevant keywords from the message."""
        keywords = []
        
        # Extract words (remove punctuation and convert to lowercase)
        words = re.findall(r'\b\w+\b', message.lower())
        
        # Check against domain keywords
        for domain, domain_data in self.domain_keywords.items():
            for keyword in domain_data['keywords']:
                if keyword in message:
                    keywords.append(keyword)
        
        # Remove duplicates and return
        return list(set(keywords))
    
    def _classify_domain(self, message: str, keywords: List[str]) -> Tuple[RequestDomain, float]:
        """Classify the domain of the request."""
        domain_scores = {}
        
        # Score based on keywords
        for domain, domain_data in self.domain_keywords.items():
            score = 0
            
            # Keyword matching
            for keyword in domain_data['keywords']:
                if keyword in message:
                    score += 1
            
            # Pattern matching
            for pattern in domain_data.get('patterns', []):
                if re.search(pattern, message, re.IGNORECASE):
                    score += 2  # Patterns are weighted higher
            
            if score > 0:
                domain_scores[domain] = score
        
        # Special handling for multi-step and coordination domains
        # Check for multi-step indicators
        multi_step_indicators = ['first', 'then', 'next', 'finally', 'step by step', 'step-by-step']
        multi_step_count = sum(1 for indicator in multi_step_indicators if indicator in message.lower())
        if multi_step_count >= 2:  # At least 2 step indicators
            domain_scores[RequestDomain.MULTI_STEP] = domain_scores.get(RequestDomain.MULTI_STEP, 0) + 3
        
        # Check for coordination indicators
        coordination_indicators = ['coordinate', 'manage', 'organize', 'multiple tasks', 'workflow']
        coordination_count = sum(1 for indicator in coordination_indicators if indicator in message.lower())
        if coordination_count >= 2:  # At least 2 coordination indicators
            domain_scores[RequestDomain.COORDINATION] = domain_scores.get(RequestDomain.COORDINATION, 0) + 3
        
        # If no specific domain detected, default to general
        if not domain_scores:
            return RequestDomain.GENERAL, 0.5
        
        # Find domain with highest score
        best_domain = max(domain_scores, key=domain_scores.get)
        max_score = domain_scores[best_domain]
        
        # Calculate confidence (normalize by maximum possible score)
        max_possible_score = len(self.domain_keywords[best_domain]['keywords']) + \
                           len(self.domain_keywords[best_domain].get('patterns', [])) * 2
        confidence = min(max_score / max_possible_score, 1.0)
        
        return best_domain, confidence
    
    def _classify_complexity(self, message: str, request: MultiAgentRequest) -> Tuple[RequestComplexity, float]:
        """Classify the complexity of the request."""
        word_count = len(message.split())
        
        # Check for simple patterns first, but only if there are no multiple questions
        question_count = len(re.findall(r'\?', message))
        if question_count <= 1:  # Only apply simple patterns for single questions
            simple_indicators = self.complexity_indicators[RequestComplexity.SIMPLE]
            for pattern in simple_indicators.get('simple_patterns', []):
                if re.match(pattern, message, re.IGNORECASE):
                    return RequestComplexity.SIMPLE, 0.9
            
            # Check for simple keywords only if it's a short message
            if word_count <= 10:
                for keyword in simple_indicators.get('keywords', []):
                    if keyword in message:
                        return RequestComplexity.SIMPLE, 0.8
        
        # Complexity indicators
        complexity_signals = {
            'multiple_questions': len(re.findall(r'\?', message)),
            'coordination_words': len(re.findall(r'\b(and|also|then|next|after|before)\b', message)),
            'analysis_words': len(re.findall(r'\b(analyze|compare|evaluate|explain|why|how)\b', message)),
            'multi_step_words': len(re.findall(r'\b(first|second|then|next|finally|step)\b', message))
        }
        
        # Calculate complexity score
        complexity_score = 0
        complexity_score += min(word_count / 15, 4)  # Word count factor (max 4 points, lower threshold)
        complexity_score += complexity_signals['multiple_questions'] * 1.0  # Increased weight for questions
        complexity_score += complexity_signals['coordination_words'] * 0.5
        complexity_score += complexity_signals['analysis_words'] * 0.6
        complexity_score += complexity_signals['multi_step_words'] * 1.0  # Increased weight for multi-step
        
        # Check for strategy hints that indicate complexity
        if request.strategy_hint:
            if request.strategy_hint.lower() in ['swarm', 'workflow', 'graph']:
                complexity_score += 2
        
        # Boost complexity for coordination and multi-step requests
        coordination_boost = sum(1 for word in ['coordinate', 'manage', 'organize', 'multiple tasks', 'workflow'] 
                               if word in message.lower())
        if coordination_boost >= 2:
            complexity_score += 1.5
        
        # Determine complexity level
        if complexity_score <= 1.0:
            return RequestComplexity.SIMPLE, 0.8
        elif complexity_score <= 2.0:
            return RequestComplexity.MODERATE, 0.7
        elif complexity_score <= 3.5:
            return RequestComplexity.COMPLEX, 0.7
        else:
            return RequestComplexity.VERY_COMPLEX, 0.8
    
    def _suggest_agents(self, domain: RequestDomain, complexity: RequestComplexity) -> List[AgentRole]:
        """Suggest appropriate agents based on domain and complexity."""
        agents = []
        
        # Domain-based agent suggestions
        domain_agent_mapping = {
            RequestDomain.RESEARCH: [AgentRole.RESEARCH],
            RequestDomain.ANALYSIS: [AgentRole.ANALYSIS],
            RequestDomain.TECHNICAL: [AgentRole.ANALYSIS, AgentRole.RESEARCH],
            RequestDomain.CREATIVE: [AgentRole.GENERAL_CHAT],
            RequestDomain.COORDINATION: [AgentRole.COORDINATION],
            RequestDomain.MULTI_STEP: [AgentRole.COORDINATION],
            RequestDomain.GENERAL: [AgentRole.GENERAL_CHAT]
        }
        
        agents.extend(domain_agent_mapping.get(domain, [AgentRole.GENERAL_CHAT]))
        
        # Complexity-based additions
        if complexity in [RequestComplexity.COMPLEX, RequestComplexity.VERY_COMPLEX]:
            if AgentRole.COORDINATION not in agents:
                agents.append(AgentRole.COORDINATION)
        
        if complexity == RequestComplexity.VERY_COMPLEX:
            # Very complex requests might need multiple agent types
            if domain == RequestDomain.RESEARCH and AgentRole.ANALYSIS not in agents:
                agents.append(AgentRole.ANALYSIS)
            elif domain == RequestDomain.ANALYSIS and AgentRole.RESEARCH not in agents:
                agents.append(AgentRole.RESEARCH)
            elif domain == RequestDomain.COORDINATION:
                # Coordination requests often need research and analysis agents too
                if AgentRole.RESEARCH not in agents:
                    agents.append(AgentRole.RESEARCH)
                if AgentRole.ANALYSIS not in agents:
                    agents.append(AgentRole.ANALYSIS)
            elif domain == RequestDomain.MULTI_STEP:
                # Multi-step requests often need research and analysis agents too
                if AgentRole.RESEARCH not in agents:
                    agents.append(AgentRole.RESEARCH)
                if AgentRole.ANALYSIS not in agents:
                    agents.append(AgentRole.ANALYSIS)
        
        return agents
    
    def _suggest_pattern(self, complexity: RequestComplexity, domain: RequestDomain, request: MultiAgentRequest) -> PatternType:
        """Suggest appropriate orchestration pattern."""
        
        # Check for explicit strategy hint
        if request.strategy_hint:
            hint_lower = request.strategy_hint.lower()
            pattern_mapping = {
                'single': PatternType.SINGLE,
                'swarm': PatternType.SWARM,
                'workflow': PatternType.WORKFLOW,
                'graph': PatternType.GRAPH,
                'a2a': PatternType.A2A
            }
            if hint_lower in pattern_mapping:
                return pattern_mapping[hint_lower]
        
        # Check coordination preference
        if request.coordination_preference:
            return request.coordination_preference
        
        # Pattern selection based on complexity and domain
        if complexity == RequestComplexity.SIMPLE:
            return PatternType.SINGLE
        
        elif complexity == RequestComplexity.MODERATE:
            if domain in [RequestDomain.RESEARCH, RequestDomain.ANALYSIS]:
                return PatternType.SINGLE
            else:
                return PatternType.SWARM
        
        elif complexity == RequestComplexity.COMPLEX:
            if domain == RequestDomain.MULTI_STEP:
                return PatternType.WORKFLOW
            elif domain == RequestDomain.COORDINATION:
                return PatternType.SWARM
            else:
                return PatternType.SWARM
        
        else:  # VERY_COMPLEX
            if domain == RequestDomain.MULTI_STEP:
                return PatternType.WORKFLOW
            elif domain == RequestDomain.COORDINATION:
                return PatternType.GRAPH
            else:
                return PatternType.SWARM
    
    def _generate_reasoning(self, domain: RequestDomain, complexity: RequestComplexity, 
                          keywords: List[str], pattern: PatternType) -> str:
        """Generate human-readable reasoning for the classification."""
        reasoning_parts = []
        
        # Domain reasoning
        reasoning_parts.append(f"Classified as {domain.value} domain")
        if keywords:
            reasoning_parts.append(f"based on keywords: {', '.join(keywords[:5])}")
        
        # Complexity reasoning
        reasoning_parts.append(f"Complexity level: {complexity.value}")
        
        # Pattern reasoning
        pattern_reasoning = {
            PatternType.SINGLE: "Single agent sufficient for this request",
            PatternType.SWARM: "Multiple agents can collaborate effectively",
            PatternType.WORKFLOW: "Sequential processing required",
            PatternType.GRAPH: "Complex decision tree needed",
            PatternType.A2A: "Agent-to-agent communication required"
        }
        reasoning_parts.append(pattern_reasoning.get(pattern, f"Using {pattern.value} pattern"))
        
        return ". ".join(reasoning_parts) + "."
    
    def get_classification_statistics(self) -> Dict[str, Any]:
        """Get statistics about classification patterns."""
        return {
            "supported_domains": [domain.value for domain in RequestDomain],
            "complexity_levels": [complexity.value for complexity in RequestComplexity],
            "supported_patterns": [pattern.value for pattern in PatternType],
            "domain_keywords_count": {
                domain.value: len(data['keywords']) 
                for domain, data in self.domain_keywords.items()
            }
        }