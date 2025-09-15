"""
Agent router for intelligent request routing and strategy selection.

This module provides the AgentRouter class that analyzes incoming requests,
classifies them, and routes them to the most appropriate agent strategy.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging

from app.models.multi_agent_models import (
    MultiAgentRequest,
    MultiAgentResult,
    AgentStrategy,
    AgentRole,
    PatternType,
    ExecutionParams,
    ExecutionStatus,
    ExecutionMetadata,
    ErrorContext,
    RecoveryAction
)
from app.core.request_classifier import RequestClassifier, ClassificationResult
from app.core.agent_registry import AgentRegistry
from app.core.base_strategy import StrategyExecutor


logger = logging.getLogger(__name__)


class AgentRouter:
    """
    Intelligent router for multi-agent requests.
    
    This class analyzes incoming requests, classifies them for complexity and domain,
    and routes them to the most appropriate agent strategy for execution.
    """
    
    def __init__(self, agent_registry: AgentRegistry, strategy_executor: StrategyExecutor):
        """
        Initialize the agent router.
        
        Args:
            agent_registry: Registry of available agents
            strategy_executor: Executor for different agent strategies
        """
        self.agent_registry = agent_registry
        self.strategy_executor = strategy_executor
        self.classifier = RequestClassifier()
        
        # Routing statistics
        self._routing_history: List[Dict[str, Any]] = []
        self._strategy_performance: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.max_fallback_attempts = 3
        self.enable_adaptive_routing = True
        
    def route_request(self, request: MultiAgentRequest) -> AgentStrategy:
        """
        Route a request to the most appropriate agent strategy.
        
        Args:
            request: The multi-agent request to route
            
        Returns:
            AgentStrategy configured for the request
            
        Raises:
            ValueError: If no suitable agents are available
            RuntimeError: If routing fails
        """
        try:
            # Classify the request
            classification = self.classifier.classify_request(request)
            
            logger.info(f"Request classified - Domain: {classification.domain.value}, "
                       f"Complexity: {classification.complexity.value}, "
                       f"Confidence: {classification.confidence:.2f}")
            
            # Create base strategy from classification
            strategy = self._create_strategy_from_classification(classification, request)
            
            # Apply user preferences and hints
            strategy = self._apply_user_preferences(strategy, request)
            
            # Validate and optimize strategy
            strategy = self._validate_and_optimize_strategy(strategy, request)
            
            # Record routing decision
            self._record_routing_decision(request, classification, strategy)
            
            return strategy
            
        except Exception as e:
            logger.error(f"Failed to route request: {str(e)}")
            # Return fallback strategy
            try:
                return self._create_fallback_strategy(request)
            except Exception as fallback_error:
                logger.error(f"Failed to create fallback strategy: {str(fallback_error)}")
                raise RuntimeError(f"No agents available for request routing: {str(e)}")
    
    def _create_strategy_from_classification(
        self, 
        classification: ClassificationResult, 
        request: MultiAgentRequest
    ) -> AgentStrategy:
        """Create an agent strategy based on classification results."""
        
        # Get available agents for the suggested roles
        suggested_roles = list(classification.suggested_agents)
        
        # Expand agent roles for complex requests
        if classification.complexity.value in ['complex', 'very_complex']:
            if classification.domain.value == 'multi_step' and len(suggested_roles) == 1:
                # Multi-step requests often need multiple agent types
                if AgentRole.RESEARCH not in suggested_roles:
                    suggested_roles.append(AgentRole.RESEARCH)
                if AgentRole.ANALYSIS not in suggested_roles:
                    suggested_roles.append(AgentRole.ANALYSIS)
            elif classification.domain.value == 'coordination' and len(suggested_roles) == 1:
                # Coordination requests often need research and analysis agents too
                if AgentRole.RESEARCH not in suggested_roles:
                    suggested_roles.append(AgentRole.RESEARCH)
                if AgentRole.ANALYSIS not in suggested_roles:
                    suggested_roles.append(AgentRole.ANALYSIS)
        
        available_agents = []
        for role in suggested_roles:
            role_agents = self.agent_registry.list_agents_by_role(role)
            enabled_agents = [agent.name for agent in role_agents if agent.enabled]
            available_agents.extend(enabled_agents)
        
        if not available_agents:
            raise ValueError(f"No available agents for roles: {suggested_roles}")
        
        # Remove duplicates while preserving order
        available_agents = list(dict.fromkeys(available_agents))
        
        # Create execution parameters based on classification
        execution_params = ExecutionParams(
            max_agents=request.max_agents or len(available_agents),
            enable_streaming=request.enable_streaming,
            coordination_preference=request.coordination_preference or classification.suggested_pattern,
            parallel_execution=classification.suggested_pattern in [PatternType.SWARM, PatternType.GRAPH]
        )
        
        # Create coordination configuration
        coordination_config = self._create_coordination_config(classification, request)
        
        # Create the strategy
        strategy = AgentStrategy(
            pattern_type=classification.suggested_pattern,
            agents=available_agents,
            coordination_config=coordination_config,
            execution_params=execution_params,
            priority=self._calculate_strategy_priority(classification)
        )
        
        # Add fallback strategy if needed
        if classification.complexity.value in ['complex', 'very_complex']:
            strategy.fallback_strategy = self._create_fallback_strategy(request)
        
        return strategy
    
    def _create_coordination_config(
        self, 
        classification: ClassificationResult, 
        request: MultiAgentRequest
    ) -> Dict[str, Any]:
        """Create coordination configuration based on classification."""
        
        config = {
            'domain': classification.domain.value,
            'complexity': classification.complexity.value,
            'confidence': classification.confidence,
            'keywords': classification.keywords,
            'reasoning': classification.reasoning
        }
        
        # Pattern-specific configuration
        if classification.suggested_pattern == PatternType.SWARM:
            config.update({
                'max_handoffs': 5,
                'collaboration_mode': 'autonomous',
                'consensus_threshold': 0.7
            })
        elif classification.suggested_pattern == PatternType.WORKFLOW:
            config.update({
                'execution_order': 'sequential',
                'dependency_handling': 'strict',
                'checkpoint_enabled': True
            })
        elif classification.suggested_pattern == PatternType.GRAPH:
            config.update({
                'conditional_logic': True,
                'branching_enabled': True,
                'loop_detection': True
            })
        elif classification.suggested_pattern == PatternType.A2A:
            config.update({
                'communication_protocol': 'async',
                'discovery_enabled': True,
                'load_balancing': True
            })
        
        return config
    
    def _apply_user_preferences(self, strategy: AgentStrategy, request: MultiAgentRequest) -> AgentStrategy:
        """Apply user preferences and hints to the strategy."""
        
        # Apply strategy hint override
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
                strategy.pattern_type = pattern_mapping[hint_lower]
                logger.info(f"Applied strategy hint: {request.strategy_hint}")
        
        # Apply coordination preference
        if request.coordination_preference:
            strategy.pattern_type = request.coordination_preference
            logger.info(f"Applied coordination preference: {request.coordination_preference.value}")
        
        # Apply max agents limit
        if request.max_agents and request.max_agents < len(strategy.agents):
            # Keep the most relevant agents based on priority
            agent_priorities = self._get_agent_priorities(strategy.agents)
            sorted_agents = sorted(strategy.agents, key=lambda a: agent_priorities.get(a, 0), reverse=True)
            strategy.agents = sorted_agents[:request.max_agents]
            logger.info(f"Limited agents to {request.max_agents}: {strategy.agents}")
        
        return strategy
    
    def _validate_and_optimize_strategy(self, strategy: AgentStrategy, request: MultiAgentRequest) -> AgentStrategy:
        """Validate and optimize the strategy configuration."""
        
        # Validate that all agents exist and are enabled
        validated_agents = []
        for agent_name in strategy.agents:
            agent_config = self.agent_registry.get_agent_config(agent_name)
            if agent_config and agent_config.enabled:
                validated_agents.append(agent_name)
            else:
                logger.warning(f"Agent {agent_name} not available, removing from strategy")
        
        if not validated_agents:
            raise ValueError("No valid agents available for strategy execution")
        
        strategy.agents = validated_agents
        
        # Optimize based on historical performance
        if self.enable_adaptive_routing:
            strategy = self._apply_adaptive_optimizations(strategy)
        
        # Validate strategy with executor
        strategy_impl = self.strategy_executor.get_strategy(strategy.pattern_type.value)
        if strategy_impl and not strategy_impl.validate_strategy(strategy):
            logger.warning(f"Strategy validation failed for {strategy.pattern_type.value}, using fallback")
            return self._create_fallback_strategy(request)
        
        return strategy
    
    def _apply_adaptive_optimizations(self, strategy: AgentStrategy) -> AgentStrategy:
        """Apply adaptive optimizations based on historical performance."""
        
        pattern_key = strategy.pattern_type.value
        if pattern_key in self._strategy_performance:
            perf_data = self._strategy_performance[pattern_key]
            
            # Adjust timeouts based on historical performance
            avg_execution_time = perf_data.get('avg_execution_time', 300)
            strategy.execution_params.timeout_config.agent_timeout = int(avg_execution_time * 1.5)
            
            # Adjust retry attempts based on success rate
            success_rate = perf_data.get('success_rate', 0.8)
            if success_rate < 0.7:
                strategy.execution_params.retry_attempts = min(5, strategy.execution_params.retry_attempts + 1)
            
            logger.debug(f"Applied adaptive optimizations for {pattern_key}")
        
        return strategy
    
    def _get_agent_priorities(self, agent_names: List[str]) -> Dict[str, int]:
        """Get priority scores for agents based on their configuration and performance."""
        priorities = {}
        
        for agent_name in agent_names:
            agent_config = self.agent_registry.get_agent_config(agent_name)
            if agent_config:
                # Base priority from configuration
                priority = agent_config.priority
                
                # Boost priority based on specialization areas
                if len(agent_config.specialization_areas) > 0:
                    priority += len(agent_config.specialization_areas)
                
                priorities[agent_name] = priority
            else:
                priorities[agent_name] = 0
        
        return priorities
    
    def _calculate_strategy_priority(self, classification: ClassificationResult) -> int:
        """Calculate priority for the strategy based on classification confidence."""
        base_priority = 1
        
        # Higher confidence gets higher priority
        confidence_boost = int(classification.confidence * 3)
        
        # Complex requests get higher priority
        complexity_boost = {
            'simple': 0,
            'moderate': 1,
            'complex': 2,
            'very_complex': 3
        }.get(classification.complexity.value, 0)
        
        return base_priority + confidence_boost + complexity_boost
    
    def _create_fallback_strategy(self, request: MultiAgentRequest) -> AgentStrategy:
        """Create a fallback strategy for when routing fails."""
        
        # Try to find any available general chat agent
        general_agents = self.agent_registry.list_agents_by_role(AgentRole.GENERAL_CHAT)
        available_general = [agent.name for agent in general_agents if agent.enabled]
        
        if not available_general:
            # If no general agents, try any available agent
            all_configs = self.agent_registry.list_agent_configs()
            available_general = [config.name for config in all_configs if config.enabled]
        
        if not available_general:
            raise RuntimeError("No agents available for fallback strategy")
        
        return AgentStrategy(
            pattern_type=PatternType.SINGLE,
            agents=[available_general[0]],  # Use first available agent
            coordination_config={'fallback': True, 'reason': 'routing_failure'},
            execution_params=ExecutionParams(
                max_agents=1,
                retry_attempts=1,
                enable_streaming=request.enable_streaming
            ),
            priority=0  # Lowest priority
        )
    
    def _record_routing_decision(
        self, 
        request: MultiAgentRequest, 
        classification: ClassificationResult, 
        strategy: AgentStrategy
    ):
        """Record routing decision for analysis and optimization."""
        
        routing_record = {
            'timestamp': datetime.now(),
            'request_message': request.message[:100],  # Truncate for storage
            'classification': {
                'domain': classification.domain.value,
                'complexity': classification.complexity.value,
                'confidence': classification.confidence,
                'keywords': classification.keywords[:5]  # Limit keywords
            },
            'strategy': {
                'pattern_type': strategy.pattern_type.value,
                'agents': strategy.agents,
                'priority': strategy.priority
            },
            'user_preferences': {
                'strategy_hint': request.strategy_hint,
                'coordination_preference': request.coordination_preference.value if request.coordination_preference else None,
                'max_agents': request.max_agents
            }
        }
        
        self._routing_history.append(routing_record)
        
        # Keep only last 1000 routing decisions
        if len(self._routing_history) > 1000:
            self._routing_history = self._routing_history[-1000:]
    
    def update_strategy_performance(self, strategy: AgentStrategy, result: MultiAgentResult):
        """Update strategy performance metrics based on execution results."""
        
        pattern_key = strategy.pattern_type.value
        
        if pattern_key not in self._strategy_performance:
            self._strategy_performance[pattern_key] = {
                'total_executions': 0,
                'successful_executions': 0,
                'total_execution_time': 0.0,
                'avg_execution_time': 0.0,
                'success_rate': 0.0
            }
        
        perf_data = self._strategy_performance[pattern_key]
        perf_data['total_executions'] += 1
        
        if result.status == ExecutionStatus.COMPLETED:
            perf_data['successful_executions'] += 1
        
        if result.execution_metadata.total_execution_time:
            perf_data['total_execution_time'] += result.execution_metadata.total_execution_time
            perf_data['avg_execution_time'] = perf_data['total_execution_time'] / perf_data['total_executions']
        
        perf_data['success_rate'] = perf_data['successful_executions'] / perf_data['total_executions']
        
        logger.debug(f"Updated performance metrics for {pattern_key}: "
                    f"Success rate: {perf_data['success_rate']:.2f}, "
                    f"Avg time: {perf_data['avg_execution_time']:.2f}s")
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing statistics and performance metrics."""
        
        total_routes = len(self._routing_history)
        if total_routes == 0:
            return {'total_routes': 0}
        
        # Pattern distribution
        pattern_counts = {}
        domain_counts = {}
        complexity_counts = {}
        
        for record in self._routing_history:
            pattern = record['strategy']['pattern_type']
            domain = record['classification']['domain']
            complexity = record['classification']['complexity']
            
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        
        # Average confidence
        avg_confidence = sum(r['classification']['confidence'] for r in self._routing_history) / total_routes
        
        return {
            'total_routes': total_routes,
            'pattern_distribution': pattern_counts,
            'domain_distribution': domain_counts,
            'complexity_distribution': complexity_counts,
            'average_confidence': avg_confidence,
            'strategy_performance': self._strategy_performance.copy(),
            'adaptive_routing_enabled': self.enable_adaptive_routing
        }
    
    def get_recommended_strategy(self, request: MultiAgentRequest) -> Tuple[AgentStrategy, float]:
        """
        Get recommended strategy with confidence score.
        
        Args:
            request: The multi-agent request
            
        Returns:
            Tuple of (strategy, confidence_score)
        """
        classification = self.classifier.classify_request(request)
        strategy = self._create_strategy_from_classification(classification, request)
        strategy = self._apply_user_preferences(strategy, request)
        
        # Calculate overall confidence based on classification and agent availability
        agent_availability_score = len(strategy.agents) / max(len(classification.suggested_agents), 1)
        overall_confidence = (classification.confidence + agent_availability_score) / 2
        
        return strategy, overall_confidence
    
    def set_adaptive_routing(self, enabled: bool):
        """Enable or disable adaptive routing optimizations."""
        self.enable_adaptive_routing = enabled
        logger.info(f"Adaptive routing {'enabled' if enabled else 'disabled'}")
    
    def clear_performance_history(self):
        """Clear routing and performance history."""
        self._routing_history.clear()
        self._strategy_performance.clear()
        logger.info("Cleared routing and performance history")
    
    async def execute_strategy(self, request: MultiAgentRequest, strategy: AgentStrategy) -> MultiAgentResult:
        """
        Execute a strategy with result synthesis and coordination.
        
        Args:
            request: The multi-agent request to process
            strategy: The strategy to execute
            
        Returns:
            MultiAgentResult containing synthesized results
            
        Raises:
            RuntimeError: If strategy execution fails
        """
        execution_start = datetime.now()
        
        try:
            logger.info(f"Executing strategy {strategy.pattern_type.value} with agents: {strategy.agents}")
            
            # Execute the strategy using the strategy executor
            result = await self.strategy_executor.execute_strategy(request, strategy)
            
            # Synthesize and coordinate results
            synthesized_result = await self._synthesize_results(result, strategy, request)
            
            # Update performance metrics
            self.update_strategy_performance(strategy, synthesized_result)
            
            logger.info(f"Strategy execution completed successfully in "
                       f"{synthesized_result.execution_metadata.total_execution_time:.2f}s")
            
            return synthesized_result
            
        except Exception as e:
            logger.error(f"Strategy execution failed: {str(e)}")
            
            # Try fallback strategy if available
            if strategy.fallback_strategy:
                logger.info("Attempting fallback strategy execution")
                try:
                    fallback_result = await self.strategy_executor.execute_strategy(request, strategy.fallback_strategy)
                    fallback_result.execution_metadata.pattern_used = strategy.fallback_strategy.pattern_type
                    return fallback_result
                except Exception as fallback_error:
                    logger.error(f"Fallback strategy also failed: {str(fallback_error)}")
            
            # Create error result
            error_context = ErrorContext(
                error_type=type(e).__name__,
                error_message=str(e),
                failed_strategy=strategy,
                timestamp=datetime.now()
            )
            
            return MultiAgentResult(
                status=ExecutionStatus.FAILED,
                primary_result=f"Strategy execution failed: {str(e)}",
                execution_metadata=ExecutionMetadata(
                    start_time=execution_start,
                    end_time=datetime.now(),
                    pattern_used=strategy.pattern_type
                ),
                error_context=error_context
            )
    
    async def _synthesize_results(
        self, 
        result: MultiAgentResult, 
        strategy: AgentStrategy, 
        request: MultiAgentRequest
    ) -> MultiAgentResult:
        """
        Synthesize and coordinate results from multi-agent execution.
        
        Args:
            result: Raw result from strategy execution
            strategy: The executed strategy
            request: Original request
            
        Returns:
            Synthesized MultiAgentResult
        """
        # If execution was successful, enhance the result
        if result.status == ExecutionStatus.COMPLETED:
            # Add coordination metadata
            result.execution_metadata.pattern_used = strategy.pattern_type
            
            # Synthesize results based on pattern type
            if strategy.pattern_type == PatternType.SWARM:
                result = await self._synthesize_swarm_results(result, strategy)
            elif strategy.pattern_type == PatternType.WORKFLOW:
                result = await self._synthesize_workflow_results(result, strategy)
            elif strategy.pattern_type == PatternType.GRAPH:
                result = await self._synthesize_graph_results(result, strategy)
            elif strategy.pattern_type == PatternType.A2A:
                result = await self._synthesize_a2a_results(result, strategy)
            
            # Add final coordination summary
            result = await self._add_coordination_summary(result, strategy, request)
        
        return result
    
    async def _synthesize_swarm_results(self, result: MultiAgentResult, strategy: AgentStrategy) -> MultiAgentResult:
        """Synthesize results from swarm pattern execution."""
        if len(result.agent_contributions) > 1:
            # Combine insights from multiple agents
            contributions = []
            for agent_name, agent_result in result.agent_contributions.items():
                if agent_result.status == ExecutionStatus.COMPLETED:
                    contributions.append(f"**{agent_name}**: {agent_result.content}")
            
            if contributions:
                result.primary_result = (
                    "## Collaborative Analysis\n\n" +
                    "\n\n".join(contributions) +
                    f"\n\n## Summary\n\n{result.primary_result}"
                )
        
        return result
    
    async def _synthesize_workflow_results(self, result: MultiAgentResult, strategy: AgentStrategy) -> MultiAgentResult:
        """Synthesize results from workflow pattern execution."""
        if len(result.agent_contributions) > 1:
            # Show sequential progression
            workflow_steps = []
            for i, (agent_name, agent_result) in enumerate(result.agent_contributions.items(), 1):
                if agent_result.status == ExecutionStatus.COMPLETED:
                    workflow_steps.append(f"**Step {i} ({agent_name})**: {agent_result.content}")
            
            if workflow_steps:
                result.primary_result = (
                    "## Workflow Execution\n\n" +
                    "\n\n".join(workflow_steps) +
                    f"\n\n## Final Result\n\n{result.primary_result}"
                )
        
        return result
    
    async def _synthesize_graph_results(self, result: MultiAgentResult, strategy: AgentStrategy) -> MultiAgentResult:
        """Synthesize results from graph pattern execution."""
        if len(result.agent_contributions) > 1:
            # Show decision tree structure
            decision_points = []
            for agent_name, agent_result in result.agent_contributions.items():
                if agent_result.status == ExecutionStatus.COMPLETED:
                    decision_points.append(f"**{agent_name} Decision**: {agent_result.content}")
            
            if decision_points:
                result.primary_result = (
                    "## Decision Tree Analysis\n\n" +
                    "\n\n".join(decision_points) +
                    f"\n\n## Conclusion\n\n{result.primary_result}"
                )
        
        return result
    
    async def _synthesize_a2a_results(self, result: MultiAgentResult, strategy: AgentStrategy) -> MultiAgentResult:
        """Synthesize results from agent-to-agent pattern execution."""
        if len(result.agent_contributions) > 1:
            # Show communication flow
            communications = []
            for agent_name, agent_result in result.agent_contributions.items():
                if agent_result.status == ExecutionStatus.COMPLETED:
                    communications.append(f"**{agent_name}**: {agent_result.content}")
            
            if communications:
                result.primary_result = (
                    "## Agent Communications\n\n" +
                    "\n\n".join(communications) +
                    f"\n\n## Coordinated Response\n\n{result.primary_result}"
                )
        
        return result
    
    async def _add_coordination_summary(
        self, 
        result: MultiAgentResult, 
        strategy: AgentStrategy, 
        request: MultiAgentRequest
    ) -> MultiAgentResult:
        """Add coordination summary to the result."""
        
        # Count successful agent contributions
        successful_agents = sum(
            1 for agent_result in result.agent_contributions.values()
            if agent_result.status == ExecutionStatus.COMPLETED
        )
        
        # Add coordination metadata
        coordination_summary = {
            'pattern_used': strategy.pattern_type.value,
            'agents_involved': len(strategy.agents),
            'successful_agents': successful_agents,
            'coordination_events': len(result.coordination_log),
            'execution_time': result.execution_metadata.total_execution_time
        }
        
        # Add to result metadata
        if not hasattr(result.execution_metadata, 'coordination_summary'):
            result.execution_metadata.coordination_summary = coordination_summary
        
        return result