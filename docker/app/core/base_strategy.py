"""
Base classes for agent strategies and execution.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime

from app.models.multi_agent_models import (
    AgentStrategy,
    MultiAgentRequest,
    MultiAgentResult,
    AgentResult,
    ExecutionStatus,
    ExecutionMetadata,
    CoordinationEvent,
    ErrorContext
)


class BaseStrategy(ABC):
    """
    Abstract base class for multi-agent execution strategies.
    
    This class defines the interface that all strategy implementations must follow.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize the strategy.
        
        Args:
            name: Name of the strategy
            description: Optional description of the strategy
        """
        self.name = name
        self.description = description
        self.execution_count = 0
        self.success_count = 0
        
    @abstractmethod
    async def execute(
        self, 
        request: MultiAgentRequest, 
        strategy: AgentStrategy
    ) -> MultiAgentResult:
        """
        Execute the strategy with the given request and configuration.
        
        Args:
            request: The multi-agent request to process
            strategy: The strategy configuration to use
            
        Returns:
            MultiAgentResult containing the execution results
        """
        pass
    
    @abstractmethod
    def validate_strategy(self, strategy: AgentStrategy) -> bool:
        """
        Validate that the strategy configuration is valid for this implementation.
        
        Args:
            strategy: The strategy to validate
            
        Returns:
            True if valid, False otherwise
        """
        pass
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for this strategy.
        
        Returns:
            Dictionary containing performance metrics
        """
        success_rate = (self.success_count / self.execution_count) if self.execution_count > 0 else 0.0
        return {
            "name": self.name,
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "success_rate": success_rate
        }
    
    def _record_execution(self, success: bool):
        """Record execution statistics."""
        self.execution_count += 1
        if success:
            self.success_count += 1


class StrategyExecutor:
    """
    Manages and executes different agent strategies.
    
    This class coordinates the execution of various multi-agent patterns
    and handles strategy selection, execution, and error recovery.
    """
    
    def __init__(self):
        """Initialize the strategy executor."""
        self._strategies: Dict[str, BaseStrategy] = {}
        self._execution_history: List[Dict[str, Any]] = []
        
    def register_strategy(self, strategy: BaseStrategy):
        """
        Register a strategy implementation.
        
        Args:
            strategy: The strategy implementation to register
        """
        self._strategies[strategy.name] = strategy
        
    def get_strategy(self, name: str) -> Optional[BaseStrategy]:
        """
        Get a registered strategy by name.
        
        Args:
            name: Name of the strategy to retrieve
            
        Returns:
            The strategy implementation or None if not found
        """
        return self._strategies.get(name)
    
    def list_strategies(self) -> List[str]:
        """
        List all registered strategy names.
        
        Returns:
            List of strategy names
        """
        return list(self._strategies.keys())
    
    async def execute_strategy(
        self, 
        request: MultiAgentRequest, 
        strategy: AgentStrategy
    ) -> MultiAgentResult:
        """
        Execute a strategy with the given request.
        
        Args:
            request: The multi-agent request to process
            strategy: The strategy configuration to use
            
        Returns:
            MultiAgentResult containing the execution results
            
        Raises:
            ValueError: If strategy pattern type is not registered
            RuntimeError: If strategy execution fails
        """
        pattern_name = strategy.pattern_type.value
        strategy_impl = self.get_strategy(pattern_name)
        
        if not strategy_impl:
            raise ValueError(f"Strategy '{pattern_name}' is not registered")
        
        if not strategy_impl.validate_strategy(strategy):
            raise ValueError(f"Invalid strategy configuration for '{pattern_name}'")
        
        execution_start = datetime.now()
        
        try:
            result = await strategy_impl.execute(request, strategy)
            
            # Record successful execution
            strategy_impl._record_execution(True)
            self._record_execution_history(request, strategy, result, None)
            
            return result
            
        except Exception as e:
            # Record failed execution
            strategy_impl._record_execution(False)
            
            # Create error result
            error_context = ErrorContext(
                error_type=type(e).__name__,
                error_message=str(e),
                failed_strategy=strategy,
                timestamp=datetime.now()
            )
            
            result = MultiAgentResult(
                status=ExecutionStatus.FAILED,
                primary_result=f"Strategy execution failed: {str(e)}",
                execution_metadata=ExecutionMetadata(
                    start_time=execution_start,
                    end_time=datetime.now(),
                    pattern_used=strategy.pattern_type
                ),
                error_context=error_context
            )
            
            self._record_execution_history(request, strategy, result, e)
            
            return result
    
    def _record_execution_history(
        self, 
        request: MultiAgentRequest, 
        strategy: AgentStrategy, 
        result: MultiAgentResult, 
        error: Optional[Exception]
    ):
        """Record execution in history for analysis."""
        self._execution_history.append({
            "timestamp": datetime.now(),
            "request_message": request.message[:100],  # Truncate for storage
            "strategy_pattern": strategy.pattern_type.value,
            "agents_used": strategy.agents,
            "status": result.status.value,
            "execution_time": result.execution_metadata.total_execution_time,
            "error": str(error) if error else None
        })
        
        # Keep only last 1000 executions
        if len(self._execution_history) > 1000:
            self._execution_history = self._execution_history[-1000:]
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """
        Get overall execution statistics.
        
        Returns:
            Dictionary containing execution statistics
        """
        total_executions = len(self._execution_history)
        if total_executions == 0:
            return {"total_executions": 0}
        
        successful_executions = sum(1 for h in self._execution_history if h["status"] == "completed")
        
        strategy_stats = {}
        for strategy_name, strategy in self._strategies.items():
            strategy_stats[strategy_name] = strategy.get_performance_metrics()
        
        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "success_rate": successful_executions / total_executions,
            "strategy_statistics": strategy_stats
        }