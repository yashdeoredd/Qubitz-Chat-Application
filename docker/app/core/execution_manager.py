"""
Execution manager for coordinating multi-agent operations.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable

from app.models.multi_agent_models import (
    MultiAgentRequest,
    MultiAgentResult,
    AgentResult,
    CoordinationEvent,
    ExecutionStatus,
    ExecutionMetadata,
    AgentRole,
    ErrorContext,
    RecoveryAction
)
from .agent_registry import AgentRegistry


class ExecutionManager:
    """
    Manages the execution of multi-agent operations.
    
    This class coordinates agent execution, handles timeouts, manages
    coordination events, and provides execution monitoring capabilities.
    """
    
    def __init__(self, agent_registry: AgentRegistry):
        """
        Initialize the execution manager.
        
        Args:
            agent_registry: Registry for agent configurations and instances
        """
        self.agent_registry = agent_registry
        self._active_executions: Dict[str, Dict[str, Any]] = {}
        self._coordination_events: List[CoordinationEvent] = []
        self._execution_callbacks: List[Callable] = []
        
    def add_execution_callback(self, callback: Callable):
        """
        Add a callback to be called on execution events.
        
        Args:
            callback: Function to call on execution events
        """
        self._execution_callbacks.append(callback)
    
    def remove_execution_callback(self, callback: Callable):
        """
        Remove an execution callback.
        
        Args:
            callback: Function to remove from callbacks
        """
        if callback in self._execution_callbacks:
            self._execution_callbacks.remove(callback)
    
    async def execute_single_agent(
        self, 
        agent_name: str, 
        message: str, 
        context: Dict[str, Any] = None
    ) -> AgentResult:
        """
        Execute a single agent with the given message.
        
        Args:
            agent_name: Name of the agent to execute
            message: Message to send to the agent
            context: Optional context information
            
        Returns:
            AgentResult containing the execution result
        """
        start_time = datetime.now()
        
        try:
            agent = self.agent_registry.get_agent_instance(agent_name)
            if not agent:
                return AgentResult(
                    agent_name=agent_name,
                    agent_role=AgentRole.GENERAL_CHAT,  # Default role
                    status=ExecutionStatus.FAILED,
                    content="",
                    error_message=f"Agent '{agent_name}' not found or disabled",
                    timestamp=start_time
                )
            
            # Get agent configuration for role information
            config = self.agent_registry.get_agent_config(agent_name)
            agent_role = config.role if config else AgentRole.GENERAL_CHAT
            
            # Execute agent
            response = agent(message)
            content = str(response)
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            result = AgentResult(
                agent_name=agent_name,
                agent_role=agent_role,
                status=ExecutionStatus.COMPLETED,
                content=content,
                execution_time=execution_time,
                timestamp=start_time
            )
            
            # Notify callbacks
            await self._notify_callbacks("agent_completed", {
                "agent_name": agent_name,
                "result": result
            })
            
            return result
            
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Get agent configuration for role information
            config = self.agent_registry.get_agent_config(agent_name)
            agent_role = config.role if config else AgentRole.GENERAL_CHAT
            
            result = AgentResult(
                agent_name=agent_name,
                agent_role=agent_role,
                status=ExecutionStatus.FAILED,
                content="",
                execution_time=execution_time,
                error_message=str(e),
                timestamp=start_time
            )
            
            # Notify callbacks
            await self._notify_callbacks("agent_failed", {
                "agent_name": agent_name,
                "result": result,
                "error": e
            })
            
            return result
    
    async def execute_parallel_agents(
        self, 
        agent_names: List[str], 
        message: str, 
        context: Dict[str, Any] = None
    ) -> List[AgentResult]:
        """
        Execute multiple agents in parallel.
        
        Args:
            agent_names: List of agent names to execute
            message: Message to send to all agents
            context: Optional context information
            
        Returns:
            List of AgentResult objects
        """
        tasks = []
        for agent_name in agent_names:
            task = self.execute_single_agent(agent_name, message, context)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to failed AgentResults
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(AgentResult(
                    agent_name=agent_names[i],
                    agent_role=AgentRole.GENERAL_CHAT,
                    status=ExecutionStatus.FAILED,
                    content="",
                    error_message=str(result),
                    timestamp=datetime.now()
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def record_coordination_event(
        self, 
        event_type: str, 
        source_agent: str, 
        target_agent: Optional[str] = None,
        context: Dict[str, Any] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """
        Record a coordination event.
        
        Args:
            event_type: Type of coordination event
            source_agent: Agent that initiated the event
            target_agent: Target agent (if applicable)
            context: Additional context information
            success: Whether the event was successful
            error_message: Error message if unsuccessful
        """
        event = CoordinationEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            source_agent=source_agent,
            target_agent=target_agent,
            context=context or {},
            success=success,
            error_message=error_message
        )
        
        self._coordination_events.append(event)
        
        # Keep only last 10000 events
        if len(self._coordination_events) > 10000:
            self._coordination_events = self._coordination_events[-10000:]
    
    def get_coordination_events(
        self, 
        limit: Optional[int] = None,
        event_type: Optional[str] = None,
        agent_name: Optional[str] = None
    ) -> List[CoordinationEvent]:
        """
        Get coordination events with optional filtering.
        
        Args:
            limit: Maximum number of events to return
            event_type: Filter by event type
            agent_name: Filter by agent name (source or target)
            
        Returns:
            List of coordination events
        """
        events = self._coordination_events
        
        # Apply filters
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if agent_name:
            events = [e for e in events if e.source_agent == agent_name or e.target_agent == agent_name]
        
        # Sort by timestamp (most recent first)
        events = sorted(events, key=lambda e: e.timestamp, reverse=True)
        
        # Apply limit
        if limit:
            events = events[:limit]
        
        return events
    
    async def _notify_callbacks(self, event_type: str, data: Dict[str, Any]):
        """Notify all registered callbacks about an event."""
        for callback in self._execution_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event_type, data)
                else:
                    callback(event_type, data)
            except Exception as e:
                # Log callback errors but don't fail execution
                print(f"Callback error: {e}")
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """
        Get execution statistics.
        
        Returns:
            Dictionary containing execution statistics
        """
        total_events = len(self._coordination_events)
        successful_events = sum(1 for e in self._coordination_events if e.success)
        
        event_type_counts = {}
        for event in self._coordination_events:
            event_type_counts[event.event_type] = event_type_counts.get(event.event_type, 0) + 1
        
        return {
            "total_coordination_events": total_events,
            "successful_coordination_events": successful_events,
            "coordination_success_rate": successful_events / total_events if total_events > 0 else 0.0,
            "event_type_counts": event_type_counts,
            "active_executions": len(self._active_executions)
        }