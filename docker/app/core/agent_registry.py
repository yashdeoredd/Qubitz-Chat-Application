"""
Registry for managing agent configurations and instances.
"""

from typing import Dict, List, Optional, Any
from strands import Agent

from app.models.multi_agent_models import AgentConfig, AgentRole


class AgentRegistry:
    """
    Registry for managing agent configurations and instances.
    
    This class maintains a registry of available agents, their configurations,
    and provides methods to create and retrieve agent instances.
    """
    
    def __init__(self):
        """Initialize the agent registry."""
        self._agent_configs: Dict[str, AgentConfig] = {}
        self._agent_instances: Dict[str, Agent] = {}
        
    def register_agent_config(self, config: AgentConfig):
        """
        Register an agent configuration.
        
        Args:
            config: The agent configuration to register
        """
        self._agent_configs[config.name] = config
        
        # Clear cached instance if it exists
        if config.name in self._agent_instances:
            del self._agent_instances[config.name]
    
    def get_agent_config(self, name: str) -> Optional[AgentConfig]:
        """
        Get an agent configuration by name.
        
        Args:
            name: Name of the agent configuration
            
        Returns:
            AgentConfig if found, None otherwise
        """
        return self._agent_configs.get(name)
    
    def list_agent_configs(self) -> List[AgentConfig]:
        """
        List all registered agent configurations.
        
        Returns:
            List of all agent configurations
        """
        return list(self._agent_configs.values())
    
    def list_agents_by_role(self, role: AgentRole) -> List[AgentConfig]:
        """
        List all agent configurations for a specific role.
        
        Args:
            role: The agent role to filter by
            
        Returns:
            List of agent configurations with the specified role
        """
        return [config for config in self._agent_configs.values() if config.role == role]
    
    def get_agent_instance(self, name: str) -> Optional[Agent]:
        """
        Get or create an agent instance by name.
        
        Args:
            name: Name of the agent
            
        Returns:
            Agent instance if configuration exists, None otherwise
        """
        if name in self._agent_instances:
            return self._agent_instances[name]
        
        config = self.get_agent_config(name)
        if not config or not config.enabled:
            return None
        
        # Create new agent instance
        # Note: This is a simplified version - actual implementation would need
        # to resolve tools and other dependencies
        agent = Agent(
            system_prompt=config.system_prompt,
            tools=[]  # Tools would be resolved from config.tools
        )
        
        self._agent_instances[name] = agent
        return agent
    
    def remove_agent_config(self, name: str) -> bool:
        """
        Remove an agent configuration.
        
        Args:
            name: Name of the agent configuration to remove
            
        Returns:
            True if removed, False if not found
        """
        if name in self._agent_configs:
            del self._agent_configs[name]
            
            # Also remove cached instance
            if name in self._agent_instances:
                del self._agent_instances[name]
                
            return True
        return False
    
    def enable_agent(self, name: str) -> bool:
        """
        Enable an agent configuration.
        
        Args:
            name: Name of the agent to enable
            
        Returns:
            True if enabled, False if not found
        """
        config = self.get_agent_config(name)
        if config:
            config.enabled = True
            return True
        return False
    
    def disable_agent(self, name: str) -> bool:
        """
        Disable an agent configuration.
        
        Args:
            name: Name of the agent to disable
            
        Returns:
            True if disabled, False if not found
        """
        config = self.get_agent_config(name)
        if config:
            config.enabled = False
            
            # Remove cached instance
            if name in self._agent_instances:
                del self._agent_instances[name]
                
            return True
        return False
    
    def get_registry_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the agent registry.
        
        Returns:
            Dictionary containing registry statistics
        """
        total_configs = len(self._agent_configs)
        enabled_configs = sum(1 for config in self._agent_configs.values() if config.enabled)
        cached_instances = len(self._agent_instances)
        
        role_counts = {}
        for config in self._agent_configs.values():
            role = config.role.value
            role_counts[role] = role_counts.get(role, 0) + 1
        
        return {
            "total_configurations": total_configs,
            "enabled_configurations": enabled_configs,
            "cached_instances": cached_instances,
            "configurations_by_role": role_counts
        }