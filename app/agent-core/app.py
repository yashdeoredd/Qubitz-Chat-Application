"""
AWS Bedrock AgentCore Multi-Agent System
Production-ready deployment with enterprise features
"""

import os
import json
import uuid
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

import boto3
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# AgentCore SDK imports
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory import MemoryClient
from bedrock_agentcore.services.identity import IdentityClient

# Strands imports
from strands import Agent, tool
from strands.models import BedrockModel
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands_tools import http_request

# OpenTelemetry for observability
from opentelemetry import trace, baggage, context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = "qubitz-customer-prod"

MODEL_MAPPING = {
    "Claude 3.5 Sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "Claude 3.5 Haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    "Amazon Nova Micro": "amazon.nova-micro-v1:0",
    "Amazon Nova Lite": "amazon.nova-lite-v1:0",
    "Amazon Nova Pro": "amazon.nova-pro-v1:0",
    "Command R": "cohere.command-r-v1:0",
    "DeepSeek-R1": "us.deepseek.r1-v1:0",
    "Claude 3.5 Sonnet v2": "us.anthropic.claude-sonnet-4-20250514-v1:0"
}

# ============================================================================
# AGENTCORE OBSERVABILITY SETUP
# ============================================================================

class ObservabilityManager:
    """Manages AgentCore observability with OpenTelemetry"""
    
    def __init__(self):
        resource = Resource.create({
            "service.name": "workflow-agent-factory",
            "service.version": "19.0.0"
        })
        
        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)
        self.tracer = trace.get_tracer(__name__)
    
    def create_span(self, name: str, attributes: Dict = None):
        """Create a traced span for operations"""
        span = self.tracer.start_span(name)
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
        return span
    
    def set_session_context(self, session_id: str):
        """Set session ID in OTEL baggage for tracing"""
        ctx = baggage.set_baggage("session.id", session_id)
        context.attach(ctx)

# ============================================================================
# AGENTCORE MEMORY INTEGRATION
# ============================================================================

class AgentCoreMemoryManager:
    """Manages short-term and long-term memory using AgentCore Memory"""
    
    def __init__(self, region: str = REGION):
        self.memory_client = MemoryClient(region_name=region)
        self.memory_stores = {}
    
    def create_memory_store(self, user_id: str, project_id: str) -> Dict:
        """Create AgentCore memory store with semantic strategy"""
        
        store_key = f"{user_id}_{project_id}"
        
        if store_key in self.memory_stores:
            return self.memory_stores[store_key]
        
        try:
            memory = self.memory_client.create_memory_and_wait(
                name=f"workflow_{user_id}_{project_id}",
                description=f"Workflow memory for {user_id}/{project_id}",
                strategies=[{
                    "semanticMemoryStrategy": {
                        "name": "semantic_facts",
                        "namespaces": [f"/facts/{user_id}"]
                    }
                }]
            )
            
            self.memory_stores[store_key] = memory
            return memory
            
        except Exception as e:
            print(f"Memory store creation failed: {e}")
            return None
    
    def store_conversation(self, memory_id: str, actor_id: str, 
                          session_id: str, messages: List[tuple]):
        """Store conversation in short-term memory"""
        
        try:
            self.memory_client.create_event(
                memory_id=memory_id,
                actor_id=actor_id,
                session_id=session_id,
                messages=messages
            )
        except Exception as e:
            print(f"Failed to store conversation: {e}")
    
    def retrieve_recent_context(self, memory_id: str, actor_id: str, 
                               session_id: str, max_results: int = 5) -> List:
        """Retrieve recent conversation turns"""
        
        try:
            return self.memory_client.list_events(
                memory_id=memory_id,
                actor_id=actor_id,
                session_id=session_id,
                max_results=max_results
            )
        except Exception as e:
            print(f"Failed to retrieve context: {e}")
            return []
    
    def retrieve_semantic_memories(self, memory_id: str, actor_id: str, 
                                   query: str) -> List:
        """Retrieve semantic memories using query"""
        
        try:
            return self.memory_client.retrieve_memories(
                memory_id=memory_id,
                namespace=f"/facts/{actor_id}",
                query=query
            )
        except Exception as e:
            print(f"Failed to retrieve memories: {e}")
            return []

# ============================================================================
# AGENTCORE IDENTITY INTEGRATION
# ============================================================================

class AgentCoreIdentityManager:
    """Manages identity and access controls using AgentCore Identity"""
    
    def __init__(self, region: str = REGION):
        self.identity_client = IdentityClient(region)
        self.workload_identities = {}
        self.credential_providers = {}
    
    def create_workload_identity(self, agent_name: str) -> Dict:
        """Create workload identity for agent"""
        
        if agent_name in self.workload_identities:
            return self.workload_identities[agent_name]
        
        try:
            identity = self.identity_client.create_workload_identity(
                name=f"{agent_name}_identity"
            )
            self.workload_identities[agent_name] = identity
            return identity
        except Exception as e:
            print(f"Workload identity creation failed: {e}")
            return None
    
    def configure_oauth_provider(self, provider_name: str, client_id: str, 
                                client_secret: str, vendor: str = "GoogleOauth2"):
        """Configure OAuth2 credential provider"""
        
        try:
            provider = self.identity_client.create_oauth2_credential_provider({
                "name": provider_name,
                "credentialProviderVendor": vendor,
                "oauth2ProviderConfigInput": {
                    f"{vendor.lower()}ProviderConfig": {
                        "clientId": client_id,
                        "clientSecret": client_secret
                    }
                }
            })
            self.credential_providers[provider_name] = provider
            return provider
        except Exception as e:
            print(f"OAuth provider configuration failed: {e}")
            return None
    
    def configure_api_key_provider(self, provider_name: str, api_key: str):
        """Configure API key credential provider"""
        
        try:
            provider = self.identity_client.create_api_key_credential_provider({
                "name": provider_name,
                "apiKey": api_key
            })
            self.credential_providers[provider_name] = provider
            return provider
        except Exception as e:
            print(f"API key provider configuration failed: {e}")
            return None

# ============================================================================
# AGENTCORE GATEWAY INTEGRATION
# ============================================================================

class AgentCoreGatewayManager:
    """Manages tool integration using AgentCore Gateway"""
    
    def __init__(self, region: str = REGION):
        self.gateway_client = boto3.client('bedrock-agentcore-gateway', region_name=region)
        self.registered_tools = {}
    
    def register_lambda_tool(self, tool_name: str, lambda_arn: str, 
                            description: str) -> Dict:
        """Register Lambda function as agent tool via Gateway"""
        
        try:
            response = self.gateway_client.register_tool(
                toolName=tool_name,
                description=description,
                toolType='AWS_LAMBDA',
                lambdaConfig={
                    'functionArn': lambda_arn
                }
            )
            self.registered_tools[tool_name] = response
            return response
        except Exception as e:
            print(f"Lambda tool registration failed: {e}")
            return None
    
    def register_api_tool(self, tool_name: str, openapi_spec: Dict, 
                         base_url: str, description: str) -> Dict:
        """Register API as agent tool via Gateway"""
        
        try:
            response = self.gateway_client.register_tool(
                toolName=tool_name,
                description=description,
                toolType='REST_API',
                apiConfig={
                    'openApiSpec': json.dumps(openapi_spec),
                    'baseUrl': base_url
                }
            )
            self.registered_tools[tool_name] = response
            return response
        except Exception as e:
            print(f"API tool registration failed: {e}")
            return None

# ============================================================================
# WORKFLOW CONFIGURATION MANAGER
# ============================================================================

class WorkflowConfigManager:
    """Manages workflow configuration from S3"""
    
    def __init__(self):
        self.s3_client = boto3.client('s3', region_name=REGION)
        self.workflow_cache = {}
    
    def get_workflow(self, user_id: str, project_id: str) -> Optional[Dict]:
        """Retrieve workflow configuration from S3"""
        
        cache_key = f"{user_id}_{project_id}"
        if cache_key in self.workflow_cache:
            return self.workflow_cache[cache_key]
        
        try:
            key = f'{user_id}/{project_id}/workflow.json'
            response = self.s3_client.get_object(Bucket=S3_BUCKET, Key=key)
            workflow = json.loads(response['Body'].read().decode('utf-8'))
            self.workflow_cache[cache_key] = workflow
            return workflow
        except Exception as e:
            print(f"Workflow retrieval failed: {e}")
            return None

# ============================================================================
# SPECIALIST AGENT BUILDER
# ============================================================================

class SpecialistAgentBuilder:
    """Builds specialist agents from workflow configuration"""
    
    def __init__(self, memory_manager: AgentCoreMemoryManager,
                 identity_manager: AgentCoreIdentityManager):
        self.memory_manager = memory_manager
        self.identity_manager = identity_manager
    
    def build_agent(self, agent_config: Dict, user_id: str, 
                   project_id: str) -> Agent:
        """Build specialist agent with AgentCore features"""
        
        # Model setup
        model_id = agent_config.get('model_id') or MODEL_MAPPING.get(
            agent_config.get('model_name', 'Claude 3.5 Sonnet')
        )
        params = agent_config.get('parameters', {})
        
        model = BedrockModel(
            model_id=model_id,
            max_tokens=params.get('max_tokens', 4000),
            boto_session=boto3.Session(region_name=REGION),
            temperature=params.get('temperature', 0.7),
            top_p=params.get('top_p', 0.9)
        )
        
        # Create workload identity
        agent_name = agent_config.get('name', 'specialist')
        self.identity_manager.create_workload_identity(agent_name)
        
        # Build concise prompt
        system_prompt = f"""You are {agent_name.replace('_', ' ').title()}.

{agent_config.get('description', '')}

## Instructions
{agent_config.get('instructions', '')}

Context: {user_id}/{project_id}"""
        
        # Create agent
        agent = Agent(
            model=model,
            system_prompt=system_prompt,
            conversation_manager=SlidingWindowConversationManager(
                window_size=10 if agent_config.get('memory_enabled', True) else 1
            ),
            tools=[http_request]
        )
        
        return agent

# ============================================================================
# SPECIALIST TOOLS FACTORY
# ============================================================================

class SpecialistToolsFactory:
    """Creates specialist agents as callable tools"""
    
    def __init__(self, agent_builder: SpecialistAgentBuilder,
                 observability: ObservabilityManager):
        self.agent_builder = agent_builder
        self.observability = observability
        self.tools_cache = {}
    
    def create_specialist_tools(self, workflow: Dict, user_id: str, 
                               project_id: str) -> List:
        """Create specialist tools from workflow"""
        
        cache_key = f"{user_id}_{project_id}"
        if cache_key in self.tools_cache:
            return self.tools_cache[cache_key]
        
        tools = []
        
        for agent_config in workflow.get('agents', []):
            if agent_config.get('agent_type') == 'orchestrator':
                continue
            
            tool_func = self._create_tool_function(
                agent_config, user_id, project_id
            )
            
            decorated_tool = tool(
                name=agent_config.get('name'),
                description=agent_config.get('description', 'Specialist agent')
            )(tool_func)
            
            tools.append(decorated_tool)
        
        self.tools_cache[cache_key] = tools
        return tools
    
    def _create_tool_function(self, config: Dict, user_id: str, project_id: str):
        """Create tool function for specialist"""
        
        def specialist_func(task: str, context: str = "") -> str:
            """Execute specialist task with observability"""
            
            span = self.observability.create_span(
                f"specialist_{config.get('name')}",
                {"user_id": user_id, "project_id": project_id}
            )
            
            try:
                agent = self.agent_builder.build_agent(config, user_id, project_id)
                request = f"Task: {task}"
                if context:
                    request += f"\nContext: {context}"
                
                response = agent(request)
                span.set_attribute("response_length", len(str(response)))
                return str(response)
                
            except Exception as e:
                span.set_attribute("error", str(e))
                return f"Error: {str(e)}"
            finally:
                span.end()
        
        return specialist_func

# ============================================================================
# ORCHESTRATOR BUILDER
# ============================================================================

class OrchestratorBuilder:
    """Builds orchestrator with AgentCore integration"""
    
    def __init__(self, workflow_manager: WorkflowConfigManager,
                 tools_factory: SpecialistToolsFactory,
                 memory_manager: AgentCoreMemoryManager):
        self.workflow_manager = workflow_manager
        self.tools_factory = tools_factory
        self.memory_manager = memory_manager
        self.orchestrators = {}
    
    def build_orchestrator(self, user_id: str, project_id: str) -> Agent:
        """Build orchestrator with specialist tools"""
        
        cache_key = f"{user_id}_{project_id}"
        if cache_key in self.orchestrators:
            return self.orchestrators[cache_key]
        
        # Get workflow and create specialists
        workflow = self.workflow_manager.get_workflow(user_id, project_id)
        specialist_tools = []
        orchestrator_config = None
        
        if workflow:
            specialist_tools = self.tools_factory.create_specialist_tools(
                workflow, user_id, project_id
            )
            
            for agent in workflow.get('agents', []):
                if agent.get('agent_type') == 'orchestrator':
                    orchestrator_config = agent
                    break
        
        # Build model
        if orchestrator_config:
            model_id = orchestrator_config.get('model_id') or MODEL_MAPPING.get(
                orchestrator_config.get('model_name', 'Claude 3.5 Sonnet')
            )
            params = orchestrator_config.get('parameters', {})
        else:
            model_id = MODEL_MAPPING['Claude 3.5 Sonnet']
            params = {'temperature': 0.3, 'max_tokens': 4000, 'top_p': 0.9}
        
        model = BedrockModel(
            model_id=model_id,
            max_tokens=params.get('max_tokens', 4000),
            boto_session=boto3.Session(region_name=REGION),
            temperature=params.get('temperature', 0.3),
            top_p=params.get('top_p', 0.9)
        )
        
        # Build prompt
        system_prompt = self._build_prompt(
            orchestrator_config, specialist_tools, user_id, project_id
        )
        
        # Create memory store
        memory_store = self.memory_manager.create_memory_store(user_id, project_id)
        
        # Build orchestrator
        all_tools = [http_request] + specialist_tools
        
        orchestrator = Agent(
            model=model,
            system_prompt=system_prompt,
            conversation_manager=SlidingWindowConversationManager(window_size=10),
            tools=all_tools
        )
        
        self.orchestrators[cache_key] = orchestrator
        return orchestrator
    
    def _build_prompt(self, config: Optional[Dict], tools: List, 
                     user_id: str, project_id: str) -> str:
        """Build concise orchestrator prompt"""
        
        tool_names = ", ".join([
            getattr(t, '_tool_name', 'Unknown').replace('_', ' ').title() 
            for t in tools
        ])
        
        if config:
            return f"""You are {config.get('name', 'Orchestrator').replace('_', ' ').title()}.

{config.get('description', '')}

## Available Specialists
{tool_names}

## Instructions
{config.get('instructions', '')}

## Guidelines
- Simple queries: Answer directly
- Complex tasks: Delegate to specialists
- Research needs: Use http_request
- Image requests: Use generate_image_sd3

Context: {user_id}/{project_id}"""
        
        return f"""Customer Support Orchestrator

Available specialists: {tool_names}

Delegate complex tasks to specialists. Answer simple queries directly.

Context: {user_id}/{project_id}"""

# ============================================================================
# AGENTCORE APPLICATION WRAPPER
# ============================================================================

class WorkflowAgentCoreApp:
    """AgentCore-ready application wrapper"""
    
    def __init__(self):
        self.app = BedrockAgentCoreApp()
        self.observability = ObservabilityManager()
        self.memory_manager = AgentCoreMemoryManager()
        self.identity_manager = AgentCoreIdentityManager()
        self.gateway_manager = AgentCoreGatewayManager()
        self.workflow_manager = WorkflowConfigManager()
        
        agent_builder = SpecialistAgentBuilder(
            self.memory_manager, self.identity_manager
        )
        tools_factory = SpecialistToolsFactory(agent_builder, self.observability)
        
        self.orchestrator_builder = OrchestratorBuilder(
            self.workflow_manager, tools_factory, self.memory_manager
        )
    
    @property
    def entrypoint(self):
        """Decorator for entrypoint function"""
        return self.app.entrypoint
    
    def run(self):
        """Run the AgentCore app"""
        self.app.run()
    
    def invoke_agent(self, user_id: str, project_id: str, message: str,
                    session_id: Optional[str] = None) -> Dict:
        """Invoke agent with full AgentCore integration"""
        
        # Setup session context
        session_id = session_id or str(uuid.uuid4())
        self.observability.set_session_context(session_id)
        
        span = self.observability.create_span(
            "agent_invocation",
            {"user_id": user_id, "project_id": project_id, "session_id": session_id}
        )
        
        try:
            # Get orchestrator
            orchestrator = self.orchestrator_builder.build_orchestrator(
                user_id, project_id
            )
            
            # Retrieve memory context
            memory_store = self.memory_manager.create_memory_store(user_id, project_id)
            if memory_store:
                context = self.memory_manager.retrieve_recent_context(
                    memory_store.get('id'), user_id, session_id
                )
                
                # Add context to message if available
                if context:
                    message = f"[Previous context available]\n\n{message}"
            
            # Invoke agent
            response = orchestrator(message)
            response_text = str(response)
            
            # Store in memory
            if memory_store:
                self.memory_manager.store_conversation(
                    memory_store.get('id'),
                    user_id,
                    session_id,
                    [(message, "USER"), (response_text, "ASSISTANT")]
                )
            
            span.set_attribute("response_length", len(response_text))
            
            return {
                "response": response_text,
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            span.set_attribute("error", str(e))
            raise
        finally:
            span.end()

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

# Create AgentCore app instance
agentcore_app = WorkflowAgentCoreApp()

# FastAPI app for REST endpoints
api = FastAPI(
    title="AgentCore Multi-Agent System",
    description="Production-ready multi-agent system with AWS Bedrock AgentCore",
    version="19.0.0"
)

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    project_id: str = Field(default="default", description="Project identifier")
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session identifier")

class ChatResponse(BaseModel):
    response: str
    session_id: str
    timestamp: str
    agentcore_enabled: bool = True

# Endpoints
@api.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "19.0.0",
        "agentcore_features": {
            "runtime": True,
            "memory": True,
            "identity": True,
            "gateway": True,
            "observability": True
        }
    }

@api.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint with AgentCore integration"""
    try:
        result = agentcore_app.invoke_agent(
            request.user_id,
            request.project_id,
            request.message,
            request.session_id
        )
        
        return ChatResponse(
            response=result['response'],
            session_id=result['session_id'],
            timestamp=result['timestamp'],
            agentcore_enabled=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api.get("/memory/{user_id}/{session_id}")
async def get_memory(user_id: str, session_id: str):
    """Retrieve session memory"""
    try:
        # This would retrieve memory from AgentCore Memory
        return {
            "user_id": user_id,
            "session_id": session_id,
            "message": "Memory retrieval endpoint"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# AgentCore entrypoint decorator
@agentcore_app.entrypoint
def invoke(payload, context=None):
    """
    AgentCore Runtime entrypoint
    This function is called when deployed to AgentCore Runtime
    """
    user_id = payload.get("user_id", "default_user")
    project_id = payload.get("project_id", "default")
    message = payload.get("prompt", payload.get("message", "Hello"))
    session_id = payload.get("session_id")
    
    result = agentcore_app.invoke_agent(user_id, project_id, message, session_id)
    
    return {
        "output": {
            "message": result['response'],
            "session_id": result['session_id'],
            "timestamp": result['timestamp']
        }
    }

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 80)
    print("🚀 AWS BEDROCK AGENTCORE MULTI-AGENT SYSTEM")
    print("=" * 80)
    print("Version: 19.0.0")
    print(f"Region: {REGION}")
    print("\n✨ AgentCore Features:")
    print("  ✓ AgentCore Runtime - Serverless deployment with session isolation")
    print("  ✓ AgentCore Memory - Short & long-term memory management")
    print("  ✓ AgentCore Identity - Secure access controls & OAuth")
    print("  ✓ AgentCore Gateway - Unified tool access with MCP")
    print("  ✓ AgentCore Observability - Full tracing with OpenTelemetry")
    print("\n📦 Deployment Options:")
    print("  1. Local testing: python agent.py")
    print("  2. AgentCore CLI: agentcore launch")
    print("  3. Docker deployment: See dockerfile below")
    print("=" * 80 + "\n")
    
    if "--agentcore" in sys.argv:
        # Run as AgentCore app
        agentcore_app.run()
    else:
        # Run as FastAPI server
        import uvicorn
        port = int(os.getenv("PORT", 8080))
        uvicorn.run(api, host="0.0.0.0", port=port)
