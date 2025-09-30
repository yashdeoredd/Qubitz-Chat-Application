import sys
import os
import json
import time
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import boto3
from botocore.exceptions import ClientError
from pathlib import Path

# Disable SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

os.environ["BYPASS_TOOL_CONSENT"] = "true"

# ============================================================================
# LOGGING SYSTEM
# ============================================================================

class DetailedLogger:
    """Comprehensive logging system for agent interactions"""
    
    def __init__(self, log_dir: str = "agent_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log_file = self.log_dir / f"session_{timestamp}.txt"
        self.current_session_id = str(uuid.uuid4())[:8]
        self._write_session_header()
    
    def _write_session_header(self):
        header = f"""
{'=' * 80}
WORKFLOW AI AGENT FACTORY - SESSION LOG
{'=' * 80}
Session ID: {self.current_session_id}
Started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{'=' * 80}

"""
        with open(self.session_log_file, 'w', encoding='utf-8') as f:
            f.write(header)
    
    def log_interaction(self, interaction_id: str, user_id: str, project_id: str,
                       user_message: str, agent_name: str, system_prompt: str,
                       tool_calls: List[Dict], agent_response: str, processing_time: float,
                       workflow_found: bool, error: Optional[str] = None):
        
        log_entry = f"""
{'=' * 80}
INTERACTION LOG
{'=' * 80}
Interaction ID: {interaction_id}
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
User ID: {user_id}
Project ID: {project_id}
Agent Used: {agent_name}
Workflow Config Found: {workflow_found}
Processing Time: {processing_time:.2f}s

{'─' * 80}
USER MESSAGE:
{'─' * 80}
{user_message}

{'─' * 80}
SYSTEM PROMPT SENT TO AGENT:
{'─' * 80}
{system_prompt}

{'─' * 80}
TOOL CALLS MADE:
{'─' * 80}
"""
        if tool_calls:
            for i, call in enumerate(tool_calls, 1):
                log_entry += f"\nTool Call #{i}:\n"
                log_entry += f"  Tool Name: {call.get('name', 'Unknown')}\n"
                log_entry += f"  Arguments: {json.dumps(call.get('arguments', {}), indent=4)}\n"
                log_entry += f"  Result: {call.get('result', 'N/A')[:200]}...\n"
        else:
            log_entry += "No tool calls made\n"

        log_entry += f"""
{'─' * 80}
AGENT RESPONSE:
{'─' * 80}
{agent_response}
"""
        
        if error:
            log_entry += f"""
{'─' * 80}
ERROR DETAILS:
{'─' * 80}
{error}
"""
        
        log_entry += f"\n{'=' * 80}\n\n"
        
        with open(self.session_log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        
        interaction_file = self.log_dir / f"interaction_{interaction_id}.txt"
        with open(interaction_file, 'w', encoding='utf-8') as f:
            f.write(log_entry)
    
    def log_workflow_load(self, user_id: str, project_id: str, workflow_data: Optional[Dict], success: bool):
        log_entry = f"""
{'=' * 80}
WORKFLOW CONFIGURATION LOAD
{'=' * 80}
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
User ID: {user_id}
Project ID: {project_id}
Load Successful: {success}

{'─' * 80}
WORKFLOW DATA:
{'─' * 80}
"""
        if workflow_data:
            log_entry += json.dumps(workflow_data, indent=2)
        else:
            log_entry += "No workflow configuration found\n"
        
        log_entry += f"\n{'=' * 80}\n\n"
        
        with open(self.session_log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    
    def log_specialist_creation(self, agent_config: Dict, specialist_name: str):
        log_entry = f"""
{'=' * 80}
SPECIALIST AGENT CREATED
{'=' * 80}
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Specialist Name: {specialist_name}

{'─' * 80}
AGENT CONFIGURATION:
{'─' * 80}
{json.dumps(agent_config, indent=2)}

{'=' * 80}

"""
        with open(self.session_log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    
    def log_orchestrator_creation(self, orchestrator_config: Optional[Dict], workflow_found: bool):
        log_entry = f"""
{'=' * 80}
ORCHESTRATOR CREATED
{'=' * 80}
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Workflow Config Found: {workflow_found}

{'─' * 80}
ORCHESTRATOR CONFIGURATION:
{'─' * 80}
"""
        if orchestrator_config:
            log_entry += json.dumps(orchestrator_config, indent=2)
        else:
            log_entry += "Using default orchestrator configuration\n"
        
        log_entry += f"\n{'=' * 80}\n\n"
        
        with open(self.session_log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)

logger = DetailedLogger()

# Strands imports
try:
    from strands import Agent, tool
    from strands.models import BedrockModel
    from strands.agent.conversation_manager import SlidingWindowConversationManager
    from strands_tools import http_request
    STRANDS_AVAILABLE = True
except Exception as e:
    print(f"Strands not available: {e}")
    STRANDS_AVAILABLE = False

# ============================================================================
# IMAGE GENERATION TOOL
# ============================================================================

@tool(
    name="generate_image_sd3",
    description="Generate high-quality images using Stable Diffusion 3 Large model when users request image creation."
)
def generate_image_sd3(
    prompt: str,
    user_id: str,
    project_id: str,
    aspect_ratio: str = "1:1",
    negative_prompt: str = "",
    seed: int = 0,
    output_format: str = "PNG"
) -> str:
    """Generate images using Stable Diffusion 3 Large model and upload to S3."""
    generation_id = str(uuid.uuid4())[:8]
    
    tool_log = f"""
Tool Called: generate_image_sd3
Generation ID: {generation_id}
Prompt: {prompt}
User: {user_id}/{project_id}
Parameters: aspect_ratio={aspect_ratio}, seed={seed}, format={output_format}
"""
    with open(logger.session_log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n[TOOL CALL - {datetime.now().strftime('%H:%M:%S')}]\n{tool_log}\n")
    
    try:
        bedrock_client = boto3.client('bedrock-runtime', region_name='us-west-2')
        s3_client = boto3.client('s3', region_name='us-east-1')
        
        request_body = {
            'prompt': prompt,
            'aspect_ratio': aspect_ratio,
            'mode': 'text-to-image',
            'output_format': output_format
        }
        
        if negative_prompt:
            request_body['negative_prompt'] = negative_prompt
        if seed > 0:
            request_body['seed'] = seed
        
        response = bedrock_client.invoke_model(
            modelId='stability.sd3-large-v1:0',
            body=json.dumps(request_body)
        )
        
        output_body = json.loads(response["body"].read().decode("utf-8"))
        
        if output_body.get("finish_reasons") and output_body["finish_reasons"][0] is not None:
            error_reason = output_body["finish_reasons"][0]
            result = f"❌ Image generation failed: {error_reason}"
            with open(logger.session_log_file, 'a', encoding='utf-8') as f:
                f.write(f"[TOOL RESULT] {result}\n")
            return result
        
        base64_image = output_body["images"][0]
        import base64
        image_data = base64.b64decode(base64_image)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"sd3_generated_{timestamp}_{uuid.uuid4().hex[:8]}.{output_format.lower()}"
        
        bucket_name = 'qubitz-customer-prod'
        s3_key = f"{user_id}/{project_id}/generated_images/{image_filename}"
        
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=image_data,
            ContentType=f'image/{output_format.lower()}',
            Metadata={
                'generated_by': 'stable_diffusion_3_large',
                'prompt': prompt[:200],
                'user_id': user_id,
                'project_id': project_id,
                'generation_timestamp': timestamp
            }
        )
        
        s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"
        
        result = f"""✅ **Image Generated Successfully!**

🎨 **Prompt**: {prompt}
🔗 **Image URL**: {s3_url}
⏱️ **Generated**: {timestamp}

The image has been uploaded to S3 and is ready for use!"""
        
        with open(logger.session_log_file, 'a', encoding='utf-8') as f:
            f.write(f"[TOOL RESULT] Image generated successfully: {s3_url}\n")
        
        return result
        
    except Exception as e:
        error_msg = f"❌ Image generation failed: {str(e)}"
        with open(logger.session_log_file, 'a', encoding='utf-8') as f:
            f.write(f"[TOOL ERROR] {error_msg}\n")
        return error_msg

# ============================================================================
# WORKFLOW-BASED AGENT TOOLS IMPLEMENTATION
# ============================================================================

class WorkflowAgentManager:
    """Manages workflow-based specialist agents as tools"""
    
    def __init__(self):
        self.s3_client = boto3.client('s3', region_name='us-east-1')
        self.agent_tools_cache = {}
        self.workflow_cache = {}
        
        self.model_mapping = {
            "Claude 3.5 Sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "Claude 3.5 Haiku": "anthropic.claude-3-haiku-20240307-v1:0", 
            "Amazon Nova Micro": "amazon.nova-micro-v1:0",
            "Amazon Nova Lite": "amazon.nova-lite-v1:0",
            "Amazon Nova Pro": "amazon.nova-pro-v1:0",
            "Command R": "cohere.command-r-v1:0",
            "DeepSeek-R1": "us.deepseek.r1-v1:0",
            "Claude 3.5 Sonnet v2": "us.anthropic.claude-sonnet-4-20250514-v1:0"
        }
    
    def get_workflow_config(self, user_id: str, project_id: str) -> Optional[Dict]:
        """Get workflow configuration from S3"""
        cache_key = f"{user_id}_{project_id}"
        
        if cache_key in self.workflow_cache:
            logger.log_workflow_load(user_id, project_id, self.workflow_cache[cache_key], True)
            return self.workflow_cache[cache_key]
        
        try:
            bucket_name = 'qubitz-customer-prod'
            key = f'{user_id}/{project_id}/workflow.json'
            
            response = self.s3_client.get_object(Bucket=bucket_name, Key=key)
            workflow_data = json.loads(response['Body'].read().decode('utf-8'))
            
            self.workflow_cache[cache_key] = workflow_data
            logger.log_workflow_load(user_id, project_id, workflow_data, True)
            return workflow_data
            
        except Exception as e:
            print(f"Could not load workflow config for {user_id}/{project_id}: {e}")
            logger.log_workflow_load(user_id, project_id, None, False)
            return None
    
    def create_specialist_agent(self, agent_config: Dict, user_id: str, project_id: str) -> Agent:
        """Create a specialist agent from workflow configuration"""
        
        # Get model from config
        model_name = agent_config.get('model_name', 'Claude 3.5 Sonnet')
        model_id = agent_config.get('model_id')
        
        if not model_id:
            model_id = self.model_mapping.get(model_name, "anthropic.claude-3-5-sonnet-20240620-v1:0")
        
        region_id = agent_config.get('region_id', 'us-east-1')
        params = agent_config.get('parameters', {})
        
        model = BedrockModel(
            model_id=model_id,
            max_tokens=params.get('max_tokens', 4000),
            boto_session=boto3.Session(region_name=region_id),
            temperature=params.get('temperature', 0.7),
            top_p=params.get('top_p', 0.9)
        )
        
        # CONCISE specialist prompt
        agent_name = agent_config.get('name', 'Specialist')
        description = agent_config.get('description', 'A specialized support agent')
        instructions = agent_config.get('instructions', '')
        
        system_prompt = f"""You are {agent_name.replace('_', ' ').title()}.

{description}

## Instructions
{instructions}

Context: {user_id}/{project_id}

Provide expert assistance within your domain."""

        logger.log_specialist_creation(agent_config, agent_name)

        agent = Agent(
            model=model,
            system_prompt=system_prompt,
            conversation_manager=SlidingWindowConversationManager(
                window_size=10 if agent_config.get('memory_enabled', True) else 1
            ),
            tools=[http_request]
        )
        
        return agent
    
    def create_agent_tools_from_workflow(self, user_id: str, project_id: str) -> Tuple[List, Optional[Dict]]:
        """Create specialist agent tools from workflow"""
        
        cache_key = f"{user_id}_{project_id}"
        if cache_key in self.agent_tools_cache:
            return self.agent_tools_cache[cache_key]
        
        workflow_config = self.get_workflow_config(user_id, project_id)
        if not workflow_config or 'agents' not in workflow_config:
            return [], None
        
        agent_tools = []
        orchestrator_config = None
        
        for agent_config in workflow_config['agents']:
            agent_name = agent_config.get('name', 'unknown_agent')
            agent_type = agent_config.get('agent_type', 'specialist')
            
            # Handle orchestrator separately
            if agent_type == 'orchestrator':
                orchestrator_config = agent_config
                continue
            
            # Create specialist tool
            def create_tool_for_agent(config, uid, pid):
                def specialist_tool_func(task: str, context: str = "", priority: str = "normal") -> str:
                    """Call the specialist agent for domain-specific tasks"""
                    tool_call_id = str(uuid.uuid4())[:8]
                    
                    with open(logger.session_log_file, 'a', encoding='utf-8') as f:
                        f.write(f"\n[SPECIALIST TOOL CALL - {datetime.now().strftime('%H:%M:%S')}]\n")
                        f.write(f"Specialist: {config.get('name')}\n")
                        f.write(f"Task: {task}\n")
                        f.write(f"Context: {context}\n")
                        f.write(f"Priority: {priority}\n")
                    
                    try:
                        specialist = self.create_specialist_agent(config, uid, pid)
                        
                        request_parts = [f"Task: {task}"]
                        if context:
                            request_parts.append(f"Context: {context}")
                        if priority != "normal":
                            request_parts.append(f"Priority: {priority}")
                        
                        full_request = "\n\n".join(request_parts)
                        response = specialist(full_request)
                        
                        with open(logger.session_log_file, 'a', encoding='utf-8') as f:
                            f.write(f"[SPECIALIST RESPONSE] {str(response)[:500]}...\n")
                        
                        return str(response)
                        
                    except Exception as e:
                        error_msg = f"Error in {config.get('name', 'specialist')}: {str(e)}"
                        with open(logger.session_log_file, 'a', encoding='utf-8') as f:
                            f.write(f"[SPECIALIST ERROR] {error_msg}\n")
                        return error_msg
                
                return specialist_tool_func
            
            tool_func = create_tool_for_agent(agent_config, user_id, project_id)
            
            # Apply @tool decorator
            decorated_tool = tool(
                name=agent_name,
                description=f"{agent_config.get('description', 'Specialist for domain tasks')}"
            )(tool_func)
            
            agent_tools.append(decorated_tool)
        
        result = (agent_tools, orchestrator_config)
        self.agent_tools_cache[cache_key] = result
        return result

# ============================================================================
# ORCHESTRATOR FACTORY 
# ============================================================================

class OrchestratorFactory:
    """Creates orchestrator from workflow configuration"""
    
    def __init__(self):
        self.workflow_manager = WorkflowAgentManager()
        self.orchestrator_cache = {}
    
    def create_orchestrator(self, user_id: str, project_id: str) -> Tuple[Agent, bool, str]:
        """Create orchestrator using workflow configuration"""
        
        cache_key = f"orchestrator_{user_id}_{project_id}"
        if cache_key in self.orchestrator_cache:
            cached_agent, workflow_found, system_prompt = self.orchestrator_cache[cache_key]
            return cached_agent, workflow_found, system_prompt
        
        # Get specialist tools and orchestrator config
        specialist_tools, orchestrator_config = self.workflow_manager.create_agent_tools_from_workflow(user_id, project_id)
        
        workflow_found = orchestrator_config is not None
        
        if orchestrator_config:
            print(f"Creating orchestrator from config: {orchestrator_config.get('name')}")
            logger.log_orchestrator_creation(orchestrator_config, True)
            
            # Use orchestrator's model configuration
            model_name = orchestrator_config.get('model_name', 'Claude 3.5 Sonnet')
            model_id = orchestrator_config.get('model_id')
            
            if not model_id:
                model_id = self.workflow_manager.model_mapping.get(model_name, "anthropic.claude-3-5-sonnet-20240620-v1:0")
            
            region_id = orchestrator_config.get('region_id', 'us-east-1')
            params = orchestrator_config.get('parameters', {})
            
            model = BedrockModel(
                model_id=model_id,
                max_tokens=params.get('max_tokens', 4000),
                boto_session=boto3.Session(region_name=region_id),
                temperature=params.get('temperature', 0.3),
                top_p=params.get('top_p', 0.9)
            )
            
            system_prompt = self.build_orchestrator_prompt(orchestrator_config, specialist_tools, user_id, project_id)
            
        else:
            logger.log_orchestrator_creation(None, False)
            
            model = BedrockModel(
                model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
                max_tokens=4000,
                boto_session=boto3.Session(region_name='us-east-1'),
                temperature=0.3,
                top_p=0.9
            )
            
            system_prompt = self.build_default_prompt(specialist_tools, user_id, project_id)
        
        # Create orchestrator with tools
        all_tools = [http_request, generate_image_sd3] + specialist_tools
        
        orchestrator = Agent(
            model=model,
            system_prompt=system_prompt,
            conversation_manager=SlidingWindowConversationManager(window_size=10),
            tools=all_tools
        )
        
        result = (orchestrator, workflow_found, system_prompt)
        self.orchestrator_cache[cache_key] = result
        return result
    
    def build_orchestrator_prompt(self, orchestrator_config: Dict, specialist_tools: List, user_id: str, project_id: str) -> str:
        """Build CONCISE orchestrator prompt from workflow configuration"""
        
        name = orchestrator_config.get('name', 'orchestrator').replace('_', ' ').title()
        description = orchestrator_config.get('description', '')
        instructions = orchestrator_config.get('instructions', '')
        
        # Build concise specialist list
        specialist_names = [getattr(tool, '_tool_name', 'Unknown').replace('_', ' ').title() 
                          for tool in specialist_tools]
        specialists_list = ", ".join(specialist_names) if specialist_names else "None"
        
        # STREAMLINED PROMPT
        system_prompt = f"""You are {name}, a customer support orchestrator.

## Role
{description}

## Your Team
Available specialists: {specialists_list}

## Instructions
{instructions}

## Tool Usage
- Simple questions: Answer directly
- Specialist tasks: Delegate to appropriate team member
- Web research: Use http_request only when needed
- Image requests: Use generate_image_sd3

Context: {user_id}/{project_id}"""

        return system_prompt
    
    def build_default_prompt(self, specialist_tools: List, user_id: str, project_id: str) -> str:
        """Build CONCISE default orchestrator prompt"""
        
        specialist_names = [getattr(tool, '_tool_name', 'Unknown').replace('_', ' ').title() 
                          for tool in specialist_tools]
        specialists_list = ", ".join(specialist_names) if specialist_names else "None"
        
        return f"""You are a Customer Support Orchestrator.

Available specialists: {specialists_list}

## Guidelines
- Answer simple questions directly
- Delegate complex tasks to specialists
- Use http_request for web research when needed
- Use generate_image_sd3 for image requests

Context: {user_id}/{project_id}"""

# ============================================================================
# SESSION MANAGER
# ============================================================================

class SessionManager:
    def __init__(self):
        self.sessions = {}
        
    def get_recent_history(self, user_id: str, project_id: str, count: int = 2) -> List[Dict]:
        return []
        
    def add_interaction(self, user_id: str, project_id: str, message: str, response: str, agent_name: str):
        pass

# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

class WorkflowOrchestrator:
    """Main orchestrator using workflow configuration"""
    
    def __init__(self):
        self.orchestrator_factory = OrchestratorFactory()
        self.session_manager = SessionManager()
        
    def process_request(self, user_id: str, project_id: str, message: str, context: Optional[Dict] = None) -> Tuple[str, float, str, bool]:
        """Process request with workflow orchestrator"""
        
        interaction_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        tool_calls = []
        
        try:
            # Get orchestrator with specialists
            orchestrator, workflow_found, system_prompt = self.orchestrator_factory.create_orchestrator(user_id, project_id)
            agent_name = "workflow_orchestrator" if workflow_found else "default_orchestrator"
            
            print(f"Using {'workflow' if workflow_found else 'default'} orchestrator")
            print(f"Processing: {message[:50]}...")
            
            # Process the message
            response = orchestrator(message)
            response_text = str(response)
            
            print(f"Response generated, length: {len(response_text)}")
            
            processing_time = time.time() - start_time
            
            # Log the complete interaction
            logger.log_interaction(
                interaction_id=interaction_id,
                user_id=user_id,
                project_id=project_id,
                user_message=message,
                agent_name=agent_name,
                system_prompt=system_prompt,
                tool_calls=tool_calls,
                agent_response=response_text,
                processing_time=processing_time,
                workflow_found=workflow_found
            )
            
            return response_text, processing_time, agent_name, workflow_found
            
        except Exception as e:
            print(f"Error in orchestrator: {e}")
            processing_time = time.time() - start_time
            error_response = f"I apologize for the error: {str(e)}"
            
            # Log error interaction
            logger.log_interaction(
                interaction_id=interaction_id,
                user_id=user_id,
                project_id=project_id,
                user_message=message,
                agent_name="error",
                system_prompt="N/A",
                tool_calls=[],
                agent_response=error_response,
                processing_time=processing_time,
                workflow_found=False,
                error=str(e)
            )
            
            return error_response, processing_time, "error", False

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Workflow AI Agent Factory",
    description="AI agents created from workflow configuration with comprehensive logging",
    version="18.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class WorkflowRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    project_id: str = Field(default="default", description="Project identifier")
    message: str = Field(..., description="User message or query")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")

class WorkflowResponse(BaseModel):
    response: str
    processing_time: float
    agent_used: str
    workflow_config_found: bool
    interaction_id: str
    log_file: str

# Initialize
main_orchestrator = WorkflowOrchestrator()

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "Workflow AI Agent Factory with streamlined prompts",
        "version": "18.0.0",
        "features": {
            "workflow_based_agents": True,
            "specialist_tools": True,
            "concise_prompts": True,
            "comprehensive_logging": True,
            "log_directory": str(logger.log_dir),
            "session_log": str(logger.session_log_file)
        }
    }

@app.post('/workflow/chat', response_model=WorkflowResponse)  
async def workflow_chat(request: WorkflowRequest):
    try:
        interaction_id = str(uuid.uuid4())[:8]
        
        response_text, processing_time, agent_used, workflow_found = main_orchestrator.process_request(
            request.user_id, request.project_id, request.message, request.context
        )
        
        return WorkflowResponse(
            response=response_text,
            processing_time=processing_time,
            agent_used=agent_used,
            workflow_config_found=workflow_found,
            interaction_id=interaction_id,
            log_file=str(logger.session_log_file)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get('/workflow/agents/{user_id}/{project_id}')
async def get_workflow_agents(user_id: str, project_id: str):
    try:
        manager = WorkflowAgentManager()
        workflow_config = manager.get_workflow_config(user_id, project_id)
        
        if not workflow_config:
            return {
                "message": "No workflow configuration found",
                "workflow_found": False
            }
        
        agents_info = []
        orchestrator_info = None
        
        for agent in workflow_config['agents']:
            info = {
                "name": agent.get('name'),
                "type": agent.get('agent_type'),
                "description": agent.get('description'),
                "model": agent.get('model_name')
            }
            
            if agent.get('agent_type') == 'orchestrator':
                orchestrator_info = info
            else:
                agents_info.append(info)
        
        return {
            "workflow_found": True,
            "orchestrator": orchestrator_info,
            "specialists": agents_info,
            "total_agents": len(workflow_config['agents'])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get('/logs/session')
async def get_session_log():
    """Get the current session log file path and contents"""
    try:
        with open(logger.session_log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            "log_file": str(logger.session_log_file),
            "session_id": logger.current_session_id,
            "content": content,
            "size_bytes": len(content)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading log: {str(e)}")

@app.get('/logs/interactions')
async def list_interactions():
    """List all interaction log files"""
    try:
        interaction_files = list(logger.log_dir.glob("interaction_*.txt"))
        
        interactions = []
        for file in sorted(interaction_files, reverse=True):
            stat = file.stat()
            interactions.append({
                "filename": file.name,
                "path": str(file),
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            })
        
        return {
            "total_interactions": len(interactions),
            "interactions": interactions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing interactions: {str(e)}")

@app.get('/logs/interaction/{interaction_id}')
async def get_interaction_log(interaction_id: str):
    """Get a specific interaction log"""
    try:
        log_file = logger.log_dir / f"interaction_{interaction_id}.txt"
        
        if not log_file.exists():
            raise HTTPException(status_code=404, detail="Interaction log not found")
        
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            "interaction_id": interaction_id,
            "log_file": str(log_file),
            "content": content
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading interaction log: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 80)
    print("🚀 WORKFLOW AI AGENT FACTORY - STREAMLINED VERSION")
    print("=" * 80)
    print(f"📁 Log Directory: {logger.log_dir}")
    print(f"📋 Session Log: {logger.session_log_file}")
    print(f"🆔 Session ID: {logger.current_session_id}")
    print("=" * 80)
    
    # Create test workflow
    test_workflow = {
        "agents": [
            {
                "name": "customer_intake_orchestrator",
                "description": "Primary orchestrator managing customer inquiries and routing to specialists",
                "agent_type": "orchestrator",
                "model_name": "Claude 3.5 Sonnet",
                "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                "region_id": "us-east-1",
                "parameters": {
                    "temperature": 0.3,
                    "max_tokens": 4000,
                    "top_p": 0.9
                },
                "instructions": "Manage customer inquiries, analyze sentiment, route to specialists, and coordinate support workflows.",
                "memory_enabled": True
            },
            {
                "name": "technical_support_specialist", 
                "description": "Technical support specialist for troubleshooting and technical issues",
                "agent_type": "specialist",
                "model_name": "Claude 3.5 Sonnet v2",
                "model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0",
                "region_id": "us-east-1",
                "parameters": {"temperature": 0.2, "max_tokens": 6000, "top_p": 0.8},
                "instructions": "Handle technical problems, debug issues, and provide comprehensive technical solutions.",
                "memory_enabled": True
            },
            {
                "name": "multilingual_communication_agent",
                "description": "Multilingual specialist for translation and international support",
                "agent_type": "communication", 
                "model_name": "Command R",
                "model_id": "cohere.command-r-v1:0",
                "region_id": "us-east-1",
                "parameters": {"temperature": 0.4, "max_tokens": 3000, "top_p": 0.9},
                "instructions": "Handle translations and multilingual customer communications.",
                "memory_enabled": True
            }
        ]
    }
    
    # Save test workflow
    try:
        s3_client = boto3.client('s3', region_name='us-east-1')
        s3_client.put_object(
            Bucket='qubitz-customer-prod',
            Key='test_user_123/test_project_456/workflow.json',
            Body=json.dumps(test_workflow, indent=2),
            ContentType='application/json'
        )
        print("✅ Test workflow saved to S3")
        logger.log_workflow_load("test_user_123", "test_project_456", test_workflow, True)
    except Exception as e:
        print(f"⚠️  Could not save test workflow: {e}")
    
    print("\n✨ KEY IMPROVEMENTS:")
    print("  • Agents ARE tools (not just prompt text)")
    print("  • Each specialist is a separate Agent instance")
    print("  • Concise, focused prompts (reduced by ~60%)")
    print("  • Removed redundant information")
    print("  • Streamlined system instructions")
    print("\n📊 LOGGING FEATURES:")
    print("  • Complete interaction tracking")
    print("  • Tool call monitoring")
    print("  • Agent response logging")
    print("  • Error tracking")
    print("\n🔗 API ENDPOINTS:")
    print("  • POST /workflow/chat - Chat with agent")
    print("  • GET /logs/session - View session log")
    print("  • GET /logs/interactions - List interactions")
    print("  • GET /logs/interaction/{id} - View specific interaction")
    print("  • GET /workflow/agents/{user}/{project} - View agents")
    print("=" * 80 + "\n")
    
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False, log_level="info")
