"""
Script to invoke deployed AgentCore agent
"""

import boto3
import json
import sys
from pathlib import Path

def load_deployment_info():
    """Load deployment info from file"""
    
    if not Path('deployment_info.json').exists():
        print("❌ deployment_info.json not found. Run deploy_to_agentcore.py first.")
        sys.exit(1)
    
    with open('deployment_info.json', 'r') as f:
        return json.load(f)

def invoke_agent(agent_arn, user_id, project_id, message, session_id=None, region='us-east-1'):
    """Invoke AgentCore agent"""
    
    client = boto3.client('bedrock-agentcore', region_name=region)
    
    # Generate session ID if not provided
    if not session_id:
        import uuid
        session_id = str(uuid.uuid4())
    
    # Ensure session ID is at least 33 characters
    if len(session_id) < 33:
        session_id = session_id + 'x' * (33 - len(session_id))
    
    payload = json.dumps({
        'user_id': user_id,
        'project_id': project_id,
        'prompt': message,
        'session_id': session_id
    })
    
    print(f"\n🔄 Invoking agent...")
    print(f"User: {user_id}/{project_id}")
    print(f"Session: {session_id}")
    print(f"Message: {message[:100]}...")
    
    try:
        response = client.invoke_agent_runtime(
            agentRuntimeArn=agent_arn,
            runtimeSessionId=session_id,
            payload=payload.encode(),
            qualifier='DEFAULT'
        )
        
        response_body = response['response'].read()
        response_data = json.loads(response_body)
        
        print(f"\n✅ Response received:")
        print(json.dumps(response_data, indent=2))
        
        return response_data
        
    except Exception as e:
        print(f"❌ Invocation failed: {str(e)}")
        sys.exit(1)

def main():
    """Main invocation workflow"""
    
    print("\n" + "=" * 80)
    print("🤖 INVOKE AGENTCORE AGENT")
    print("=" * 80)
    
    # Load deployment info
    deployment_info = load_deployment_info()
    
    print(f"\n📋 Deployment Info:")
    print(f"   Agent ARN: {deployment_info['agent_runtime_arn']}")
    print(f"   Region: {deployment_info['region']}")
    
    # Get inputs
    user_id = input("\nEnter user_id [test_user]: ").strip() or 'test_user'
    project_id = input("Enter project_id [default]: ").strip() or 'default'
    message = input("Enter message: ").strip()
    
    if not message:
        print("❌ Message cannot be empty")
        sys.exit(1)
    
    # Invoke agent
    invoke_agent(
        deployment_info['agent_runtime_arn'],
        user_id,
        project_id,
        message,
        region=deployment_info['region']
    )

if __name__ == "__main__":
    main()
