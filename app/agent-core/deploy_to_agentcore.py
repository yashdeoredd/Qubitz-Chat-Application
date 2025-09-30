"""
Deployment script for AWS Bedrock AgentCore
"""

import boto3
import json
import sys
from pathlib import Path

def create_iam_role():
    """Create IAM role for AgentCore Runtime"""
    
    iam_client = boto3.client('iam')
    
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "bedrock-agentcore.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }
    
    policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream"
                ],
                "Resource": "arn:aws:bedrock:*::foundation-model/*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:PutObject"
                ],
                "Resource": "arn:aws:s3:::qubitz-customer-prod/*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents"
                ],
                "Resource": "arn:aws:logs:*:*:log-group:/aws/bedrock-agentcore/*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "cloudwatch:PutMetricData"
                ],
                "Resource": "*"
            }
        ]
    }
    
    try:
        # Create role
        role_response = iam_client.create_role(
            RoleName='AgentCoreWorkflowRuntimeRole',
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description='IAM role for AgentCore Workflow Runtime'
        )
        
        # Attach inline policy
        iam_client.put_role_policy(
            RoleName='AgentCoreWorkflowRuntimeRole',
            PolicyName='AgentCoreWorkflowPolicy',
            PolicyDocument=json.dumps(policy_document)
        )
        
        print(f"✅ IAM Role created: {role_response['Role']['Arn']}")
        return role_response['Role']['Arn']
        
    except iam_client.exceptions.EntityAlreadyExistsException:
        role = iam_client.get_role(RoleName='AgentCoreWorkflowRuntimeRole')
        print(f"ℹ️  IAM Role already exists: {role['Role']['Arn']}")
        return role['Role']['Arn']

def build_and_push_image(region='us-east-1', account_id=None):
    """Build ARM64 image and push to ECR"""
    
    import subprocess
    
    if not account_id:
        sts = boto3.client('sts')
        account_id = sts.get_caller_identity()['Account']
    
    ecr_client = boto3.client('ecr', region_name=region)
    repository_name = 'agentcore-workflow-agents'
    
    # Create ECR repository if it doesn't exist
    try:
        repo = ecr_client.create_repository(
            repositoryName=repository_name,
            imageTagMutability='MUTABLE',
            imageScanningConfiguration={'scanOnPush': True}
        )
        print(f"✅ ECR Repository created: {repository_name}")
    except ecr_client.exceptions.RepositoryAlreadyExistsException:
        print(f"ℹ️  ECR Repository already exists: {repository_name}")
    
    # Get ECR login
    print("\n📦 Building and pushing Docker image...")
    
    ecr_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repository_name}"
    
    commands = [
        # ECR login
        f"aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {account_id}.dkr.ecr.{region}.amazonaws.com",
        
        # Build ARM64 image
        f"docker buildx build --platform linux/arm64 -t {ecr_uri}:latest --push .",
    ]
    
    for cmd in commands:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Command failed: {cmd}")
            print(f"Error: {result.stderr}")
            sys.exit(1)
    
    print(f"✅ Image pushed to: {ecr_uri}:latest")
    return f"{ecr_uri}:latest"

def deploy_to_agentcore(image_uri, role_arn, region='us-east-1'):
    """Deploy agent to AgentCore Runtime"""
    
    client = boto3.client('bedrock-agentcore-control', region_name=region)
    
    try:
        response = client.create_agent_runtime(
            agentRuntimeName='workflow-multi-agent-system',
            agentRuntimeArtifact={
                'containerConfiguration': {
                    'containerUri': image_uri
                }
            },
            networkConfiguration={
                'networkMode': 'PUBLIC'  # or 'SANDBOX' for restricted access
            },
            roleArn=role_arn,
            description='Production multi-agent system with workflow orchestration'
        )
        
        print(f"\n✅ Agent Runtime created successfully!")
        print(f"Agent Runtime ARN: {response['agentRuntimeArn']}")
        print(f"Status: {response['status']}")
        
        return response
        
    except Exception as e:
        print(f"❌ Deployment failed: {str(e)}")
        sys.exit(1)

def enable_observability(region='us-east-1'):
    """Enable CloudWatch Transaction Search for observability"""
    
    cloudwatch = boto3.client('cloudwatch', region_name=region)
    
    print("\n📊 Enabling observability...")
    print("⚠️  Please manually enable CloudWatch Transaction Search:")
    print("   1. Go to CloudWatch Console")
    print("   2. Navigate to Application Signals > Transaction search")
    print("   3. Click 'Enable Transaction Search'")
    print("   4. Enable 'Ingest spans as structured logs'")
    print("   5. Save")

def main():
    """Main deployment workflow"""
    
    print("\n" + "=" * 80)
    print("🚀 AWS BEDROCK AGENTCORE DEPLOYMENT")
    print("=" * 80)
    
    region = input("Enter AWS region [us-east-1]: ").strip() or 'us-east-1'
    
    print("\n📋 Deployment Steps:")
    print("  1. Create IAM Role")
    print("  2. Build & Push Docker Image")
    print("  3. Deploy to AgentCore Runtime")
    print("  4. Enable Observability")
    
    proceed = input("\nProceed? (yes/no): ").strip().lower()
    if proceed != 'yes':
        print("Deployment cancelled.")
        sys.exit(0)
    
    # Step 1: Create IAM Role
    print("\n" + "=" * 80)
    print("STEP 1: Creating IAM Role")
    print("=" * 80)
    role_arn = create_iam_role()
    
    # Step 2: Build and push image
    print("\n" + "=" * 80)
    print("STEP 2: Building and Pushing Docker Image")
    print("=" * 80)
    image_uri = build_and_push_image(region)
    
    # Step 3: Deploy to AgentCore
    print("\n" + "=" * 80)
    print("STEP 3: Deploying to AgentCore Runtime")
    print("=" * 80)
    deployment = deploy_to_agentcore(image_uri, role_arn, region)
    
    # Step 4: Observability setup
    print("\n" + "=" * 80)
    print("STEP 4: Observability Setup")
    print("=" * 80)
    enable_observability(region)
    
    # Save deployment info
    deployment_info = {
        'agent_runtime_arn': deployment['agentRuntimeArn'],
        'image_uri': image_uri,
        'role_arn': role_arn,
        'region': region,
        'deployed_at': str(boto3.client('sts').get_caller_identity()['Account'])
    }
    
    with open('deployment_info.json', 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    print("\n" + "=" * 80)
    print("✅ DEPLOYMENT COMPLETE")
    print("=" * 80)
    print(f"\n📝 Deployment info saved to: deployment_info.json")
    print(f"\n🔗 Agent Runtime ARN:")
    print(f"   {deployment['agentRuntimeArn']}")
    print(f"\n📚 Next Steps:")
    print(f"   1. Test with: python invoke_agent.py")
    print(f"   2. View logs in CloudWatch")
    print(f"   3. Monitor metrics in AgentCore Console")

if __name__ == "__main__":
    main()
