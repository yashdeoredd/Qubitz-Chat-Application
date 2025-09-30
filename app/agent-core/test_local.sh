#!/bin/bash

echo "================================"
echo "Local AgentCore Testing"
echo "================================"

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Configure AgentCore
echo "🔧 Configuring AgentCore..."
agentcore configure --entrypoint agent.py -er arn:aws:iam::YOUR_ACCOUNT:role/AgentCoreWorkflowRuntimeRole

# Launch locally
echo "🚀 Launching agent locally..."
agentcore launch --local &

# Wait for startup
sleep 5

# Test invocation
echo "🧪 Testing invocation..."
agentcore invoke --local '{
  "user_id": "test_user",
  "project_id": "test_project",
  "prompt": "Hello, introduce yourself and your capabilities"
}'

echo "✅ Local testing complete"
