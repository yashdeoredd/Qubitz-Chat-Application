"""
Setup observability for AgentCore agents
"""

import boto3

def enable_cloudwatch_transaction_search(region='us-east-1'):
    """Enable CloudWatch Transaction Search"""
    
    cloudwatch = boto3.client('cloudwatch', region_name=region)
    
    try:
        # Note: This is typically done via console
        # Programmatic enablement requires CloudWatch API
        print("📊 Enabling CloudWatch Transaction Search...")
        print("⚠️  Manual step required:")
        print("   1. Open CloudWatch Console")
        print("   2. Go to Application Signals > Transaction search")
        print("   3. Click 'Enable Transaction Search'")
        print("   4. Check 'Ingest spans as structured logs'")
        print("   5. Set X-Ray trace indexing to 1%")
        print("   6. Save")
        
        return True
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def create_cloudwatch_dashboard(agent_name, region='us-east-1'):
    """Create CloudWatch dashboard for agent monitoring"""
    
    cloudwatch = boto3.client('cloudwatch', region_name=region)
    
    dashboard_body = {
        "widgets": [
            {
                "type": "metric",
                "properties": {
                    "metrics": [
                        ["AWS/BedrockAgentCore", "Invocations", {"stat": "Sum"}],
                        [".", "Errors", {"stat": "Sum"}],
                        [".", "Latency", {"stat": "Average"}]
                    ],
                    "period": 300,
                    "stat": "Average",
                    "region": region,
                    "title": "Agent Performance"
                }
            },
            {
                "type": "metric",
                "properties": {
                    "metrics": [
                        ["AWS/BedrockAgentCore", "TokenUsage", {"stat": "Sum"}]
                    ],
                    "period": 300,
                    "stat": "Sum",
                    "region": region,
                    "title": "Token Usage"
                }
            }
        ]
    }
    
    try:
        cloudwatch.put_dashboard(
            DashboardName=f"{agent_name}-metrics",
            DashboardBody=json.dumps(dashboard_body)
        )
        print(f"✅ Dashboard created: {agent_name}-metrics")
    except Exception as e:
        print(f"❌ Dashboard creation failed: {str(e)}")

if __name__ == "__main__":
    enable_cloudwatch_transaction_search()
    create_cloudwatch_dashboard("workflow-multi-agent-system")
