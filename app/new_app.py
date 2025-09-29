from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Tuple
import uvicorn
import boto3
import json
import time
import os
from datetime import datetime
import re
import PyPDF2
import docx
from io import BytesIO
from botocore.exceptions import ClientError
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import lru_cache
import uuid
import urllib3
from PIL import Image
import base64
import csv
import hashlib
from googlesearch import search
from bs4 import BeautifulSoup
import requests

# Disable SSL warnings (for development)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

os.environ["BYPASS_TOOL_CONSENT"] = "true"

# Strands imports with fallback
try:
    from strands import Agent, tool
    from strands.models import BedrockModel
    from strands.agent.conversation_manager import SlidingWindowConversationManager
    
    STRANDS_AVAILABLE = True
except Exception as e:
    # Stub implementations for testing
    class Agent:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            return "Agent response (stub)"
        async def stream_async(self, *args, **kwargs):
            yield {"data": "Streaming response (stub)"}
    
    class BedrockModel:
        def __init__(self, *args, **kwargs):
            pass
    
    class SlidingWindowConversationManager:
        def __init__(self, *args, **kwargs):
            pass
    
    def tool(name=None, description=None):
        def decorator(func):
            func._tool_name = name or func.__name__
            func._tool_description = description or func.__doc__
            return func
        return decorator
    
    STRANDS_AVAILABLE = False

# ============================================================================
# INTERNET SEARCH TOOLS
# ============================================================================

@tool(
    name="search_internet",
    description="Search the internet using Google and extract content from web pages"
)
def search_internet(query: str, num_results: int = 5) -> str:
    """
    Search the internet for information and extract content from results.
    
    Args:
        query: Search query to find information about
        num_results: Number of search results to process (default 5)
    
    Returns:
        String with search results and extracted content
    """
    search_id = str(uuid.uuid4())[:8]
    
    try:
        # Perform Google search
        search_results = []
        urls_processed = []
        
        for url in search(query, num_results=num_results, stop=num_results, pause=2):
            if len(urls_processed) >= num_results:
                break
                
            try:
                # Extract content from webpage
                response = requests.get(url, timeout=10, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                })
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract title
                title = soup.find('title')
                title_text = title.get_text().strip() if title else "No title"
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Extract text content
                text_content = soup.get_text()
                
                # Clean up text
                lines = (line.strip() for line in text_content.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text_content = ' '.join(chunk for chunk in chunks if chunk)
                
                # Limit content length
                if len(text_content) > 1000:
                    text_content = text_content[:1000] + "..."
                
                search_results.append({
                    'title': title_text[:100],
                    'url': url,
                    'content': text_content[:500],
                    'status': 'success'
                })
                
                urls_processed.append(url)
                
            except Exception as e:
                search_results.append({
                    'title': 'Error accessing page',
                    'url': url,
                    'content': f'Could not access content: {str(e)}',
                    'status': 'error'
                })
                urls_processed.append(url)
        
        # Format results
        if not search_results:
            return f"❌ No search results found for '{query}'"
        
        response_parts = [
            f"🔍 **Internet Search Results for '{query}':**",
            f"Found {len(search_results)} results",
            ""
        ]
        
        for i, result in enumerate(search_results, 1):
            status_emoji = "✅" if result['status'] == 'success' else "❌"
            response_parts.extend([
                f"{i}. {status_emoji} **{result['title']}**",
                f"   🔗 URL: {result['url']}",
                f"   📄 Content: {result['content']}",
                ""
            ])
        
        return "\n".join(response_parts)
        
    except Exception as e:
        return f"❌ Internet search failed: {str(e)}"

@tool(
    name="extract_webpage_content",
    description="Extract and analyze content from a specific webpage URL"
)
def extract_webpage_content(url: str) -> str:
    """
    Extract content from a specific webpage.
    
    Args:
        url: The URL to extract content from
    
    Returns:
        String with extracted webpage content
    """
    extract_id = str(uuid.uuid4())[:8]
    
    try:
        response = requests.get(url, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract metadata
        title = soup.find('title')
        title_text = title.get_text().strip() if title else "No title"
        
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        description = meta_desc.get('content', '').strip() if meta_desc else "No description"
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
            element.decompose()
        
        # Extract main content
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile('content|main'))
        
        if main_content:
            text_content = main_content.get_text()
        else:
            text_content = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text_content.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        cleaned_text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Extract headings
        headings = []
        for h in soup.find_all(['h1', 'h2', 'h3']):
            if h.get_text().strip():
                headings.append(f"{h.name.upper()}: {h.get_text().strip()}")
        
        # Extract links
        links = []
        for link in soup.find_all('a', href=True)[:10]:  # Limit to 10 links
            if link.get_text().strip() and link['href'].startswith('http'):
                links.append(f"{link.get_text().strip()}: {link['href']}")
        
        response_parts = [
            f"🌐 **Webpage Content Analysis**",
            f"🔗 **URL**: {url}",
            f"📝 **Title**: {title_text}",
            ""
        ]
        
        if description:
            response_parts.extend([
                f"📄 **Description**: {description}",
                ""
            ])
        
        if headings:
            response_parts.extend([
                "📋 **Main Headings**:",
                *[f"  • {heading}" for heading in headings[:5]],
                ""
            ])
        
        # Limit content length
        if len(cleaned_text) > 2000:
            cleaned_text = cleaned_text[:2000] + "..."
        
        response_parts.extend([
            "📄 **Content**:",
            cleaned_text,
            ""
        ])
        
        if links:
            response_parts.extend([
                "🔗 **Related Links**:",
                *[f"  • {link}" for link in links[:5]],
                ""
            ])
        
        response_parts.extend([
            f"📊 **Analysis**: Extracted {len(cleaned_text)} characters from {url}",
            f"⏱️ **Processed**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ])
        
        return "\n".join(response_parts)
        
    except requests.exceptions.RequestException as e:
        return f"❌ Failed to access webpage {url}: {str(e)}"
    except Exception as e:
        return f"❌ Content extraction failed for {url}: {str(e)}"

# ============================================================================
# CUSTOM IMAGE GENERATION TOOLS
# ============================================================================

class ImageGenerationError(Exception):
    """Custom exception for image generation errors"""
    def __init__(self, message):
        self.message = message

@tool(
    name="generate_image_sd3",
    description="Generate high-quality images using Stable Diffusion 3 Large model. Creates images from text prompts and uploads them to S3."
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
    """
    Generate images using Stable Diffusion 3 Large model and upload to S3.
    
    Args:
        prompt: Descriptive text prompt for image generation
        user_id: User identifier for S3 path
        project_id: Project identifier for S3 path
        aspect_ratio: Image aspect ratio (16:9, 1:1, 21:9, 2:3, 3:2, 4:5, 5:4, 9:16, 9:21)
        negative_prompt: What you don't want to see in the image
        seed: Random seed for reproducibility (0 for random)
        output_format: Output format (PNG)
    
    Returns:
        String with generation status and S3 URL
    """
    generation_id = str(uuid.uuid4())[:8]
    
    try:
        # Initialize Bedrock client
        bedrock_client = boto3.client('bedrock-runtime', region_name='us-west-2')
        s3_client = boto3.client('s3', region_name='us-east-1')
        
        # Prepare request body for SD3 Large
        request_body = {
            'prompt': prompt,
            'aspect_ratio': aspect_ratio,
            'mode': 'text-to-image',
            'output_format': output_format
        }
        
        # Add optional parameters
        if negative_prompt:
            request_body['negative_prompt'] = negative_prompt
        if seed > 0:
            request_body['seed'] = seed
        
        # Generate image using Stable Diffusion 3 Large
        response = bedrock_client.invoke_model(
            modelId='stability.sd3-large-v1:0',
            body=json.dumps(request_body)
        )
        
        # Parse response
        output_body = json.loads(response["body"].read().decode("utf-8"))
        
        # Check for errors
        if output_body.get("finish_reasons") and output_body["finish_reasons"][0] is not None:
            error_reason = output_body["finish_reasons"][0]
            return f"❌ Image generation failed: {error_reason}"
        
        # Get base64 image
        base64_image = output_body["images"][0]
        image_data = base64.b64decode(base64_image)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"sd3_generated_{timestamp}_{uuid.uuid4().hex[:8]}.{output_format.lower()}"
        
        # Upload to S3
        bucket_name = 'qubitz-customer-prod'
        s3_key = f"{user_id}/{project_id}/generated_images/{image_filename}"
        
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=image_data,
            ContentType=f'image/{output_format.lower()}',
            Metadata={
                'generated_by': 'stable_diffusion_3_large',
                'prompt': prompt[:200],  # Truncate for metadata
                'user_id': user_id,
                'project_id': project_id,
                'generation_timestamp': timestamp
            }
        )
        
        # Generate S3 URL
        s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"
        
        # Get image dimensions for response
        image_obj = Image.open(BytesIO(image_data))
        width, height = image_obj.size
        
        response_message = f"""✅ **Image Generated Successfully with Stable Diffusion 3 Large!**

🎨 **Generated Image Details:**
• **Prompt**: {prompt}
• **Dimensions**: {width}x{height} pixels
• **Aspect Ratio**: {aspect_ratio}
• **Format**: {output_format}
• **Model**: Stable Diffusion 3 Large

🔗 **Image URL**: {s3_url}

📁 **S3 Location**: {s3_key}
⏱️ **Generated**: {timestamp}

The image has been successfully uploaded to S3 and is ready for use!"""

        return response_message
        
    except ClientError as e:
        return f"❌ AWS Error during image generation: {str(e)}"
    except Exception as e:
        return f"❌ Image generation failed: {str(e)}"

@tool(
    name="generate_image_from_reference",
    description="Generate images based on a reference image using Stable Diffusion XL. Supports image-to-image generation with style transfer."
)
def generate_image_from_reference(
    prompt: str,
    reference_image_s3_url: str,
    user_id: str,
    project_id: str,
    image_strength: float = 0.7,
    cfg_scale: float = 7.0,
    style_preset: str = "photographic",
    steps: int = 30,
    seed: int = 0
) -> str:
    """
    Generate images from a reference image using Stable Diffusion XL.
    
    Args:
        prompt: Text description for the desired output
        reference_image_s3_url: S3 URL of the reference image
        user_id: User identifier for S3 path
        project_id: Project identifier for S3 path
        image_strength: How much the output should resemble the input (0.0-1.0)
        cfg_scale: How closely to follow the prompt (0-35)
        style_preset: Style to apply (3d-model, analog-film, anime, cinematic, etc.)
        steps: Number of generation steps (10-50)
        seed: Random seed for reproducibility (0 for random)
    
    Returns:
        String with generation status and S3 URL
    """
    generation_id = str(uuid.uuid4())[:8]
    
    try:
        # Initialize clients
        bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
        s3_client = boto3.client('s3', region_name='us-east-1')
        
        # Download reference image from S3
        # Extract bucket and key from S3 URL
        if reference_image_s3_url.startswith('https://'):
            # Parse S3 URL
            url_parts = reference_image_s3_url.replace('https://', '').split('/')
            ref_bucket = url_parts[0].split('.s3.amazonaws.com')[0]
            ref_key = '/'.join(url_parts[1:])
        else:
            return "❌ Invalid S3 URL format. Please provide a valid S3 URL."
        
        # Download reference image
        try:
            ref_response = s3_client.get_object(Bucket=ref_bucket, Key=ref_key)
            reference_image_data = ref_response['Body'].read()
            reference_image_base64 = base64.b64encode(reference_image_data).decode('utf-8')
        except ClientError as e:
            return f"❌ Could not download reference image: {str(e)}"
        
        # Prepare request body for SDXL image-to-image
        request_body = {
            "text_prompts": [
                {
                    "text": prompt,
                    "weight": 1.0
                }
            ],
            "init_image": reference_image_base64,
            "init_image_mode": "IMAGE_STRENGTH",
            "image_strength": image_strength,
            "cfg_scale": cfg_scale,
            "steps": steps,
            "samples": 1
        }
        
        # Add optional parameters
        if style_preset:
            request_body["style_preset"] = style_preset
        if seed > 0:
            request_body["seed"] = seed
        
        # Generate image using Stable Diffusion XL
        response = bedrock_client.invoke_model(
            modelId='stability.stable-diffusion-xl-v1',
            body=json.dumps(request_body),
            accept="application/json",
            contentType="application/json"
        )
        
        # Parse response
        response_body = json.loads(response.get("body").read())
        
        # Check for errors
        artifacts = response_body.get("artifacts", [])
        if not artifacts:
            return "❌ No image artifacts returned from the model."
        
        artifact = artifacts[0]
        finish_reason = artifact.get("finishReason")
        
        if finish_reason in ['ERROR', 'CONTENT_FILTERED']:
            return f"❌ Image generation failed: {finish_reason}"
        
        # Get base64 image
        base64_image = artifact.get("base64")
        image_data = base64.b64decode(base64_image)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"sdxl_img2img_{timestamp}_{uuid.uuid4().hex[:8]}.png"
        
        # Upload to S3
        bucket_name = 'qubitz-customer-prod'
        s3_key = f"{user_id}/{project_id}/generated_images/{image_filename}"
        
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=image_data,
            ContentType='image/png',
            Metadata={
                'generated_by': 'stable_diffusion_xl_img2img',
                'prompt': prompt[:200],
                'reference_image': reference_image_s3_url,
                'image_strength': str(image_strength),
                'style_preset': style_preset,
                'user_id': user_id,
                'project_id': project_id,
                'generation_timestamp': timestamp
            }
        )
        
        # Generate S3 URL
        s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"
        
        # Get image dimensions
        image_obj = Image.open(BytesIO(image_data))
        width, height = image_obj.size
        
        response_message = f"""✅ **Image-to-Image Generation Successful with Stable Diffusion XL!**

🎨 **Generated Image Details:**
• **Prompt**: {prompt}
• **Reference Image**: {reference_image_s3_url}
• **Dimensions**: {width}x{height} pixels
• **Image Strength**: {image_strength} (how much it resembles the reference)
• **Style**: {style_preset}
• **CFG Scale**: {cfg_scale}
• **Steps**: {steps}
• **Model**: Stable Diffusion XL

🔗 **Generated Image URL**: {s3_url}

📁 **S3 Location**: {s3_key}
⏱️ **Generated**: {timestamp}

The image has been successfully created based on your reference image and uploaded to S3!"""

        return response_message
        
    except ClientError as e:
        return f"❌ AWS Error during image-to-image generation: {str(e)}"
    except Exception as e:
        return f"❌ Image-to-image generation failed: {str(e)}"

@tool(
    name="upload_image_to_s3",
    description="Upload an image file to S3 and return the URL. Useful for preparing reference images for image-to-image generation."
)
def upload_image_to_s3(
    image_base64: str,
    filename: str,
    user_id: str,
    project_id: str,
    image_format: str = "PNG"
) -> str:
    """
    Upload a base64 encoded image to S3.
    
    Args:
        image_base64: Base64 encoded image data
        filename: Desired filename (without extension)
        user_id: User identifier for S3 path
        project_id: Project identifier for S3 path
        image_format: Image format (PNG, JPEG, etc.)
    
    Returns:
        String with upload status and S3 URL
    """
    upload_id = str(uuid.uuid4())[:8]
    
    try:
        s3_client = boto3.client('s3', region_name='us-east-1')
        
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = re.sub(r'[^a-zA-Z0-9_-]', '_', filename)
        image_filename = f"{safe_filename}_{timestamp}.{image_format.lower()}"
        
        # Upload to S3
        bucket_name = 'qubitz-customer-prod'
        s3_key = f"{user_id}/{project_id}/uploaded_images/{image_filename}"
        
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=image_data,
            ContentType=f'image/{image_format.lower()}',
            Metadata={
                'uploaded_by': 'user',
                'original_filename': filename,
                'user_id': user_id,
                'project_id': project_id,
                'upload_timestamp': timestamp
            }
        )
        
        # Generate S3 URL
        s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_key}"
        
        # Get image info
        image_obj = Image.open(BytesIO(image_data))
        width, height = image_obj.size
        
        response_message = f"""✅ **Image Uploaded Successfully to S3!**

📸 **Upload Details:**
• **Filename**: {image_filename}
• **Dimensions**: {width}x{height} pixels
• **Format**: {image_format}
• **Size**: {len(image_data)} bytes

🔗 **Image URL**: {s3_url}

📁 **S3 Location**: {s3_key}
⏱️ **Uploaded**: {timestamp}

You can now use this S3 URL as a reference image for image-to-image generation!"""

        return response_message
        
    except Exception as e:
        return f"❌ Image upload failed: {str(e)}"

@tool(
    name="image_reader",
    description="Analyze and extract text/information from images using AI vision capabilities"
)
def image_reader(
    image_url: str,
    analysis_type: str = "general"
) -> str:
    """
    Analyze images using AI vision capabilities.
    
    Args:
        image_url: URL of the image to analyze
        analysis_type: Type of analysis (general, text_extraction, technical, creative)
    
    Returns:
        String with image analysis results
    """
    analysis_id = str(uuid.uuid4())[:8]
    
    try:
        # Initialize Bedrock client
        bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
        
        # Download image from URL
        if image_url.startswith('https://'):
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image_data = response.content
        else:
            return "❌ Invalid image URL format. Please provide a valid HTTPS URL."
        
        # Convert to base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Determine analysis prompt based on type
        analysis_prompts = {
            "general": "Describe this image in detail, including objects, people, settings, colors, and overall composition.",
            "text_extraction": "Extract and transcribe all visible text from this image. Include any signs, labels, documents, or written content.",
            "technical": "Analyze this image from a technical perspective, including composition, lighting, visual elements, and any technical aspects.",
            "creative": "Provide a creative analysis of this image, including artistic elements, mood, style, and aesthetic qualities."
        }
        
        prompt = analysis_prompts.get(analysis_type, analysis_prompts["general"])
        
        # Use Claude Vision for image analysis
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        }
        
        # Call Claude Vision
        response = bedrock_client.invoke_model(
            modelId='anthropic.claude-3-sonnet-20240229-v1:0',
            body=json.dumps(request_body)
        )
        
        # Parse response
        response_body = json.loads(response['body'].read())
        analysis_result = response_body['content'][0]['text']
        
        # Get image metadata
        try:
            image_obj = Image.open(BytesIO(image_data))
            width, height = image_obj.size
            format_info = image_obj.format
        except Exception:
            width, height, format_info = "Unknown", "Unknown", "Unknown"
        
        response_message = f"""🔍 **AI Image Analysis Results**

📷 **Image Details:**
• **URL**: {image_url}
• **Dimensions**: {width}x{height} pixels
• **Format**: {format_info}
• **Analysis Type**: {analysis_type.title()}

🤖 **AI Vision Analysis:**
{analysis_result}

⏱️ **Analyzed**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🆔 **Analysis ID**: {analysis_id}

The image has been successfully analyzed using advanced AI vision capabilities!"""

        return response_message
        
    except Exception as e:
        return f"❌ Image analysis failed: {str(e)}"

# ============================================================================
# FASTAPI APPLICATION SETUP
# ============================================================================

app = FastAPI(
    title="AI Agent Factory with Internet Search",
    description="Advanced multi-agent system with internet search capabilities using Google and BeautifulSoup",
    version="11.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Request/Response Models
class WorkflowRequest(BaseModel):
    user_id: str = Field(..., description="User identifier")
    project_id: str = Field(default="default", description="Project identifier")
    message: str = Field(..., description="User message or query")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    file_upload: Optional[Dict[str, Any]] = Field(None, description="File upload data")
    agent_name: Optional[str] = Field(None, description="Specific agent name to use")

class WorkflowResponse(BaseModel):
    response: str
    processing_time: float
    agent_used: str
    workflow_config_found: bool

# ============================================================================
# AGENT FACTORY
# ============================================================================

class AgentFactory:
    def __init__(self):
        """Initialize the agent factory"""
        self.s3_client = boto3.client('s3', region_name='us-east-1')
        self.config_cache = {}
        self.agent_cache = {}
        
        # Available tools
        self.default_tools = [
            search_internet,
            extract_webpage_content,
            generate_image_sd3,
            generate_image_from_reference,
            upload_image_to_s3,
            image_reader
        ]
        
        self.model_mapping = {
            "Claude 3.5 Sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "Claude 3 Opus": "anthropic.claude-3-opus-20240229-v1:0",
            "Claude 3 Sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
            "Claude 3 Haiku": "anthropic.claude-3-haiku-20240307-v1:0",
            "Command R": "cohere.command-r-v1:0", 
            "Command R+": "cohere.command-r-plus-v1:0",
            "Titan Express": "amazon.titan-text-express-v1:0",
            "Mistral Large": "mistral.mistral-large-2402-v1:0",
            "Mixtral 8x7B": "mistral.mixtral-8x7b-v0:1",
            "Llama 3 70B": "meta.llama3-70b-instruct-v1:0",
            "Amazon Nova Pro": "amazon.nova-pro-v1:0"
        }
    
    def get_workflow_config(self, user_id: str, project_id: str) -> Optional[Dict]:
        """Get workflow configuration from cloud storage"""
        cache_key = f"{user_id}_{project_id}"
        
        if cache_key in self.config_cache:
            return self.config_cache[cache_key]
        
        try:
            bucket_name = 'qubitz-customer-prod'
            key = f'{user_id}/{project_id}/workflow.json'
            
            response = self.s3_client.get_object(Bucket=bucket_name, Key=key)
            workflow_data = json.loads(response['Body'].read().decode('utf-8'))
            
            self.config_cache[cache_key] = workflow_data
            
            return workflow_data
            
        except Exception as e:
            return None

    def create_agent_tools_from_config(self, agent_config: Dict) -> List:
        """Create tools for an agent based on configuration"""
        tools = []
        
        # Always add essential tools
        essential_tools = [
            search_internet,
            extract_webpage_content,
            generate_image_sd3,
            generate_image_from_reference,
            upload_image_to_s3,
            image_reader
        ]
        
        for tool in essential_tools:
            if tool and tool not in tools:
                tools.append(tool)
        
        if not tools and self.default_tools:
            tools = self.default_tools
            
        return tools

    def build_system_prompt(self, agent_config: Dict, workflow_config: Dict, user_id: str, project_id: str) -> str:
        """Build system prompt from agent configuration"""
        agent_name = agent_config.get('name', 'Assistant')
        agent_type = agent_config.get('agent_type', 'general')
        description = agent_config.get('description', 'A helpful assistant.')
        instructions = agent_config.get('instructions', '')
        
        return f"""You are {agent_name.replace('_', ' ').title()}, a {agent_type} assistant with advanced internet search and AI capabilities.

{description}

{instructions}

## Your Core Capabilities

**Internet Search & Research:**
- `search_internet`: Search Google and extract content from multiple web pages
- `extract_webpage_content`: Analyze specific websites in detail
- Real-time information research and verification

**Professional AI Image Generation:**
- `generate_image_sd3`: Create high-quality images using Stable Diffusion 3 Large
- `generate_image_from_reference`: Transform images using Stable Diffusion XL
- `upload_image_to_s3`: Upload and manage reference images
- `image_reader`: Advanced AI vision for image analysis and text extraction

User Context: {user_id}/{project_id}

## Excellence Standards

**Research & Information:**
- Always search the internet for current information when needed
- Extract and analyze content from relevant web sources
- Provide accurate, up-to-date information with source citations
- Cross-reference multiple sources for verification

**Image Generation & Analysis:**
- Create compelling visual content with latest AI models
- Use detailed, descriptive prompts for high-quality results
- Automatically upload all generated images to S3
- Always provide S3 URLs to users for immediate access
- Analyze images with advanced AI vision capabilities

**Response Quality:**
- Provide comprehensive, well-structured responses
- Include source URLs and citations for all research
- Explain methodology and approach used
- Maintain professional communication standards

Your goal is to provide exceptional assistance through internet research and advanced AI capabilities. Always search for current information and generate visual content when it would enhance your response.

Remember: All generated images are automatically uploaded to S3 - always provide the S3 URL to users!"""

    def get_agent_from_config(self, agent_config: Dict, workflow_config: Dict, user_id: str, project_id: str) -> Agent:
        """Create agent based on configuration"""
        agent_name = agent_config.get('name', 'default_agent')
        cache_key = f"{user_id}_{project_id}_{agent_name}"
        
        if cache_key in self.agent_cache:
            return self.agent_cache[cache_key]
            
        model_name = agent_config.get('model_name', 'Claude 3.5 Sonnet')
        model_id = agent_config.get('model_id')
        
        if not model_id and model_name in self.model_mapping:
            model_id = self.model_mapping[model_name]
        elif not model_id:
            model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
            
        params = agent_config.get('parameters', {})
        max_tokens = "4000"
        temperature = params.get('temperature', 0.7)
        top_p = params.get('top_p', 0.9)
        
        model = BedrockModel(
            model_id=model_id,
            max_tokens=max_tokens,
            boto_session=boto3.Session(region_name='us-east-1'),
            params={
                "temperature": temperature,
                "top_p": top_p
            }
        )
        
        system_prompt = self.build_system_prompt(agent_config, workflow_config, user_id, project_id)
        tools = self.create_agent_tools_from_config(agent_config)
        
        agent = Agent(
            model=model,
            system_prompt=system_prompt,
            conversation_manager=SlidingWindowConversationManager(
                window_size=10 if agent_config.get('memory_enabled', True) else 1
            ),
            tools=tools
        )
        
        self.agent_cache[cache_key] = agent
        return agent
    
    def get_default_agent(self, user_id: str, project_id: str) -> Agent:
        """Get default agent with internet search capabilities"""
        cache_key = f"default_agent_{user_id}_{project_id}"
        if cache_key in self.agent_cache:
            return self.agent_cache[cache_key]
            
        model = BedrockModel(
            model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
            max_tokens=4000,
            boto_session=boto3.Session(region_name='us-east-1'),
            params={
                "temperature": 0.7,
                "top_p": 0.9
            }
        )
        
        system_prompt = f"""You are an advanced AI assistant with internet search capabilities and professional image generation.

## Your Core Capabilities

**Internet Search & Research:**
- `search_internet`: Search Google and extract content from web pages
- `extract_webpage_content`: Analyze specific websites in detail
- Real-time information research and verification

**Professional AI Image Generation:**
- `generate_image_sd3`: Create high-quality images using Stable Diffusion 3 Large
- `generate_image_from_reference`: Transform images using Stable Diffusion XL
- `upload_image_to_s3`: Upload and manage reference images
- `image_reader`: Advanced AI vision for image analysis

User Context: {user_id}/{project_id}

## Excellence Standards

**Research Excellence:**
- Always search the internet for current information when users ask about recent events, facts, or need verification
- Extract content from relevant sources and provide citations
- Cross-reference multiple sources for accuracy
- Provide up-to-date, verified information

**Image Generation Excellence:**
- Create compelling visual content when requested
- Use detailed, descriptive prompts for high-quality results
- ALL generated images are automatically uploaded to S3
- ALWAYS provide S3 URLs to users for immediate access

**Response Quality:**
- Provide comprehensive, well-structured responses
- Include source URLs for all research
- Explain your research and generation process
- Use proper formatting and clear organization

Your goal is to provide outstanding assistance through internet research and advanced AI capabilities. Search for current information and create visual content to enhance your responses.

Remember: Always provide S3 URLs for generated images and cite sources for research!"""
        
        agent = Agent(
            model=model,
            system_prompt=system_prompt,
            conversation_manager=SlidingWindowConversationManager(window_size=5),
            tools=self.default_tools
        )
        
        self.agent_cache[cache_key] = agent
        return agent

    def select_agent(self, message: str, workflow_config: Dict, requested_agent: Optional[str] = None) -> Dict:
        """Select appropriate agent based on message content and configuration"""
        if not workflow_config or 'agents' not in workflow_config:
            return None
            
        if requested_agent:
            for agent in workflow_config['agents']:
                if agent.get('name') == requested_agent:
                    return agent
        
        # Check for orchestrator first
        for agent in workflow_config['agents']:
            if agent.get('agent_type') == 'orchestrator':
                return agent
                
        # Intelligent selection based on message content
        agents = workflow_config['agents']
        best_match = None
        best_score = -1
        
        for agent in agents:
            score = 0
            agent_name = agent.get('name', '').lower()
            agent_desc = agent.get('description', '').lower()
            message_lower = message.lower()
            
            # Scoring logic
            if agent_name.replace('_', ' ') in message_lower:
                score += 10
            
            # Check description keywords
            desc_words = re.findall(r'\b\w{4,}\b', agent_desc)
            for word in desc_words:
                if word in message_lower:
                    score += 2
            
            if score > best_score:
                best_score = score
                best_match = agent
        
        return best_match or workflow_config['agents'][0]

class SessionManager:
    def __init__(self):
        self.sessions = {}
        
    def get_or_create_session(self, user_id: str, project_id: str) -> Dict:
        session_key = f"{user_id}_{project_id}"
        
        if session_key not in self.sessions:
            self.sessions[session_key] = {
                "history": [],
                "context": {},
                "created_at": datetime.now().isoformat(),
                "last_active": datetime.now().isoformat()
            }
            
        return self.sessions[session_key]
        
    def add_interaction(self, user_id: str, project_id: str, message: str, response: str, agent_name: str):
        session = self.get_or_create_session(user_id, project_id)
        
        session["history"].append({
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "response": response[:500] + ("..." if len(response) > 500 else ""),
            "agent": agent_name
        })
        
        session["last_active"] = datetime.now().isoformat()
        
        # Keep last 15 interactions
        if len(session["history"]) > 15:
            session["history"] = session["history"][-15:]
            
    def get_recent_history(self, user_id: str, project_id: str, count: int = 3) -> List[Dict]:
        session = self.get_or_create_session(user_id, project_id)
        return session["history"][-count:] if session["history"] else []

class Orchestrator:
    def __init__(self):
        self.agent_factory = AgentFactory()
        self.session_manager = SessionManager()
        
    def process_request(self, request: WorkflowRequest) -> tuple:
        start_time = time.time()
        
        workflow_config = self.agent_factory.get_workflow_config(request.user_id, request.project_id)
        workflow_found = workflow_config is not None
        
        history = self.session_manager.get_recent_history(request.user_id, request.project_id)
        
        if workflow_found:
            agent_config = self.agent_factory.select_agent(
                request.message, 
                workflow_config,
                requested_agent=request.agent_name
            )
            
            if agent_config:
                agent = self.agent_factory.get_agent_from_config(
                    agent_config, 
                    workflow_config,
                    request.user_id, 
                    request.project_id
                )
                agent_name = agent_config.get('name', 'unknown')
            else:
                agent = self.agent_factory.get_default_agent(request.user_id, request.project_id)
                agent_name = "default_internet_agent"
        else:
            agent = self.agent_factory.get_default_agent(request.user_id, request.project_id)
            agent_name = "default_internet_agent"
        
        # Build context
        context_parts = []
        
        if request.context:
            for key, value in request.context.items():
                context_parts.append(f"{key}: {value}")
        
        if request.file_upload:
            context_parts.append(f"File upload: {json.dumps(request.file_upload)}")
        
        if history:
            context_parts.append("Recent conversation:")
            for item in history[-2:]:
                context_parts.append(f"User: {item['message']}")
                context_parts.append(f"Assistant: {item['response'][:100]}...")
        
        if context_parts:
            full_request = f"Context: {' | '.join(context_parts)}\n\nUser: {request.message}"
        else:
            full_request = request.message
        
        response = agent(full_request)
        response_text = self._extract_response_text(response)
        
        self.session_manager.add_interaction(
            request.user_id, 
            request.project_id,
            request.message,
            response_text,
            agent_name
        )
        
        processing_time = time.time() - start_time
        
        return response_text, processing_time, agent_name, workflow_found
        
    def stream_request(self, request: WorkflowRequest):
        workflow_config = self.agent_factory.get_workflow_config(request.user_id, request.project_id)
        
        history = self.session_manager.get_recent_history(request.user_id, request.project_id)
        
        if workflow_config:
            agent_config = self.agent_factory.select_agent(
                request.message, 
                workflow_config,
                requested_agent=request.agent_name
            )
            
            if agent_config:
                agent = self.agent_factory.get_agent_from_config(
                    agent_config, 
                    workflow_config,
                    request.user_id, 
                    request.project_id
                )
                agent_name = agent_config.get('name', 'unknown')
            else:
                agent = self.agent_factory.get_default_agent(request.user_id, request.project_id)
                agent_name = "default_internet_agent"
        else:
            agent = self.agent_factory.get_default_agent(request.user_id, request.project_id)
            agent_name = "default_internet_agent"
        
        # Build context
        context_parts = []
        
        if request.context:
            for key, value in request.context.items():
                context_parts.append(f"{key}: {value}")
        
        if request.file_upload:
            context_parts.append(f"File upload: {json.dumps(request.file_upload)}")
        
        if history:
            context_parts.append("Recent conversation:")
            for item in history[-2:]:
                context_parts.append(f"User: {item['message']}")
                context_parts.append(f"Assistant: {item['response'][:100]}...")
        
        if context_parts:
            full_request = f"Context: {' | '.join(context_parts)}\n\nUser: {request.message}"
        else:
            full_request = request.message
        
        async def stream_generator():
            full_response = ""
            
            async for chunk in agent.stream_async(full_request):
                if 'data' in chunk:
                    yield chunk['data']
                    full_response += chunk['data']
                    
            self.session_manager.add_interaction(
                request.user_id, 
                request.project_id,
                request.message,
                full_response,
                agent_name
            )
                
        return stream_generator()
    
    def _extract_response_text(self, response) -> str:
        try:
            if hasattr(response, 'content'):
                if isinstance(response.content, list):
                    text_parts = []
                    for item in response.content:
                        if isinstance(item, dict):
                            if 'text' in item:
                                text_parts.append(item['text'])
                            elif 'content' in item:
                                text_parts.append(str(item['content']))
                        else:
                            text_parts.append(str(item))
                    return ' '.join(text_parts)
                else:
                    return str(response.content)
            else:
                return str(response)
        except Exception as e:
            return str(response)

# Initialize the orchestrator
orchestrator = Orchestrator()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "AI Agent Factory with Internet Search is running",
        "version": "11.0.0",
        "features": {
            "internet_search": True,
            "web_content_extraction": True,
            "google_search_integration": True,
            "beautifulsoup_parsing": True,
            "image_generation": True,
            "stable_diffusion_3_large": True,
            "stable_diffusion_xl_img2img": True,
            "ai_vision_analysis": True,
            "s3_auto_upload": True,
            "agent_factory": True
        }
    }

@app.post('/workflow/chat', response_model=WorkflowResponse)  
async def workflow_chat(request: WorkflowRequest):
    try:
        response_text, processing_time, agent_used, workflow_found = orchestrator.process_request(request)
        return WorkflowResponse(
            response=response_text,
            processing_time=processing_time,
            agent_used=agent_used,
            workflow_config_found=workflow_found
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post('/workflow/stream')
async def workflow_stream(request: WorkflowRequest):
    try:
        return StreamingResponse(
            orchestrator.stream_request(request),
            media_type="text/plain"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error streaming response: {str(e)}")

@app.get('/workflow/agents/{user_id}/{project_id}')
async def get_workflow_agents(user_id: str, project_id: str):
    try:
        workflow_config = orchestrator.agent_factory.get_workflow_config(user_id, project_id)
        
        if not workflow_config or 'agents' not in workflow_config:
            return {
                "message": "No workflow configuration found",
                "agents": [],
                "default_capabilities": [
                    "internet_search",
                    "web_content_extraction", 
                    "image_generation",
                    "ai_vision_analysis"
                ]
            }
        
        agents_info = []
        for agent in workflow_config['agents']:
            agents_info.append({
                "name": agent.get('name'),
                "type": agent.get('agent_type'),
                "description": agent.get('description'),
                "capabilities": [
                    "internet_search",
                    "web_content_extraction",
                    "image_generation", 
                    "ai_vision_analysis",
                    "s3_auto_upload"
                ]
            })
        
        return {
            "workflow_found": True,
            "total_agents": len(agents_info),
            "agents": agents_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting workflow agents: {str(e)}")

@app.get('/test/search')
async def test_search(query: str = "latest AI developments"):
    """Test endpoint for internet search functionality"""
    try:
        result = search_internet(query, num_results=3)
        return {
            "query": query,
            "result": result,
            "status": "success"
        }
    except Exception as e:
        return {
            "query": query,
            "error": str(e),
            "status": "error"
        }

@app.get('/test/webpage')
async def test_webpage(url: str = "https://www.example.com"):
    """Test endpoint for webpage content extraction"""
    try:
        result = extract_webpage_content(url)
        return {
            "url": url,
            "result": result,
            "status": "success"
        }
    except Exception as e:
        return {
            "url": url,
            "error": str(e),
            "status": "error"
        }

# Run the application
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    
    print("🚀 Starting AI Agent Factory with Internet Search")
    print(f"🔍 Google Search Integration: Enabled")
    print(f"🌐 Web Content Extraction: BeautifulSoup4")
    print(f"🎨 Image Generation: Stable Diffusion 3 Large + SDXL")
    print(f"👁️ AI Vision: Advanced image analysis")
    print(f"☁️ S3 Integration: Auto-upload for generated images")
    print(f"🤖 Agent Factory: Dynamic agent creation from workflows")
    print(f"🌐 Server starting on port {port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
