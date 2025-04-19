import os
import yaml
import json
import logging
import time
from fastapi import FastAPI, HTTPException
from datetime import datetime
from openai import OpenAI
from github import Github
import random
from dotenv import load_dotenv
from vercel_blob import put
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

for var in ["OPENAI_API_KEY", "GIT_TOKEN","BLOB_READ_WRITE_TOKEN"]:
    if not os.getenv(var):
        logger.error(f"{var} not set")
        raise ValueError(f"{var} not set")

app = FastAPI(title="Crew AI Bot API", description="API to run Crew AI Bot", version="1.0.0")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def research_topic(topic, current_year):
    if not topic or not topic.strip():
        raise ValueError("Topic cannot be empty")
    if not current_year.isdigit():
        raise ValueError("Current year must be a number")

    prompt = f"""
    Summarize key developments in {topic} for {current_year} in 8 short bullet points:
    - Recent innovations (1 sentence).
    - Current trends (1 sentence).
    - Key statistics (1 sentence).
    - Future predictions (1 sentence).
    - Practical applications (1 sentence).
    - Notable challenge (1 sentence).
    - Industry impact (1 sentence).
    - Emerging opportunity (1 sentence).
    Keep it concise, max 30 words per bullet.
    """
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7
        )
        content = response.choices[0].message.content.strip()
        if not content:
            raise ValueError("Empty OpenAI response")
        logger.info(f"Research took {time.time() - start_time:.2f} seconds")
        return content
    except Exception as e:
        logger.error(f"OpenAI error: {str(e)}")
        raise RuntimeError(f"Research failed: {str(e)}")


def generate_image_prompt(topic,title,research_output):
    prompt = f"""
    Create a prompt for an image based on the blog post titled '{title}' in the category '{topic}'.
    Use this research: {research_output}
    The image should be visually striking, modern, and relevant (e.g., futuristic AI for AI category, blockchain nodes for Web3).
    Keep the prompt concise, max 20 words.
    """
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.7
        )
        content = response.choices[0].message.content.strip()
        if not content:
            raise ValueError("Empty image prompt response")
        logger.info(f"Image prompt generation took {time.time() - start_time:.2f} seconds")
        return content
    except Exception as e:
        logger.error(f"Image prompt generation error: {str(e)}")
        raise RuntimeError(f"Image prompt generation failed: {str(e)}")


def generate_and_upload_image(image_prompt, title):
    start_time = time.time()
    try:
        logger.info(f"Generating image with prompt: {image_prompt}")
        # Generate image with DALL·E 3
        response = client.images.generate(
            model="dall-e-3",
            prompt=image_prompt,
            size="1024x1024",  # Use supported size; resize later if needed
            quality="standard",
            n=1
        )
        image_url = response.data[0].url
        logger.info(f"Generated image URL: {image_url}")

        # Download image
        image_response = requests.get(image_url, timeout=10)
        image_response.raise_for_status()
        image_data = image_response.content

        # Upload to Vercel Blob
        blob_filename = f"images/{title.lower().replace(' ', '-')}-{int(time.time())}.png"
        blob_response = put(blob_filename, image_data, {"access": "public", "contentType": "image/png"})
        blob_url = blob_response['url']  # Extract only the URL
        logger.info(f"Uploaded image response: {blob_response}")
        logger.info(f"Using image URL: {blob_url}")

        logger.info(f"Image generation and upload took {time.time() - start_time:.2f} seconds")
        return blob_url
    except Exception as e:
        logger.error(f"Image generation/upload error: {str(e)}")
        # Fallback to placeholder image
        placeholder_url = "https://example.com/placeholder.jpg"
        logger.info(f"Using placeholder image: {placeholder_url}")
        return placeholder_url


def write_blog_post(topic, research_output, author_name, author_picture_url, cover_image_url, current_date_iso):
    prompt = f"""
    Write a short blog post about {topic} in Markdown, using this research:
    {research_output}
    Start with this frontmatter (single quotes):
    
    ---
    title: '(Catchy title)'
    status: 'published'
    author:
      name: '{author_name}'
      picture: '{author_picture_url}'
    slug: '(URL-friendly title)'
    description: '(One-sentence summary)'
    coverImage: '{cover_image_url}'
    category: '{topic}'
    publishedAt: '{current_date_iso}'
    ---
    
    Content: 2-3 paragraphs, max 200 words total, no code blocks.
    """
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.7
        )
        content = response.choices[0].message.content.strip()
        if not content:
            raise ValueError("Empty OpenAI response")
        logger.info(f"Blog post took {time.time() - start_time:.2f} seconds")
        return content
    except Exception as e:
        logger.error(f"OpenAI error: {str(e)}")
        raise RuntimeError(f"Blog post failed: {str(e)}")


def git_push_callback(task_output):
    start_time = time.time()
    pat = os.getenv("GIT_TOKEN")
    if not pat:
        logger.error("GIT_TOKEN not set")
        raise ValueError("GIT_TOKEN not set")

    g = Github(pat)
    repo = g.get_repo("abdullahhsajid/bmd-portfolio")
    
    report_file = "/tmp/report.md"
    if not os.path.exists(report_file):
        logger.error(f"Report file missing: {report_file}")
        raise FileNotFoundError(f"Report file missing")

    with open(report_file, 'r') as f:
        content = f.read().strip()

    metadata = {}
    if content.startswith('---'):
        frontmatter_end = content.index('---', 3)
        frontmatter = content[3:frontmatter_end].strip()
        metadata = yaml.safe_load(frontmatter) or {}
    slug = metadata.get('slug', 'default-slug')
    new_filename = f"{slug}.md"

    try:
        repo.create_file(
            f"outstatic/content/blogs/{new_filename}",
            f"Add {new_filename}",
            content
        )
    except Exception as e:
        logger.error(f"Push failed: {str(e)}")
        raise RuntimeError(f"Push failed: {str(e)}")

    metadata_json = {"metadata": []}
    try:
        metadata_file = repo.get_contents("outstatic/content/metadata.json")
        metadata_json = json.loads(metadata_file.decoded_content.decode())
    except:
        logger.info("metadata.json not found")

    new_entry = {
        "category": metadata.get('category', 'Uncategorized'),
        "collection": "blogs",
        "coverImage": metadata.get('coverImage', ''),
        "description": metadata.get('description', ''),
        "publishedAt": metadata.get('publishedAt', ''),
        "slug": slug,
        "status": metadata.get('status', 'draft'),
        "title": metadata.get('title', 'Untitled'),
        "path": f"outstatic/content/blogs/{slug}.md",
        "author": {
            "name": metadata.get('author', {}).get('name', ''),
            "picture": metadata.get('author', {}).get('picture', '')
        },
        "__outstatic": {
            "path": f"outstatic/content/blogs/{slug}.md"
        }
    }
    metadata_json['metadata'].append(new_entry)

    try:
        if 'metadata_file' in locals():
            repo.update_file(
                "outstatic/content/metadata.json",
                "Update metadata",
                json.dumps(metadata_json, indent=2),
                metadata_file.sha
            )
        else:
            repo.create_file(
                "outstatic/content/metadata.json",
                "Create metadata",
                json.dumps(metadata_json, indent=2)
            )
    except Exception as e:
        logger.error(f"Metadata update failed: {str(e)}")
        raise RuntimeError(f"Metadata update failed: {str(e)}")

    logger.info(f"Git push took {time.time() - start_time:.2f} seconds")
    return "Successfully pushed blog post"


def select_category_and_title():
    categories = [
        "AI", "Web3", "Blockchain Fusion", "Startups", "Tech Culture",
        "Tools & Reviews", "How-Tos", "Editorials", "AGI"
    ]
    selected_category = random.choice(categories)
    
    prompt = f"""
    Generate a catchy blog post title for the category '{selected_category}'.
    Focus on recent trends or innovations (e.g., AI breakthroughs, Web3 scalability, AGI ethics).
    Keep it concise, max 10 words.
    """
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.7
        )
        title = response.choices[0].message.content.strip()
        if not title:
            raise ValueError("Empty title response")
        logger.info(f"Title generation took {time.time() - start_time:.2f} seconds")
        return selected_category, title
    except Exception as e:
        logger.error(f"Title generation error: {str(e)}")
        raise RuntimeError(f"Title generation failed: {str(e)}")


@app.get("/")
async def root():
    return {"message": "Blog Agent API is running"}


@app.get("/run-agent")
async def trigger_event():
    run_agent()
    return {"message": "Agent is running in the background"}

@app.get("/api/cron")
async def run_cron():
    return {"message": "Cron job executed successfully"}  

def run_agent():
    start_time = time.time()
    try:
        selected_category, title = select_category_and_title()
        topic = title
        logger.info(f"Selected category: {selected_category}, Title: {title}")

        # required_fields = ['topic', 'author_name', 'author_picture_url']
        # for field in required_fields:
        #     if field not in inputs or not inputs[field] or not str(inputs[field]).strip():
        #         raise ValueError(f"Missing field: {field}")

        current_datetime_iso = datetime.now().isoformat() + "Z"
        topic = topic.strip()
        current_year = str(datetime.now().year)
        author_name = "Abdullah Sajid"
        author_picture_url = "https://avatars.githubusercontent.com/u/176460407?v=4"
        cover_image_url = "https://avatars.githubusercontent.com/u/176460407?v=4"

        research_output = research_topic(topic, current_year)
        logger.info(f"Research output length: {len(research_output)} chars")

        image_prompt = generate_image_prompt(topic, title, research_output)
        logger.info(f"Image prompt: {image_prompt}")
        cover_image_url = generate_and_upload_image(image_prompt, title)

        blog_content = write_blog_post(
            topic, research_output, author_name, author_picture_url, cover_image_url, current_datetime_iso
        )
        logger.info(f"Blog content length: {len(blog_content)} chars")
        
        report_file = "/tmp/report.md"
        with open(report_file, 'w') as f:
            f.write(blog_content)
        logger.info(f"File write took {time.time() - start_time:.2f} seconds so far")

        result = git_push_callback(None)
        total_time = time.time() - start_time
        logger.info(f"Total execution took {total_time:.2f} seconds")
        return {"result": result, "execution_time": total_time}
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
