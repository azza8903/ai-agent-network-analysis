from google import genai
from google.genai import types
import os
import re
import requests
from requests.exceptions import RequestException
from typing import Optional
from dotenv import load_dotenv
from markdownify import markdownify

# --- smolagents Components ---
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    InferenceClientModel,
    DuckDuckGoSearchTool,
    GoogleSearchTool,
    tool
)

from abc import ABC, abstractmethod

# --- Load Environment Variables ---
load_dotenv()


class AI_agent_manager(ABC):
    """Abstract base class for AI agent managers."""

    @abstractmethod
    def configure(self, **kwargs):
        """Configure the agent with necessary parameters."""
        self.model_id = None
        pass

    @abstractmethod
    def run(self, prompt: str) -> Optional[str]:
        """Run the agent with a given input prompt."""
        pass

    @abstractmethod
    def get_name(self) -> Optional[str]:
        """Get the name of the agent based on its class name."""
        pass

    ########################################
    # Gemini Agent Manager
    ########################################

class Gemini_agent_manager(AI_agent_manager):
    def __init__(self):
        self.client = None

    def configure(self, **kwargs):
        self.client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"]
        )

        self.grounding_tool = types.Tool(
            google_search=types.GoogleSearch()
        )

        self.config = types.GenerateContentConfig(
            tools=[self.grounding_tool]
        )

        self.model_id = kwargs.get("model_id")

        print("Gemini agent configured.")

    def run(self, prompt: str) -> Optional[str]:
        if not self.client:
            raise RuntimeError("Gemini agent not configured.")
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("model_id must be a non-empty string.")
        
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=self.config,
        )

        # Ensure a string is returned even if response.text is None
        return response.text

    def get_name(self) -> Optional[str]:
        """Get the name of the agent based on its class name."""
        return "genai/" + self.model_id if self.model_id else "genai/unknown_model"

########################################
# SMAL Agent Manager
########################################

# --- Custom Tool: Visit Webpage ---
# Not strictly a part of smalagents, but used from therein 
@tool
def visit_webpage(url: str) -> str:
    """
    Visits a webpage at the given URL and returns its content as Markdown.

    Args:
        url (str): The URL of the webpage to visit.

    Returns:
        str: The content of the webpage converted to Markdown, or an
            error message if the request fails.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible)'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        markdown_content = markdownify(response.text).strip()
        return re.sub(r"\n{3,}", "\n\n", markdown_content)
    except RequestException as e:
        return f"Request error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

class Smal_agent_manager(AI_agent_manager):
    def __init__(self):
        self.session = None
    
    def configure(self, **kwargs):
        # --- Authenticate Hugging Face ---
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            from huggingface_hub import login
            try:
                login(hf_token)
                print("✅ Logged into Hugging Face Hub.")
            except Exception as e:
                print(f"⚠️ Hugging Face login failed: {e}")
        else:
            print("⚠️ HF_TOKEN not set. Gated models may not work.")


        self.model_id = kwargs.get("model_id")
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ValueError("model_id must be provided as a non-empty string")
        provider = kwargs.get("provider")
        self.model = InferenceClientModel(model_id=self.model_id, provider=provider)

        WEB_TOOLS = {
            "google": GoogleSearchTool(provider="serper"),
            "duckduckgo": DuckDuckGoSearchTool()
        }

        web_agent = ToolCallingAgent(
            tools=[WEB_TOOLS["duckduckgo"], visit_webpage],
            model=self.model,
            max_steps=10,
            name="web_search_agent",
            description="Performs web searches and visits pages.",
            verbosity_level=1,
        )

        self.manager_agent = CodeAgent(
            tools=[],
            model=self.model,
            managed_agents=[web_agent],
            additional_authorized_imports=["time", "numpy", "pandas"],
            verbosity_level=1,
        )
        print("SMAL agent configured.")

    def run(self, prompt: str) -> Optional[str]:
        final_answer = self.manager_agent.run(prompt)

        try:
            coerced_answer = str(final_answer)
            return coerced_answer
        except Exception:
            return None

    def get_name(self) -> Optional[str]:
        """Get the name of the agent based on its class name."""
        return "smal/" + self.model_id if self.model_id else "smal/unknown_model"