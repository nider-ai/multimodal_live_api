from functools import lru_cache
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from utils.tools import lookup_policy
from langgraph.prebuilt import ToolNode
import os
from dotenv import load_dotenv
import vertexai
from langchain_google_vertexai import (
    ChatVertexAI,
)
from google.oauth2 import service_account

load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION_ID = os.getenv("LOCATION_ID")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

print(f"Project ID: {PROJECT_ID}")  # Add this line to check the project ID
print(f"Credentials file path: {GOOGLE_APPLICATION_CREDENTIALS}")  # Debug print

credentials = service_account.Credentials.from_service_account_file(
    GOOGLE_APPLICATION_CREDENTIALS,
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)

vertexai.init(project=PROJECT_ID, location=LOCATION_ID, credentials=credentials)


@lru_cache(maxsize=4)
def _get_model(model_name: str):
    if model_name == "gemini":
        model = ChatVertexAI(
            model_name="gemini-1.5-pro-002",
            max_output_tokens=2048,
            temperature=0.1,
        )
    elif model_name == "anthropic":
        model = ChatAnthropic(temperature=0, model_name="claude-3-sonnet-20240229")

    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    model = model.bind_tools([lookup_policy])
    return model


# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there are no tool calls, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


system_prompt = """Be a helpful assistant"""


# Define the function that calls the model
def call_model(state, config):
    messages = state["messages"]
    messages = [{"role": "system", "content": system_prompt}] + messages
    model_name = config.get("configurable", {}).get("model_name", "gemini")
    model = _get_model(model_name)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define the function to execute tools
tool_node = ToolNode([lookup_policy])
