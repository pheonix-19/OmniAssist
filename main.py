import os
import requests
import openai
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from pydub.utils import make_chunks
from newspaper import Article
from sklearn.feature_extraction.text import CountVectorizer
from dotenv import load_dotenv
from langchain.tools import StructuredTool, load_tools
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from fpdf import FPDF
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import smtplib

# Load environment variables from .env file
load_dotenv()

# Set API keys
openai.api_key = os.getenv('OPENAI_API_KEY')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
FIGMA_API_KEY = os.getenv('FIGMA_API_KEY')
EMAIL_ADDRESS = os.getenv('EMAIL_ADDRESS')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')
os.environ["SERPAPI_API_KEY"] = "API-KEY"


# Tool 1: Video to Notes
def video_to_notes(video_path):
    """
    Extracts audio from a video file, transcribes the audio into text, and generates detailed notes from the transcribed text.

    1. Extract audio from the video file and save it as a WAV file.
    2. Split the audio into chunks and transcribe each chunk using OpenAI's Whisper model.
    3. Combine the transcribed text into a single transcript.
    4. Use OpenAI's GPT model to generate detailed notes from the transcript.
    """
    pass


# Tool 2: News Aggregator
def news_aggregator(question):
    """
    Aggregates news articles based on a query and generates a response using the articles' content.

    1. Extract keywords from the user's question.
    2. Fetch news articles related to the keywords using a news API.
    3. Extract content from each article.
    4. Generate a response to the user's question based on the article contents using OpenAI's GPT model.
    """
    pass


# Tool 3: Figma to HTML/CSS
def figma_to_html_css(figma_file_id):
    """
    Converts design elements from a Figma file into HTML and CSS code.

    1. Fetch design data from the Figma API using the file ID.
    2. Extract design elements such as dimensions, colors, and fonts.
    3. Generate HTML and CSS code for each design element using OpenAI's GPT model.
    4. Save the generated HTML and CSS code to files in a specified directory.
    """
    pass


# Tool 4: Generate Python Code from Prompt
def generate_code(prompt):
    """
    Generates Python code based on a given prompt and optionally saves it to a file.

    1. Use OpenAI's GPT model to generate Python code based on the prompt.
    2. If the user chooses, save the generated code to a specified file.
    """
    pass


# New Email Tool functions
def create_pdf(text, pdf_filename):
    """
    Creates a PDF file from the provided text.

    1. Initialize a PDF document.
    2. Add a page and set the font.
    3. Write the text to the PDF document.
    4. Save the PDF to the specified filename.
    """
    pass


def send_email_with_pdf(subject, body, recipients, text_for_pdf):
    """
    Sends an email with a PDF attachment.

    1. Create a PDF from the provided text.
    2. Compose an email with the given subject and body.
    3. Attach the PDF to the email.
    4. Send the email to the specified recipients using an SMTP server.
    """
    pass


# Tool 6: Save Content to Directory
def save_content_to_directory(content, directory, file_name):
    """
    Saves the provided content to a file in a specified directory.

    1. Create the directory if it does not exist.
    2. Save the content to a file within the directory.
    """
    pass


# Tool 7: Internet Access Tool
class InternetAccessTool(StructuredTool):
    def __init__(self):
        super().__init__()

    def call(self, query):
        """
        Placeholder for internet querying logic.

        This function should be implemented to perform internet queries based on the provided query.
        """
        pass


# Setup the ChatGPT agent with tools
def setup_agent():
    """
    Sets up a ChatGPT agent with various tools.

    1. Define a list of tools, each represented by a StructuredTool or custom tool.
    2. Create a ChatPromptTemplate for interacting with the agent.
    3. Initialize the AgentExecutor with the tools and prompt.
    """
    tools = [
        StructuredTool(name="Video to Notes", func=video_to_notes),
        StructuredTool(name="News Aggregator", func=news_aggregator),
        StructuredTool(name="Figma to HTML/CSS", func=figma_to_html_css),
        StructuredTool(name="Generate Python Code", func=generate_code),
        StructuredTool(name="Send Email with PDF", func=send_email_with_pdf),
        StructuredTool(name="Save Content to Directory", func=save_content_to_directory),
        InternetAccessTool(name="Internet Access", func=None)  # Define specific call function
    ]

    prompt = ChatPromptTemplate(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What do you need help with today?"}
        ]
    )
    agent = AgentExecutor(
        tools=tools,
        agent_type="chat-openai",
        prompt=prompt
    )

    return agent


def main():
    """
    Main function to run the ChatGPT agent.

    1. Initialize the agent.
    2. Continuously prompt the user for tasks.
    3. Process each task using the agent and print the response.
    """
    agent = setup_agent()
    while True:
        user_input = input("\nEnter your task (type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        response = agent.run(user_input)
        print(f"\nResponse: {response}")


if __name__ == "__main__":
    main()
