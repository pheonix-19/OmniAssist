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
from langchain.agents import load_tools
from langchain.agents import initialize_agent
import openai
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Set API keys
openai.api_key = os.getenv('OPENAI_API_KEY')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
FIGMA_API_KEY = os.getenv('FIGMA_API_KEY')
EMAIL_ADDRESS = os.getenv('EMAIL_ADDRESS')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')
os.environ["SERPAPI_API_KEY"] = "7a0718dd8a8cb05dfe2928b51110dc9485a986d9d89dca7bc3922a1624da5173"

# Tool 1: Video to Notes
def video_to_notes(video_path):
    # Extract audio from video
    def extract_audio(video_path, audio_path):
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)

    audio_path = "output_audio.wav"
    extract_audio(video_path, audio_path)

    # Transcribe audio chunks
    def transcribe_audio_chunk(audio_chunk, chunk_number):
        chunk_path = f"chunk{chunk_number}.wav"
        audio_chunk.export(chunk_path, format="wav")
        with open(chunk_path, 'rb') as audio_file:
            response = openai.Audio.transcribe("whisper-1", audio_file)
        os.remove(chunk_path)  # Clean up the chunk file after transcribing
        return response['text']

    def split_and_transcribe_audio(audio_path):
        audio = AudioSegment.from_wav(audio_path)
        chunk_length_ms = 60000  # 1 minute (60,000 ms)
        chunks = make_chunks(audio, chunk_length_ms)

        transcript = ""
        for i, chunk in enumerate(chunks):
            transcript += transcribe_audio_chunk(chunk, i) + " "

        return transcript

    transcript = split_and_transcribe_audio(audio_path)

    # Generate notes from transcribed text
    def generate_notes(text, temperature=0.5):
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Create detailed notes from the following text: {text}"}
            ],
            temperature=temperature
        )
        return response.choices[0].message['content']

    notes = generate_notes(transcript)
    return notes

# Tool 2: News Aggregator
def news_aggregator(question):
    def extract_keywords(query):
        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform([query])
        return vectorizer.get_feature_names_out()

    def fetch_news(keywords, language='en', limit=3):
        base_url = 'https://api.thenewsapi.com/v1/news/all'
        query = ' '.join(keywords)  # Join keywords to form a search query
        params = {
            'api_token': NEWS_API_KEY,
            'language': language,
            'limit': limit,
            'search': query
        }
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            return response.json().get('data', [])
        else:
            print(f"Error: {response.status_code}")
            return []

    def extract_article_content(url):
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article.text
        except Exception as e:
            print(f"Failed to extract content from {url}: {e}")
            return ""

    def get_article_contents(articles):
        contents = []
        for article in articles:
            title = article['title']
            description = article['description']
            url = article['url']
            content = extract_article_content(url)
            contents.append({'title': title, 'description': description, 'content': content, 'url': url})
        return contents

    def generate_llm_response(question, contents):
        combined_content = "\n\n".join(
            f"Title: {article['title']}\nDescription: {article['description']}\nContent: {article['content']}"
            for article in contents
        )
        prompt = f"""
        Based on the following articles:

        {combined_content}

        Please answer the following question: {question}
        """
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100
        )
        return response.choices[0].message['content'].strip()

    # Extract keywords from the query
    keywords = extract_keywords(question)
    # Fetch news articles based on extracted keywords
    articles = fetch_news(keywords)
    # Get the contents of the fetched articles
    article_contents = get_article_contents(articles)
    # Generate a response using the LLM
    return generate_llm_response(question, article_contents)

# Tool 3: Figma to HTML/CSS
def figma_to_html_css(figma_file_id):
    def fetch_figma_design(figma_file_id):
        headers = {'X-Figma-Token': FIGMA_API_KEY}
        url = f'https://api.figma.com/v1/files/{figma_file_id}'
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to fetch data: {response.status_code}")

    def extract_design_elements(design_data):
        elements = []
        for page in design_data['document']['children']:
            for frame in page['children']:
                element = {
                    'name': frame['name'],
                    'width': frame['absoluteBoundingBox']['width'],
                    'height': frame['absoluteBoundingBox']['height'],
                    'color': frame.get('backgroundColor', {}),
                    'font': frame.get('style', {}).get('fontFamily'),
                    'font_size': frame.get('style', {}).get('fontSize'),
                }
                elements.append(element)
        return elements

    def generate_code_from_design(design_element):
        prompt = f"""
        Convert the following design element into HTML and CSS:
        Name: {design_element['name']}
        Width: {design_element['width']}px
        Height: {design_element['height']}px
        Background Color: {design_element['color']}
        Font: {design_element['font']}
        Font Size: {design_element['font_size']}px
        """
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message['content'].strip()

    def save_code_to_files(element_name, code):
        output_dir = 'output3'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save HTML code
        html_path = os.path.join(output_dir, f'{element_name}.html')
        with open(html_path, 'w', encoding='utf-8') as html_file:
            html_file.write(code)

        # Save CSS code
        css_path = os.path.join(output_dir, f'{element_name}.css')
        with open(css_path, 'w', encoding='utf-8') as css_file:
            css_file.write(code)

    design_data = fetch_figma_design(figma_file_id)
    design_elements = extract_design_elements(design_data)

    for element in design_elements:
        code = generate_code_from_design(element)
        save_code_to_files(element['name'], code)

    return "HTML/CSS files have been saved in the output directory."

# Tool 4: Generate Python Code from Prompt
def generate_code(prompt):
    """
    Generates Python code based on the given prompt using GPT-4o-mini.

    Parameters:
    prompt (str): The prompt describing the task or code to be generated.

    Returns:
    str: The generated Python code as a string.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        generated_code = response.choices[0].message['content'].strip()
        if generated_code:
            # Optionally, save the generated code to a file
            save_option = input("\nDo you want to save the generated code to a file? (y/n): ").lower()
            if save_option == "y":
                file_name = input("Enter the file name (e.g., generated_code.py): ")
                with open(file_name, "w") as file:
                    file.write(generated_code)
                print(f"Code saved to {file_name}")



        return generated_code
    except Exception as e:
        return f"Error generating code: {str(e)}"


# New Email Tool functions
def create_pdf(text, pdf_filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    pdf.output(pdf_filename)


def send_email_with_pdf(subject, body, recipients, text_for_pdf):
    pdf_filename = "attachment.pdf"
    create_pdf(text_for_pdf, pdf_filename)

    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    with open(pdf_filename, "rb") as attachment:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f"attachment; filename={pdf_filename}")
        msg.attach(part)

    try:
        server = smtplib.SMTP('smtp.hostinger.com', 587)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)

        for recipient in recipients:
            msg['To'] = recipient
            text = msg.as_string()
            server.sendmail(EMAIL_ADDRESS, recipient, text)

        server.quit()
        os.remove(pdf_filename)  # Clean up the PDF file
        return "Email sent successfully!"
    except Exception as e:
        return f"Failed to send email: {e}"


def save_to_directory(content, file_name, output_dir='output'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)
    return f"File saved as {file_path}"



def internet_access_tool_function(query):
    """
    Function to run the agent with the provided query.

    Parameters:
    query (str): The input query for the agent.

    Returns:
    str: The response from the agent.
    """
    return agent.run(query)
# Argument models
class VideoToNotesArgs(BaseModel):
    video_path: str

class NewsAggregatorArgs(BaseModel):
    question: str

class FigmaToHtmlCssArgs(BaseModel):
    figma_file_id: str

class GenerateCodeArgs(BaseModel):
    prompt: str


class EmailWithPdfArgs(BaseModel):
    subject: str
    body: str
    recipients: list
    text_for_pdf: str
class SaveToDirectoryArgs(BaseModel):
    content: str
    file_name: str
    output_dir: str = 'output'

class TextToPdfArgs(BaseModel):
    text: str
    file_name: str

class InternetAccessTool(StructuredTool):
    def __init__(self):
        super().__init__(
            name="Internet_Access_Tool",
            func=internet_access_tool_function,
            description="A tool to query the internet using a language model."
        )

# agent = initialize_agent(tools, llm_i, agent="zero-shot-react-description", verbose=True)
# Create LangChain Tools
save_to_directory_tool = StructuredTool(
    name="Save_to_Directory",
    func=save_to_directory,
    description="Save content to a specified directory.",
    args_schema=SaveToDirectoryArgs
)
video_to_notes_tool = StructuredTool(
    name="Video_to_Notes",
    func=video_to_notes,
    description="Convert video content to detailed notes.",
    args_schema=VideoToNotesArgs
)

news_aggregator_tool = StructuredTool(
    name="News_Aggregator",
    func=news_aggregator,
    description="Fetch and summarize news articles based on a query.",
    args_schema=NewsAggregatorArgs
)

figma_to_html_css_tool = StructuredTool(
    name="Figma_to_HTML-CSS",
    func=figma_to_html_css,
    description="Convert Figma design to HTML and CSS code.",
    args_schema=FigmaToHtmlCssArgs
)

generate_code_tool = StructuredTool(
    name="Generate_Python_Code",
    func=generate_code,
    description="Generate Python code based on a given prompt.",
    args_schema=GenerateCodeArgs
)
email_tool = StructuredTool(
    name="Send_Email_with_PDF",
    func=send_email_with_pdf,
    description="Send an email with a PDF attachment created from provided text.",
    args_schema=EmailWithPdfArgs
)

# Initialize chat history
chat_history = []
llm_i = OpenAI(temperature=0)
# Combine all tools

tools = load_tools(["serpapi", "llm-math"], llm=llm_i) + [
    save_to_directory_tool, video_to_notes_tool, news_aggregator_tool,
    figma_to_html_css_tool, generate_code_tool, email_tool
]

agent = initialize_agent(tools, llm_i, agent="zero-shot-react-description", verbose=True)

# Define the agent prompt with additional context
agent_prompt = """
You are an intelligent assistant capable of performing the following tasks:
1. Convert a video into detailed notes.
2. Aggregate and summarize news articles.
3. Convert Figma designs into HTML/CSS code.
4. Generate Python code by given prompts.
5. Send an email with a PDF attachment created from provided text.
4. Convert text to a PDF file.
5. Save content to a specified directory.
6.internet acesses

When a user gives you a task, determine which tool to use and ask them for any necessary input parameters. 
If the user doesn't provide enough information, ask follow-up questions to gather the necessary details.

Task: {task}

Parameters:
{parameters}

Thought Process:
{agent_scratchpad}
"""

# Initialize the LLM and agent
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
prompt = ChatPromptTemplate.from_template(agent_prompt)
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools,verbose=True)

# Function to handle user query
def handle_user_query(task, params={}):
    global chat_history
    # Construct the chat history string
    history_str = "\n".join(chat_history)

    try:
        # Prepare parameters for the agent
        agent_params = {
            "task": task,
            "parameters": params,
            "agent_scratchpad": "",
            "history": history_str
        }

        # Get response from the agent executor
        response = agent_executor.invoke(agent_params)

        # Update chat history with the user query and agent response
        chat_history.append(f"User: {task}")
        chat_history.append(f"Agent: {response}")

        return response
    except Exception as e:
        return f"Error while processing task: {e}"


# Main loop to handle user input
while True:
    task = input("Enter the task (type 'exit' to quit): ").strip()
    if task.lower() == 'exit':
        print("Exiting the program.")
        break
    response = handle_user_query(task)
    print("Response:", response)
