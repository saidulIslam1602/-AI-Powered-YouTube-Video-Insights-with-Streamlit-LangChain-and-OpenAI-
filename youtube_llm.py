import streamlit as st
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import HumanMessage, SystemMessage, Document
from dotenv import find_dotenv, load_dotenv
import os
import re
from langdetect import detect
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Retrieve the OpenAI API key from the environment variables
api_key = os.getenv('OPENAI_API_KEY')

# Check if the API key is loaded
if not api_key:
    st.error("OpenAI API key is not set. Please ensure the OPENAI_API_KEY is set in your environment variables.")
    st.stop()

# Initialize the OpenAIEmbeddings with the correct API key
embeddings = OpenAIEmbeddings(openai_api_key=api_key)

def create_db_from_youtube_video_url(video_url: str) -> FAISS:
    video_id = video_url.split("v=")[-1]
    try:
        # Attempt to get the transcript in English
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = transcript_list.find_transcript(['en'])
        except NoTranscriptFound:
            st.warning("English transcript not found. Attempting to retrieve the transcript in another language.")
            transcript = transcript_list.find_transcript(transcript_list.video_languages)

        transcript_text = " ".join([entry['text'] for entry in transcript.fetch()])
    except (NoTranscriptFound, TranscriptsDisabled):
        st.error("Could not retrieve a transcript for this video in any available language.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.stop()

    st.write("Transcript loaded successfully!")

    # Handle the possibility that the transcript is empty or None
    if not transcript_text:
        st.error("Transcript is empty or not found. Please check the video content.")
        st.stop()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(transcript_text)

    # Convert chunks to Document objects
    docs = [Document(page_content=chunk) for chunk in chunks]

    # Check if the documents are created successfully
    if not docs or len(docs) == 0:
        st.error("Failed to split the transcript into documents. Please check the video content.")
        st.stop()

    st.write(f"Transcript split into {len(docs)} document chunks.")

    db = FAISS.from_documents(docs, embeddings)
    return db

def detect_language(text):
    try:
        return detect(text)
    except:
        return None

def get_response_from_query(db, query, language, k=4):
    docs = db.similarity_search(query, k=k)

    # Check if any documents are returned from the similarity search
    if not docs or len(docs) == 0:
        st.error("No relevant documents found in the database. Please try a different query.")
        st.stop()

    docs_page_content = " ".join([d.page_content for d in docs])

    llm = ChatOpenAI(model="gpt-4-turbo", openai_api_key=api_key)

    messages = [
        SystemMessage(content=f"You are a helpful assistant that can answer questions about YouTube videos based on the video's transcript. The answer should be in {language}."),
        HumanMessage(content=f"Answer the following question: {query} by searching the following video transcript: {docs_page_content}. Only use the factual information from the transcript to answer the question. If you feel like you don't have enough information to answer the question, say 'I don't know'. Your answers should be verbose and detailed.")
    ]

    response = llm.invoke(messages)

    response_text = response.content
    return response_text, docs

# Streamlit UI
st.set_page_config(page_title="YouTube Video Insights", layout="centered")

# Custom CSS for design
st.markdown(
    """
    <style>
    body {
        background-color: #f4f4f4;
        color: #333333;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
    }
    .stTextInput input {
        border: 2px solid #4CAF50;
        border-radius: 8px;
        padding: 10px;
    }
    .stSpinner {
        color: #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("YouTube Video Insights with LLM")

# Input field for YouTube video URL
video_url = st.text_input("Enter YouTube Video URL:")

# Regular expression for validating YouTube URLs
youtube_url_pattern = re.compile(
    r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')

# Input field for query
query = st.text_input("Enter your query:")

# Button to generate answer
if st.button("Generate Answer") or (st.session_state.get('query_submit') and query):
    if not video_url:
        st.warning("Please enter a YouTube video URL.")
    elif not re.match(youtube_url_pattern, video_url):
        st.warning("Please enter a valid YouTube video URL.")
    elif query:
        with st.spinner('Processing...'):
            try:
                # Detect language of the query
                language = detect_language(query)
                if language is None:
                    st.error("Could not detect the language of the query.")
                    st.stop()

                st.write(f"Detected language: {language}")

                db = create_db_from_youtube_video_url(video_url)
                response, docs = get_response_from_query(db, query, language)
                st.success("Answer generated successfully!")
                st.write("### Answer:")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a query.")

# Check for the Enter key press for the query input
def on_enter():
    st.session_state.query_submit = True