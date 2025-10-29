import gradio as gr
import sqlite3
import pandas as pd
import traceback
import datetime
import requests
import os
"""
python gemini_test.py
This script uses the Gemini API to analyze surveillance logs and provide a summary of the logs.
It uses the Gradio library to create a web interface for the user to interact with the script.
It uses the sqlite3 library to query the database.
It uses the pandas library to read the database.
It uses the traceback library to print the error traceback.
It uses the datetime library to get the current date and time.
It uses the requests library to make HTTP requests.
"""

from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Path to your SQLite DB
db_path = r"C:\Users\sesha\Surveillance_Video_Summarizer\Florence_2_video_analytics\Florence_2_video_analytics.db"

"""
models/gemini-2.5-pro
Best for: Deep multimodal reasoning — text + image + video frames.
models/gemini-2.5-flash best for speed and accuracy
models/embedding-gecko-001 best for text outputs and clustering
"""
# Function to analyze surveillance logs
def analyze_surveillance_logs(descriptions):
    model = genai.GenerativeModel("models/gemini-2.5-pro")  

    
    prompt = f"""
    You are a surveillance expert tasked with analyzing surveillance logs provided for each video frame.
    Your objective is to identify and summarize any unusual activities or noteworthy highlights based on
    the descriptions given for each frame. Highlight any unusual activities or issues, maintaining continuity
    where the context is relevant.

    Surveillance Logs:
    {descriptions}
    """

    try:
        response = model.generate_content(prompt)
        if response and response.text:
            return response.text
        else:
            return "No response generated from Gemini."
    except Exception as e:
        traceback.print_exc()
        return f"Error while generating summary: {e}"


# Function to query DB and summarize logs
def summarize_logs(query_prompt, start_datetime, end_datetime):
    conn = sqlite3.connect(db_path)
    
    # Correcting the format to match the input strings
    format_string = '%Y-%m-%d %I:%M:%S %p'  # Example: 2024-05-25 06:20:00 PM
    print(f"Received start datetime: {start_datetime}")
    print(f"Received end datetime: {end_datetime}")

    try:
        start_dt = pd.to_datetime(start_datetime, format=format_string)
        end_dt = pd.to_datetime(end_datetime, format=format_string)
    except ValueError as e:
        print(f"Error parsing datetime strings: {e}")
        return "Invalid datetime format. Please use 'YYYY-MM-DD HH:MM:SS AM/PM' format."

    # Format the datetime for SQL query
    start_dt_formatted = start_dt.strftime('%Y-%m-%d %H:%M:%S')
    end_dt_formatted = end_dt.strftime('%Y-%m-%d %H:%M:%S')

    print(f"Formatted start datetime for SQL: {start_dt_formatted}")
    print(f"Formatted end datetime for SQL: {end_dt_formatted}")

    query = """
        SELECT * FROM frame_data
        WHERE timestamp BETWEEN ? AND ?
    """
    df = pd.read_sql_query(query, conn, params=(start_dt_formatted, end_dt_formatted))
    conn.close()
    
    if df.empty:
        return "No data available for the specified timeframe."
    else:
        descriptions = ' '.join(df['result'].astype(str).tolist())
        if descriptions: 
            descriptions = analyze_surveillance_logs(descriptions)
            return descriptions


# Gradio Interface
def launch_interface():
    interface = gr.Interface(
        fn=summarize_logs,
        inputs=[
            gr.Textbox(label="Enter your query about the surveillance logs (e.g., unusual activities, highlights)", placeholder="Type here...", value=""),
            gr.Textbox(label="Start DateTime (format: YYYY-MM-DD HH:MM:SS AM/PM)", placeholder="2024-05-25 06:20:00 PM", value="2024-05-25 06:20:00 PM"),
            gr.Textbox(label="End DateTime (format: YYYY-MM-DD HH:MM:SS AM/PM)", placeholder="2024-05-25 06:21:00 PM", value="2024-05-25 06:21:00 PM")
        ],
        outputs=gr.Textbox(label="Output", lines=20, max_lines=10000, placeholder="Results will appear here..."),
        title="Surveillance Video Summarizer",
        description="Enter your query and the time range to analyze surveillance logs."
    )
    interface.launch(share=True)


if __name__ == "__main__":
   launch_interface()
