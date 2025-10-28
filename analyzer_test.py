import os
import sqlite3
import cv2
import threading
from queue import Queue
from PIL import Image
import torch
import datetime
import logging
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor


'''
python analyzer_test.py
This script extracts frames from videos and processes them using the Florence-2 fine-tuned model.
It saves the frames to the frames folder and the results to the database.
It also logs the results to the debug.log file.
It uses the transformers library to load the model and processor.
It uses the cv2 library to extract the frames from the videos.
It uses the sqlite3 library to save the results to the database.
It uses the logging library to log the results to the debug.log file.
'''

# ===============================
# CONFIGURATION
# ===============================

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Florence-2 fine-tuned model
MODEL_NAME = "kndrvitja/florence-SPHAR-finetune-2"

# Creating a model from scratch (without downloading pretrained weights).
config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
if config.vision_config.model_type != 'davit':
    config.vision_config.model_type = 'davit'

# language model for Causal Language Modeling text generation
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, config=config, trust_remote_code=True
).to(device)

# combines multiple preprocessing tools (like tokenizer + feature extractor) into a single object.
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

# ===============================
# PATHS
# ===============================

BASE_DIR = r"C:\Users\sesha\Surveillance_Video_Summarizer\Florence_2_video_analytics"
VIDEO_FOLDER_PATH = r"C:\Users\sesha\Surveillance_Video_Summarizer\video_content"
FRAME_SAVE_PATH = os.path.join(BASE_DIR, "frames")

if not os.path.exists(FRAME_SAVE_PATH):
    os.makedirs(FRAME_SAVE_PATH)

DB_FILE_PATH = os.path.join(BASE_DIR, "Florence_2_video_analytics.db")

# ===============================
# LOGGING
# ===============================
LOG_FILE = os.path.join(BASE_DIR, "debug.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s"
)

# ===============================
# DATABASE SETUP
# ===============================
def setup_database(db_file_path):
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS frame_data (
            timestamp TEXT,
            frame_path TEXT,
            result TEXT
        )
    ''')
    conn.commit()
    conn.close()

# ===============================
# FRAME EXTRACTION
# ===============================
def extract_frames(video_files, interval=1):
    """ Extracts frames from a list of videos every `interval` seconds. """
    for video_file in video_files:
        try:
            cap = cv2.VideoCapture(video_file)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                current_time = frame_number / fps
                if current_time >= interval:
                    frame_path = os.path.join(
                        FRAME_SAVE_PATH, f"frame_{int(current_time)}.jpg"
                    )
                    cv2.imwrite(frame_path, frame)
                    frame_queue.put(frame_path)

                    frame_number += int(fps * interval)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                else:
                    frame_number += 1
            cap.release()

        except Exception as e:
            logging.error(f"Error extracting frames from {video_file}: {e}")

# ===============================
# MODEL INFERENCE
# ===============================
def run_example(task_prompt, text_input, image):
    prompt = task_prompt if text_input is None else task_prompt + text_input

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=False
    )[0]

    parsed_answer = processor.post_process_generation(
        generated_text, task=task_prompt, image_size=(image.width, image.height)
    )

    return str(parsed_answer) if isinstance(parsed_answer, dict) else parsed_answer

# ===============================
# FRAME PROCESSING
# ===============================
def process_frames(frame_queue, db_file_path):
    setup_database(db_file_path)
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()

    while True:
        frame_path = frame_queue.get()
        if frame_path is None:
            break

        try:
            image = Image.open(frame_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            answer = run_example(
                task_prompt="<SURVEILLANCE>",
                text_input=(
                    "Given the surveillance image, create a detailed annotation "
                    "capturing dynamic elements, individuals, actions, interactions, "
                    "notable objects, and events. The annotation should summarize "
                    "observed behaviors and situations in 2-3 lines."
                ),
                image=image
            )

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result = f"Processed frame at {frame_path}: {answer}"

            cursor.execute(
                "INSERT INTO frame_data (timestamp, frame_path, result) VALUES (?, ?, ?)",
                (timestamp, frame_path, result)
            )
            conn.commit()

            logging.info(result)

        except Exception as e:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            error_msg = f"Error processing {frame_path}: {e}"
            cursor.execute(
                "INSERT INTO frame_data (timestamp, frame_path, result) VALUES (?, ?, ?)",
                (timestamp, frame_path, error_msg)
            )
            conn.commit()
            logging.error(error_msg)

    conn.close()

# ===============================
# MAIN EXECUTION
# ===============================

if __name__ == "__main__":
    frame_queue = Queue()

    # Setup database
    setup_database(DB_FILE_PATH)

    # Collect videos
    video_files = [
        os.path.join(VIDEO_FOLDER_PATH, f)
        for f in os.listdir(VIDEO_FOLDER_PATH)
        if f.endswith(('.mp4', '.avi', '.mov'))
    ]

    if not video_files:
        logging.warning("No video files found in folder.")

    # Start threads
    threading.Thread(target=extract_frames, args=(video_files,)).start()
    threading.Thread(target=process_frames, args=(frame_queue, DB_FILE_PATH)).start()
