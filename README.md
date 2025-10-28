
## Installation
1. **Clone the repository**:
2. **Navigate to the project directory**:
  ```bash
  cd Surveillance_Video_Summarizer
  ```
3. **Install the required Python libraries**:
```bash
pip install -r requirements.txt
```
## **Configuration**
- Have API key stored in a .env file as required.
## **Usage**
Run the frame extraction and logging to db.
```bash
python analyzer_test.py
```
Next, run the interface with log analysis:
```bash
python surveillance_log_analyzer_with_gradio.py
```
