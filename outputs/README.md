# API Knowledge Assistant
A Streamlit application for developers and technical writers to store, manage, and analyze API documentation.

## Overview
The API Knowledge Assistant is a Streamlit application designed to help developers and technical writers store, manage, and analyze API documentation. It allows users to ask questions, receive answers, and track feedback, while also providing an admin panel for monitoring database health and exporting chat history.

## Features
* Ask questions and receive answers based on stored API documentation
* Upload, manage, and delete API documentation files in various formats
* Display performance metrics, including total questions, thumbs up rate, average latency, and feedback distribution
* Admin panel for monitoring database health, clearing all data, and exporting chat history
* Navigation between four pages: Chat Assistant, Document Manager, Analytics, and Admin Panel

## Tech Stack
| Layer        | Technology              |
|-------------|-------------------------|
| UI           | Streamlit               |
| Database     | SQLite (stdlib sqlite3) |
| Language     | Python 3.10+            |
| Charts       | Plotly Express          |
| Tests        | pytest                  |

## Setup
### Prerequisites
* Python 3.10+
* pip

### Clone / Download the Project
Clone the repository using Git:
```bash
git clone https://github.com/your-username/api-knowledge-assistant.git
```
Or download the project as a ZIP file from the GitHub repository.

### Install Dependencies
```bash
pip install streamlit pandas plotly python-dotenv
```

### Create a `.env` file
Create a `.env` file in the project root directory:
```bash
DB_PATH=app.db
```

## Running the App
```bash
streamlit run generated_app.py
```
The application will start on http://localhost:8501 by default.

## Running Tests
```bash
pip install pytest
python -m pytest test_generated_app.py -v
```
The tests cover the main functionality of the application, including question answering, document uploading, and analytics display.

## Database Schema
```sql
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    document_type TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    upload_timestamp DATETIME NOT NULL,
    char_count INTEGER NOT NULL
);

CREATE TABLE conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    latency_ms INTEGER NOT NULL,
    chunks_used TEXT NOT NULL,
    created_at DATETIME NOT NULL
);

CREATE TABLE citations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    chunk_id INTEGER NOT NULL,
    filename TEXT NOT NULL,
    snippet TEXT NOT NULL,
    relevance_score REAL NOT NULL
);

CREATE TABLE feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    score INTEGER NOT NULL CHECK(score IN (0, 1)),
    created_at DATETIME NOT NULL
);
```

## Project Structure
```
generated_app.py          # Main Streamlit application
test_generated_app.py               # Pytest test suite
README.md                 # This file
app.db                    # SQLite database (auto-created on first run)
```

## License
MIT