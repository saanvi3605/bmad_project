# Library Management System
A tool designed for librarians to manage book collections and member registrations.

## Overview
The Library Management System is a Streamlit application that helps librarians keep track of available books, issued books, and members with overdue fines. It is intended for use by librarians in libraries.

## Features
* Add new books with title, author, ISBN, genre, and total copies
* Remove books from the collection
* Register new members with name, member ID, and phone number
* Issue books to registered members with a 14-day due date
* Calculate an overdue fine of Rs. 2 per day if a book is returned late
* Display metric cards showing total books, copies currently issued, available copies, and members with overdue books
* Search for books by title or author using a search bar

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

### Clone / download the project
Clone the repository using `git clone` or download the zip file from the repository.

### Install dependencies
```bash
pip install streamlit pandas plotly python-dotenv
```

### Create a `.env` file
Create a `.env` file in the project root with the following content:
```makefile
DB_PATH=app.db
```

## Running the App
```bash
streamlit run generated_app.py
```
The default browser URL is http://localhost:8501.

## Running Tests
```bash
pip install pytest
python -m pytest test_generated_app.py -v
```
The tests cover the core functionality of the application, including adding and removing books, registering and issuing books to members, and calculating overdue fines.

## Database Schema
```sql
CREATE TABLE books (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    author TEXT NOT NULL,
    ISBN TEXT NOT NULL,
    genre TEXT NOT NULL,
    total_copies INTEGER NOT NULL,
    available_copies INTEGER NOT NULL,
    issued_copies INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE members (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    member_ID TEXT NOT NULL,
    phone TEXT NOT NULL,
    overdue_fine REAL NOT NULL DEFAULT 0.0,
    last_issued_book TEXT
);

CREATE TABLE issued_books (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id INTEGER NOT NULL,
    member_id INTEGER NOT NULL,
    issue_date DATE NOT NULL,
    due_date DATE NOT NULL
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