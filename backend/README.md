# Stock Prediction Backend

This is the backend API for a stock prediction website built using Python Flask and Flask-RESTful.

## Prerequisites

- Python 3.x
- pip (Python package installer)

## Setup

1.  **Clone the repository** (if you have one).
2.  **Navigate to the `backend` directory:**
    ```bash
    cd backend
    ```
3.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    # python -m venv venv
    ```
4.  **Activate the virtual environment:**
    ```bash
    source venv/bin/activate  # On Linux/macOS
    # On Windows:
    # venv\Scripts\activate
    ```
5.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
6.  **Configure the database:**
    - By default, the application uses SQLite (`site.db`). The database file will be created in the `backend` directory when you run the app.
    - To use PostgreSQL, update the `SQLALCHEMY_DATABASE_URI` in `config.py` with your PostgreSQL connection details. You might also need to install the `psycopg2` library (`pip install psycopg2`).

## Running the Backend

```bash
python app.py

venv\Scripts\activate