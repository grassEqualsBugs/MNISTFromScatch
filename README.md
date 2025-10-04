# MNIST from Scratch

## Description

A simple web application that implements a neural network from scratch to recognize handwritten digits from the MNIST dataset. It features a React frontend with a canvas for user input and a Python/Flask/NumPy backend for the neural network logic.

## Frameworks & Libraries

*   **Frontend:** React, JavaScript
*   **Backend:** Python, Flask, NumPy

## How to Run Locally

To run this project, you will need to run the backend server and the frontend application in two separate terminals.

### Backend (Flask Server)

1.  **Navigate to the server directory:**
    ```bash
    cd server
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install the dependencies:**
    ```bash
    # For macOS/Linux
    pip3 install -r requirements.txt

    # For Windows
    pip install -r requirements.txt
    ```

4.  **Run the server:**
    ```bash
    # For macOS/Linux
    python3 server.py

    # For Windows
    python server.py
    ```
    The backend server will start on `http://localhost:5000`.

### Frontend (React App)

1.  **Navigate to the client directory in a new terminal:**
    ```bash
    cd client
    ```

2.  **Install the dependencies:**
    ```bash
    npm install
    ```

3.  **Run the app:**
    ```bash
    npm start
    ```
    The frontend development server will start, and your browser should open to `http://localhost:3000`.
