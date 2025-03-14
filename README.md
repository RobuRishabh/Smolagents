# SmolAgent ML Assistant

This project is a Streamlit-based application that provides an end-to-end machine learning workflow. Users can upload a dataset, select features and a target variable, choose a task type (Regression or Classification), and run a predefined ML pipeline. The application uses a custom agent to preprocess data, train a model, evaluate it, and generate a reproducible Python script. Results include performance metrics, visualizations, and the generated code.

## Features
- Upload a CSV dataset via a user-friendly Streamlit interface.
- Select target and feature columns for the ML task.
- Choose between Regression and Classification tasks with supported models (e.g., Linear Regression, Random Forest).
- Execute a full ML pipeline: preprocessing, training, and evaluation.
- Display performance metrics (e.g., MSE, R² for regression; accuracy, precision, recall for classification).
- Visualize results with Plotly (e.g., residual plots for regression, confusion matrices for classification).
- Generate a standalone Python script for the ML workflow.

## Prerequisites
Before running the project, ensure you have the following installed:
- **Python 3.11** (recommended for compatibility with Streamlit and dependencies).
- **Docker**: For containerization.
- **Minikube**: For local Kubernetes deployment.
- **kubectl**: To interact with Kubernetes.
- **Git**: To clone the repository.

## Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/RobuRishabh/Smolagents.git
cd smolagent
```

### 2. Install Dependencies
Install the required Python packages:
```bash
pip install streamlit smolagents pandas numpy scikit-learn plotly pyyaml python-dotenv
```

> Note: If smolagents is a custom package, ensure it's available in your repository or install it manually (e.g., `pip install -e .` if it's part of the project).

### 3. Configure Environment Variables
Create a `.env` file in the project root with the necessary environment variables:
```bash
# .env
API_BASE=http://127.0.0.1:11434
MODEL_ID=ollama_chat/qwen2.5-coder:3b
```

### 5. Run the Application Locally
```bash
streamlit run app.py
```
Open your browser at http://localhost:8501. Upload a CSV dataset, select your features, and run the agent.

---

## Containerization with Docker

### 1. Create a Dockerfile
Create a file named `Dockerfile` in the project root with the following content:
```dockerfile
# Use an official Python runtime as the base image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file (create this file if not already present)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files
COPY . .

# Expose the Streamlit port
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 2. Create a requirements.txt
Generate a `requirements.txt` file:
```bash
pip freeze > requirements.txt
```
Alternatively, manually create `requirements.txt` with the following:
```
streamlit==1.36.0
pandas
numpy
scikit-learn
plotly
pyyaml
python-dotenv
```
> Note: Add `smolagents` if it's a PyPI package, or ensure its source code is included in the project.

### 3. Build the Docker Image
```bash
docker build -t smolagent-ml-assistant:latest .
```

### 4. Test the Docker Container Locally
```bash
docker run -p 8501:8501 --env-file .env smolagent-ml-assistant:latest
```
Access the app at http://localhost:8501.
