# Dockerfile

# Stage 1: Build Stage
# Use a lightweight official Python image for the base
FROM python:3.12-slim-bookworm

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app

# Create and set the working directory inside the container
WORKDIR $APP_HOME



COPY requirements.txt .

# FIX 1: Install necessary system libraries
RUN apt-get update && \
    apt-get install -y build-essential libatlas-base-dev gfortran && \
    rm -rf /var/lib/apt/lists/*

# FIX 2 & 3: Consolidated Installation
# Install all dependencies (including TensorFlow) in one clean step
RUN pip install --no-cache-dir -r requirements.txt


# Copy the entire project code (including src/, models/, and data/) into the container
COPY . $APP_HOME

# Expose the port the FastAPI application runs on
EXPOSE 8000


# Command to run the application using Gunicorn for production robustness
# src.api.main:app points to the FastAPI instance
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]