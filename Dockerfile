# Use an official lightweight Python image.
FROM python:3.9-slim

# Install system dependencies (if any are required)
RUN apt-get update && apt-get install -y build-essential

# Set the working directory
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port Streamlit will run on (default is 8501)
EXPOSE 8501

# Command to run the Streamlit app; disable CORS if needed for external access
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
