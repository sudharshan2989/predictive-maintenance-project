# Use Python 3.11 slim (better wheel support than 3.13) 
FROM python:3.11-slim


# Set the working directory inside the container to /app
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# Upgrade pip, setuptools, and wheel to ensure latest wheel handling
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies listed in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt


COPY src/ ./src/

# Expose Streamlit default port
EXPOSE 8501

# Add healthcheck for container monitoring
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# Run the Streamlit app
ENTRYPOINT ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]

# NOTE: Disable XSRF protection for easier external access in order to make batch predictions
