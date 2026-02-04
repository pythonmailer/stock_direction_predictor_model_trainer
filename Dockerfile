# 1. Base Image: Use a lightweight Python 3.10 image (matches your CI pipeline)
FROM python:3.10-slim

# 2. Environment Variables
# Prevents Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE=1
# Prevents Python from buffering stdout and stderr (helps you see logs immediately)
ENV PYTHONUNBUFFERED=1

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Install System Dependencies
# 'git' is often needed for DVC or installing packages from repositories
# 'build-essential' helps compile C-extensions for libraries like Polars/Numpy if needed
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy Dependencies First (Caching Layer)
# We copy ONLY requirements.txt first. Docker caches this step.
# If you change your code but not requirements, Docker skips reinstalling libraries.
COPY requirements.txt .

# 6. Install Python Libraries
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 7. Copy the Application Code
# This copies everything from your project folder into /app inside the container
COPY . .

# 8. Expose the Port
# Streamlit runs on 8501 by default
EXPOSE 8501

# 9. Healthcheck (Optional but recommended)
# Tells Docker if the container is healthy or hung
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 10. Run the Application
# We specify the host as 0.0.0.0 so it is accessible outside the container
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]