# Use a lightweight Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first (this caches dependencies to make future builds faster)
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# CRITICAL: Copy your actual application code INTO the container
COPY app.py .

# Command to run the application
CMD ["python", "app.py"]