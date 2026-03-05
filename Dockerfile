# Use the full Python image instead of -slim to ensure build tools are present
FROM python:3.11

WORKDIR /app

COPY requirements.txt .

# Upgrade pip first, then install requirements
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "-u", "app.py"]