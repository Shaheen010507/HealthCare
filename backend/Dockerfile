FROM python:3.10-slim

WORKDIR /app

# Copy dependency list
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and models
COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
