FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Render expects the app to listen on $PORT
EXPOSE 10000

CMD ["python", "app.py"]
