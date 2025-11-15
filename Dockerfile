FROM tensorflow/tensorflow:2.17.0

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install --ignore-installed blinker && \
    pip install tf-keras==2.17.0 && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Render exposes a random port at runtime.
EXPOSE 10000

CMD ["python", "app.py"]
